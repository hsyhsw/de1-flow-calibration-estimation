from typing import Dict, Union, List, Any, TextIO
from xml.etree import ElementTree as et
from esprima.nodes import ExpressionStatement, AssignmentExpression
import esprima as js_parser
import re
import json
import time
import requests
import requests.status_codes


class Shot:
    DO_NOT_TRIM_KEY = 'DO_NOT_TRIM'

    def __init__(self, shot_values: Dict[str, Union[List[Any], Any]]):
        if isinstance(shot_values['timestamp'], str):
            self.shot_time = shot_values['timestamp']
        else:
            self.shot_time = time.ctime(shot_values['timestamp'])
        self.elapsed: List[float] = shot_values['elapsed']
        self.pressure: List[float] = shot_values['pressure']
        self.flow: List[float] = shot_values['flow']
        self.weight: List[float] = shot_values['weight']
        if 'weight_accum_raw' in shot_values:
            self.weight_accum_raw: List[float] = shot_values['weight_accum_raw']
        self.bw: float = shot_values['drink_weight']
        if 'drink_tds' in shot_values and shot_values['drink_tds'] > 0.1:
            self.tds: float = shot_values['drink_tds']
        else:
            self.tds = 10.0
        if 'calibration_flow_multiplier' in shot_values:
            self.current_calibration = shot_values['calibration_flow_multiplier']
        else:
            self.current_calibration = 1.0

    @staticmethod
    def parse(shot_file: TextIO):
        if shot_file.name.endswith('.shot'):
            extr = Shot._extract_raw_tcl(shot_file)
        else:
            extr = Shot._extract_raw_json(json.load(shot_file))

        # trim vector datasets
        shortest_vector_len = min(map(lambda v: len(v), filter(lambda i: isinstance(i, list), extr.values())))

        def trim_vector(item):
            k, v = item
            if isinstance(v, list):
                return k, v[:shortest_vector_len]
            elif isinstance(v, dict):
                return k, v[Shot.DO_NOT_TRIM_KEY]
            else:
                return k, v
        trimmed = dict(map(trim_vector, extr.items()))

        return Shot(trimmed)

    @staticmethod
    def parse_visualizer(url: str):
        resp = requests.get(url)
        if not resp.ok:
            raise RuntimeError('failed to fetch shot at %s' % url)
        extr = Shot._extract_visualizer_html(resp.text)
        extr['timestamp'] = next(filter(lambda p: len(p) != 0, reversed(url.split('/'))))
        return Shot(extr)

    @staticmethod
    def _extract_raw_tcl(shot_file: TextIO) -> Dict[str, Union[List[Any], Any]]:
        labels = [  # (label_id_str, required_bool, rename_to_str, cast_to_type, do_not_trim)
            ('clock', True, 'timestamp', int, False),
            ('espresso_elapsed {', True, 'elapsed', float, False),
            ('espresso_flow {', True, 'flow', float, False),
            ('espresso_flow_weight {', True, 'weight', float, False),
            ('scale_raw_weight {', False, 'weight_accum_raw', float, True),
            ('espresso_pressure {', True, 'pressure', float, False),
            ('drink_weight', True, None, float, False),
            ('drink_tds', False, None, float, False),
            ('calibration_flow_multiplier', False, None, float, False)
        ]
        data = dict()
        required_fields = len([filter(lambda l: l[1], labels)])
        required_count = 0
        for line in shot_file:
            line = line.strip()
            for label, required, rename, cast_to, no_trim in labels:
                if not line.startswith(label):
                    continue
                vector = '{' in label
                splits = line.replace('{', '').replace('}', '').split()
                key = splits[0] if rename is None else rename
                vals = list(map(cast_to, splits[1:]))
                if required:
                    required_count += 1
                if vector:
                    data[key] = vals
                else:
                    data[key] = vals[0]
                if no_trim:
                    current = data.get(key)
                    data[key] = {Shot.DO_NOT_TRIM_KEY: current}
            if len(data) == len(labels):
                break

        if required_count < required_fields:
            raise RuntimeError('shot file not extracted properly!')

        return data

    @staticmethod
    def _extract_raw_json(shot_json: Dict[str, Any]) -> Dict[str, Union[List[Any], Any]]:
        def resolve(o, path):
            for _id in path.split('.'):
                o = o[_id]
            return o

        labels = [  # (label_path_str, required_bool, rename_to_str, cast_to_type)
            ('timestamp', True, None, int),
            ('elapsed', True, None, float),
            ('flow.flow', True, 'flow', float),
            ('flow.by_weight', True, 'weight', float),
            ('pressure.pressure', True, 'pressure', float),
            ('app.data.settings.drink_weight', True, None, float),
            ('app.data.settings.drink_tds', True, None, float),
            ('app.data.settings.calibration_flow_multiplier', True, None, float)
        ]

        data = dict()
        required_fields = len([filter(lambda l: l[1], labels)])
        required_count = 0
        for label, required, rename, cast_to in labels:
            key = label.split('.')[-1] if rename is None else rename
            if required:
                required_count += 1
            val = resolve(shot_json, label)
            if isinstance(val, list):
                val = list(map(cast_to, val))
            else:
                val = cast_to(val)
            data[key] = val

        if required_count < required_fields:
            raise RuntimeError('shot file not extracted properly!')

        return data

    @staticmethod
    def _extract_visualizer_html(raw_html) -> Dict[str, Union[List[Any], Any]]:
        data = dict()
        raw_html = raw_html.replace('<br>', '\n')
        parsed_page = et.XML(raw_html)

        # shot metadata (pattern matching)
        output_filter_pat = re.compile('in\s+(\d+\.\d+)s')
        weight_pat = re.compile('(\d{1,2}(\.\d+){0,1}g:){0,1}(\d{1,2}(\.\d+){0,1})g{0,1}')
        tds_pat = re.compile('TDS\s+(\d{1,2}?(\.\d+){0,1})')
        weight_done = False
        tds_done = False
        for text_div in filter(lambda d: d.text and len(d.text) != 0, parsed_page.iter('div')):
            # <div>21.0g:46.8g (1:2.2) in 62.9s</div>
            # <div>41.7g in 35.1s</div>
            if output_filter_pat.search(text_div.text):
                weight_match = weight_pat.match(text_div.text)
                if weight_match:
                    weight_str = weight_match.group(3)
                    data['drink_weight'] = float(weight_str)
                    weight_done = True
            # <div>TDS 9.66%  EY 21.53% </div>
            tds_match = tds_pat.match(text_div.text)
            if tds_match:
                tds_str = tds_match.group(1)
                data['drink_tds'] = float(tds_str)
                tds_done = True

            if weight_done and tds_done:
                break

        # shot plotting data
        # <script> ... window.shotData = [{...}] ... </script>
        raw_json = None
        for script in filter(lambda e: e.text is not None, parsed_page.iter('script')):
            p = js_parser.parseScript(script.text, {'range': True})
            for stmt in p.body:
                if isinstance(stmt, ExpressionStatement) and isinstance(stmt.expression, AssignmentExpression):
                    lhs_b, lhs_e = stmt.expression.left.range
                    if script.text[lhs_b:lhs_e] == 'window.shotData':
                        rhs_b, rhs_e = stmt.expression.right.range
                        shot_data_raw = script.text[rhs_b:rhs_e]
                        raw_json = json.loads(shot_data_raw)
                        break
            if raw_json:
                break

        # time, pressure, flow, weight
        labels = {
            # time series from unpacked data pair
            'Pressure': 'pressure',
            'Flow': 'flow',
            'Weight Flow': 'weight'
        }
        time_s = list()
        need_time_series = True
        for label, label_to in labels.items():
            if label_to not in data:
                data[label_to] = list()
            for item in raw_json:
                if item['name'] == label:
                    for t_ms, val in item['data']:
                        if need_time_series:
                            time_s.append(t_ms / 1000.0)
                        data[label_to].append(0.0 if val is None else val)
                    need_time_series = False
        data['elapsed'] = time_s

        return data
