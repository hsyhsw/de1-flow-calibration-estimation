from typing import Dict, Union, List, Any, TextIO, Tuple
import json
import time
import requests
import requests.status_codes


class Shot:
    DO_NOT_TRIM_KEY = 'DO_NOT_TRIM'  # XXX: tcl hack

    def __init__(self, shot_values: Dict[str, Union[List[Any], Any]]):
        if 'timestamp' in shot_values:
            if isinstance(shot_values['timestamp'], str):
                self.shot_time = shot_values['timestamp']
            else:
                self.shot_time = time.ctime(shot_values['timestamp'])
        else:
            self.shot_time = 'UNKNOWN'
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
    def _trim_vectors(extr: dict):
        """
        Trim vectors to match length of the shortest one.
        """
        shortest_len = min(map(lambda v: len(v), filter(lambda i: isinstance(i, list), extr.values())))

        def impl(item):
            k, v = item
            if isinstance(v, list):
                return k, v[:shortest_len]
            elif isinstance(v, dict):  # XXX: tcl hack
                return k, v[Shot.DO_NOT_TRIM_KEY]
            else:
                return k, v

        return dict(map(impl, extr.items()))

    @staticmethod
    def parse(shot_file: TextIO):
        if shot_file.name.endswith('.shot'):
            extr = Shot._extract_raw_tcl(shot_file)
        else:
            config = [  # (label_path_str, required_bool, rename_to_str, cast_to_type)
                ('timestamp', True, None, int),
                ('elapsed', True, None, float),
                ('flow.flow', True, 'flow', float),
                ('flow.by_weight', True, 'weight', float),
                ('pressure.pressure', True, 'pressure', float),
                ('app.data.settings.drink_weight', True, None, float),
                ('app.data.settings.drink_tds', True, None, float),
                ('app.data.settings.calibration_flow_multiplier', True, None, float)
            ]
            extr = Shot._extract_raw_json(json.load(shot_file), config)

        return Shot(Shot._trim_vectors(extr))

    @staticmethod
    def parse_visualizer(url: str):
        def _to_api_url(page_url: str) -> str:
            """
            https://visualizer.coffee/shots/{id} --> https://visualizer.coffee/api/shots/{id}/download
            """
            page_url = page_url.replace('/shots', '/api/shots')
            return page_url + '/download'

        resp = requests.get(_to_api_url(url))
        if not resp.ok:
            raise RuntimeError('failed to fetch shot at %s' % url)

        config = [  # (label_path_str, required_bool, rename_to_str, cast_to_type)
            ('start_time', False, 'timestamp', str),
            ('timeframe', True, 'elapsed', float),
            ('data.espresso_flow', True, 'flow', float),
            ('data.espresso_flow_weight', True, 'weight', float),
            ('data.espresso_pressure', True, 'pressure', float),
            ('drink_weight', True, None, float),
            ('drink_tds', False, None, float),
        ]
        extr = Shot._extract_raw_json(json.loads(resp.text), config)

        return Shot(Shot._trim_vectors(extr))

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
    def _extract_raw_json(shot_json: Dict[str, Any], extr_config: List[Tuple]) -> Dict[str, Union[List[Any], Any]]:
        """
        shot_json: dict-accessible JSON object
        extr_config: [(label_path_str, required_bool, rename_to_str, cast_to_type)]
        """
        def _resolve(o, path):
            for _id in path.split('.'):
                o = o[_id]
            return o

        data = dict()
        required_fields = len([filter(lambda l: l[1], extr_config)])
        required_count = 0
        for label, required, rename, cast_to in extr_config:
            key = label.split('.')[-1] if rename is None else rename
            try:
                val = _resolve(shot_json, label)
                if required:
                    required_count += 1
                if isinstance(val, list):
                    val = list(map(cast_to, val))
                else:
                    val = cast_to(val)
                data[key] = val
            except Exception as e:
                if required:
                    raise e
                else:
                    pass  # swallow it

        if required_count < required_fields:
            raise RuntimeError('shot data extraction failed!')

        return data
