from tkinter import filedialog
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
try:  # try importing RangeSlider from matplotlib package
    from matplotlib.widgets import RangeSlider
except:  # ERROR: old matplotlib. using copied one to support Pydroid
    print('WARNING: matplotlib is old!')
    from mpl341_compat import RangeSlider
from typing import TextIO, Dict, List, Tuple, Union, Iterable
from scipy import optimize
from scipy.signal import savgol_filter
from statistics import median
from functools import partial
import sys
import time
import scipy.integrate as integration
import matplotlib
matplotlib.use('TkAgg')


class Shot:
    def __init__(self, shot_time: int, espresso_values: Dict[str, Union[List[float], float]]):
        self.shot_time = time.ctime(shot_time)
        self.time: List[float] = espresso_values['espresso_elapsed']
        self.pressure: List[float] = espresso_values['espresso_pressure']
        self.flow: List[float] = espresso_values['espresso_flow']
        self.weight: List[float] = espresso_values['espresso_flow_weight']
        self.bw: float = espresso_values['drink_weight']
        if 'drink_tds' in espresso_values and espresso_values['drink_tds'] > 0.1:
            self.tds: float = espresso_values['drink_tds']
        else:
            self.tds = 10.0

    @staticmethod
    def parse(shot_file: TextIO):
        shot_time_raw = int(shot_file.readline().split()[1])  # first line should be clock.. plz!
        extr = Shot._extract_raw(shot_file)
        return Shot(shot_time_raw, extr)

    @staticmethod
    def _extract_raw(shot_file: TextIO) -> Dict[str, Union[List[float], float]]:
        labels = [  # (label_id_str, required_bool, vector_bool}
            ('espresso_elapsed {', True, True),
            ('espresso_flow {', True, True),
            ('espresso_flow_weight {', True, True),
            ('espresso_pressure {', True, True),
            ('drink_weight', True, False),
            ('drink_tds', False, False)
        ]
        data = dict()
        required_fields = len([filter(lambda l: l[1], labels)])
        required_count = 0
        shortest_vector_len = 999999
        for line in shot_file:
            line = line.strip()
            for label, required, vector in labels:
                if not line.startswith(label):
                    continue
                splits = line.replace('{', '').replace('}', '').split()
                key = splits[0]
                vals = list(float(v) for v in splits[1:])
                if required:
                    required_count += 1
                if vector:
                    data[key] = vals
                    if len(vals) < shortest_vector_len:
                        shortest_vector_len = len(vals)
                else:
                    data[key] = vals[0]
            if len(data) == len(labels):
                break

        # trim vector datasets, with some assertions
        if required_count < required_fields:
            print('WARNING: shot file not extracted properly!')
            return dict()

        def trim_vector(item):
            k, v = item
            if isinstance(v, list):
                return k, v[:shortest_vector_len]
            else:
                return k, v
        return dict(map(trim_vector, data.items()))


class Analysis:
    def __init__(self, s: Shot, weight_threshold: float = 0.5):
        self.shot = s

        smoothed_weight = self._smoothing(s.weight)
        self._stable_weight_t = self._stable_weight_time_begin(s.time, smoothed_weight, weight_threshold)
        self._initial_window = (self._stable_weight_t + 0.5, s.time[-1] - 0.5)
        self._tds_effect = self._make_tds_effect()
        self._resistance = self._calculate_resistance()

        self._diffs = self._calculate_difference(s.time, s.flow, s.weight, self._stable_weight_t)

        self._calculate_optimal_preped = partial(self._estimate_optimal, s.time, s.flow, s.weight, self._tds_effect, self._diffs)
        self._corr_suggestion = self._calculate_optimal_preped(self._initial_window)

        # temp storage for redrawn objects
        self._main_fig = None
        self._x_axis = None
        self._window_fill = None
        self._flow_plt = None
        self._error_line = None
        self._sugg_line = None
        self._corr_line = None

    def show(self):
        # graph things...
        self._main_fig, self._x_axis = plt.subplots()
        self._x_axis.set_xlabel('seconds')
        plt.title('Shot at [%s] %.01fg, %.02f%% TDS' % (self.shot.shot_time, self.shot.bw, self.shot.tds))

        self._flow_plt, = plt.plot(self.shot.time, self.shot.flow, label='flow (ml/s)', color='blue', lw=3.5)
        plt.plot(self.shot.time, self.shot.weight, label='weight (g/s)', color='brown', lw=3.5)
        plt.plot(self.shot.time, self.shot.pressure, label='pressure (bar)', color='green')
        resistance, = plt.plot(self.shot.time, [0.0] * len(self.shot.time), label='resistance', color='yellow')
        resistance.set_ydata(self._resistance)
        plt.plot(self.shot.time, [v * 10 for v in self._diffs], label='difference (x10)', color='red')
        plt.plot(self.shot.time, [v * 10 for v in self._tds_effect], label='TDS weight (g/s, x10)', color='navy')

        self._error_line, = plt.plot(self.shot.time, [0.0] * len(self.shot.time), label='error (x10)', color='grey')
        error_data = self._error_series(self.shot.flow, self._stable_weight_t, 10)
        self._error_line.set_ydata(error_data)  # little hack to clip large error values

        self._sugg_line = plt.axhline(self._corr_suggestion * 10, label='suggestion (x10)', color='pink', linestyle='dashed')
        self._corr_line = plt.axhline(10.0, label='correction (x10)', color='magenta', linestyle='dotted')

        # sliders
        plt.subplots_adjust(right=0.85, bottom=0.18)
        # correction slider
        correction_ax = plt.axes([0.9, 0.2, 0.03, 0.65])
        correction_slider = Slider(correction_ax, 'correction\nvalue', orientation='vertical',
                                   valinit=1.0, valmin=0.3, valmax=2.5, valstep=0.01)

        correction_slider.on_changed(partial(Analysis._update_flow, self, self.shot.flow))

        # window slider
        window_ax = plt.axes([0.16, 0.05, 0.66, 0.03])
        window_slider = RangeSlider(window_ax, 'opt.\nwindow', valmin=0.0, valmax=self.shot.time[-1], valstep=0.1)
        window_slider.set_val(self._initial_window)
        window_slider.valtext.set_visible(False)

        # to be redrawn on slider move... I know it's dirty!!
        self._window_fill = self._x_axis.axvspan(*self._initial_window, ymin=0.0, ymax=1.0, alpha=0.15, color='green')
        window_slider.on_changed(partial(Analysis._update_window, self))

        self._x_axis.legend()
        plt.show()

    def _calculate_resistance(self):
        r = list()
        # using weight, not flow
        for _w, _p in zip(self._smoothing(self.shot.weight, 15), self._smoothing(self.shot.pressure)):
            if _w > 0:
                r.append(_p ** 0.5 / _w)
            else:
                r.append(0.0)
        return r

    def _error_series(self, target_flow: List[float], from_t: float, magnifier: int) -> List[float]:
        dat = list()
        for z in zip(self.shot.time, self.shot.weight, self._tds_effect, target_flow):
            dat.append(magnifier * abs(z[1] - z[2] - z[3]) if from_t <= z[0] else 0.0)
        return dat

    def _make_tds_effect(self, extraction_threshold_weight=0.2):
        estimated_tds_weight = self.shot.bw * self.shot.tds / 100.0
        tds_series = list()
        in_extraction = False
        times = list()
        weights = list()
        for t, p, w in zip(self._smoothing(self.shot.time), self._smoothing(self.shot.weight)):
            # collect weight vals after extraction phase starts
            if not in_extraction and extraction_threshold_weight <= w:
                in_extraction = True
            if not in_extraction:
                tds_series.append(0.0)
            else:
                times.append(t)
                weights.append(w)
        return tds_series + self._guessimate_to_tds_weight(times, weights, estimated_tds_weight)

    @staticmethod
    def _guessimate_to_tds_weight(t, w, target_tds_weight: float) -> List[float]:
        t_begin = t[0]
        t_end = t[-1]
        t_span = t_end - t_begin

        # TODO: use resistance curve to estimate puck degradation more precisely
        def puck_degradation(t: float) -> float:  # degradation = (begin) 1.0 ... 0.6 (end)
            return -0.4 / t_span * (t - t_begin) + 1.0
        integral_cummul = [0.0] + list(v for v in integration.cumtrapz(w, t))
        tds_weight_curve = list()
        for idx, (w0, w1) in enumerate(zip(integral_cummul[:-1], integral_cummul[1:])):
            dw = w1 - w0
            dt = t[idx + 1] - t[idx]
            ratio = dw / integral_cummul[-1]  # take ratio of tds in the current weight "shard"
            tds_weight_curve.append(ratio * target_tds_weight / dt * puck_degradation(t[idx]))
        tds_weight_curve = tds_weight_curve
        tds_weight_degraded = integration.simpson(tds_weight_curve, t[1:])
        weight_loss = tds_weight_degraded / target_tds_weight
        return Analysis._smoothing([0.0] + [w / weight_loss for w in tds_weight_curve])

    def _update_flow(self, base_flow: List[float], correction_val: float):
        new_flow = [val * correction_val for val in base_flow]
        self._flow_plt.set_ydata(new_flow)
        self._error_line.set_ydata(self._error_series(new_flow, self._stable_weight_t, 10))
        self._corr_line.set_ydata(correction_val * 10)
        self._main_fig.canvas.draw_idle()

    def _update_window(self, window_val: Tuple[float, float]):
        self._window_fill.remove()
        self._window_fill = self._x_axis.axvspan(*window_val, ymin=0.0, ymax=1.0, alpha=0.15, color='green')
        self._sugg_line.set_ydata(self._calculate_optimal_preped(window_val) * 10)
        self._main_fig.canvas.draw_idle()

    @staticmethod
    def _calculate_difference(t, f, w, from_t: float) -> List[float]:
        diffs = list()
        for z in zip(t, Analysis._smoothing(f), Analysis._smoothing(w, 11)):  # time, flow, weight
            if z[0] < from_t or z[1] < 0.1 or z[2] < 0.1:
                diffs.append(0.0)
            else:  # d = weight / flow
                diffs.append(z[2] / z[1])
        return diffs

    @staticmethod
    def _stable_weight_time_begin(t, w, weight_threshold) -> float:
        for z in zip(t, w):  # time, weight
            if z[1] > weight_threshold:
                return z[0]
        return t[0]  # failed

    @staticmethod
    def _estimate_optimal(t, f, w, tds_w, d, calc_range_t) -> float:
        def mse_windowed(flow_correction: float) -> float:
            calc_begin_t, calc_end_t = calc_range_t
            c = 0
            accum = 0.0
            for _, flow, weight, tds_weight in filter(lambda _z: calc_begin_t <= _z[0] < calc_end_t, zip(t, f, w, tds_w)):
                accum += (flow * flow_correction - (weight - tds_weight)) ** 2
                c += 1
            return accum / c
        diff_median = median(filter(lambda v: v > 0.01, d))
        opt = optimize.minimize(mse_windowed, diff_median, method='Nelder-Mead')
        if opt.status == 0:
            return opt.x
        else:
            print('WARNING: optimization failed. using diff median.')
            return diff_median

    @staticmethod
    def _smoothing(l: Iterable, win_len: int = 7, order: int = 3) -> List:
        return [v for v in savgol_filter(l, win_len, order)]


if __name__ == '__main__':
    file_name = filedialog.askopenfilename()
    with open(file_name, encoding='utf-8') as shot_file:
        if not shot_file.name.endswith('.shot'):
            print('%s doesn\'t seem like a proper shot file.' % shot_file.name)
            exit(2)
        s = Shot.parse(shot_file)
    if len(sys.argv) > 1:  # optionally first argument being a weight threshold
        Analysis(s, float(sys.argv[1])).show()
    else:
        Analysis(s).show()
