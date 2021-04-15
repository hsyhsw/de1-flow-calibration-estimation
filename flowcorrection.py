from tkinter import filedialog
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, RangeSlider
from typing import TextIO, Dict, List, Tuple
from scipy import optimize
from statistics import median
from functools import partial
import time
import matplotlib
matplotlib.use('TkAgg')


class Shot:
    def __init__(self, shot_time: int, espresso_values: Dict[str, List[float]]):
        self.shot_time = time.ctime(shot_time)
        self.time = espresso_values['espresso_elapsed']
        self.pressure = espresso_values['espresso_pressure']
        self.flow = espresso_values['espresso_flow']
        self.weight = espresso_values['espresso_flow_weight']

    @staticmethod
    def parse(shot_file: TextIO):
        shot_time_raw = int(shot_file.readline().split()[1])  # first line should be clock.. plz!
        extr = Shot._extract_raw(shot_file)
        return Shot(shot_time_raw, extr)

    @staticmethod
    def _extract_raw(shot_file: TextIO) -> Dict[str, List[float]]:
        target_labels = [
            'espresso_elapsed {',
            'espresso_flow {',
            'espresso_flow_weight {',
            'espresso_pressure {']
        data = {}
        for line in shot_file:
            if any(line.startswith(target) for target in target_labels):
                buf = line.replace('{', '')
                buf = buf.replace('}', '')
                splits = buf.split()
                label = splits[0]
                vals = list(float(v) for v in splits[1:])
                data[label] = vals
            if len(data) == len(target_labels):
                break

        # trim datasets, with some assertions
        if len(data) != len(target_labels):
            print('WARNING: shot file not extracted properly!')
            return dict()
        shortest_len = min([len(it) for it in data.values()])
        for k in data.keys():
            data[k] = data[k][:shortest_len]
        return data


class Analysis:
    def __init__(self, s: Shot):
        self._diffs = self._calculate_difference(s.time, s.flow, s.weight)

        stable_weight_t = self._stable_weight_time_begin(s.time, s.weight, 0.7)
        self._initial_window = (stable_weight_t, s.time[-1])

        self._calculate_optimal_preped = partial(self._estimate_optimal, s.time, s.flow, s.weight, self._diffs)
        self._corr_suggestion = self._calculate_optimal_preped(self._initial_window)

        # temp storage for redrawn objects
        self._main_fig = None
        self._x_axis = None
        self._window_fill = None
        self._flow_plt = None
        self._sugg_line = None
        self._corr_line = None

    def show(self):
        # graph things...
        self._main_fig, self._x_axis = plt.subplots()
        self._x_axis.set_xlabel('seconds')
        plt.title('Shot at [%s]' % s.shot_time)

        self._flow_plt, = plt.plot(s.time, s.flow, label='flow (ml/s)', color='blue', lw=3.5)
        plt.plot(s.time, s.weight, label='weight (g/s)', color='brown', lw=3.5)
        plt.plot(s.time, s.pressure, label='pressure (bar)', color='green')
        plt.plot(s.time, [v * 10 for v in self._diffs], label='difference (x10)', color='red')
        self._sugg_line = plt.axhline(self._corr_suggestion * 10, label='suggestion (x10)', color='pink', linestyle='dashed')
        self._corr_line = plt.axhline(10.0, label='correction (x10)', color='magenta', linestyle='dotted')

        # sliders
        plt.subplots_adjust(right=0.85, bottom=0.18)
        # correction slider
        correction_ax = plt.axes([0.9, 0.2, 0.03, 0.65])
        correction_slider = Slider(correction_ax, 'correction\nvalue', orientation='vertical',
                                   valinit=1.0, valmin=0.3, valmax=2.5, valstep=0.01)

        correction_slider.on_changed(partial(Analysis._update_flow, self, s.flow))

        # window slider
        window_ax = plt.axes([0.16, 0.05, 0.66, 0.03])
        try:
            window_slider = RangeSlider(window_ax, 'opt.\nwindow',
                                        valinit=self._initial_window, valmin=0.0, valmax=s.time[-1], valstep=0.1)
        except:  # FIXME: matplotlib bug: move slider to see the correct calculations
            print('ERROR: matplotlib bug: move slider to see the correct calculations.')
            window_slider = RangeSlider(window_ax, 'opt.\nwindow', valmin=0.0, valmax=s.time[-1], valstep=0.1)
        window_slider.valtext.set_visible(False)

        # to be redrawn on slider move... I know it's dirty!!
        self._window_fill = self._x_axis.axvspan(*self._initial_window, ymin=0.0, ymax=1.0, alpha=0.15, color='green')
        window_slider.on_changed(partial(Analysis._update_window, self))

        self._x_axis.legend()
        plt.show()

    def _update_flow(self, base_flow: List[float], correction_val: float):
        self._flow_plt.set_ydata([val * correction_val for val in base_flow])
        self._corr_line.set_ydata(correction_val * 10)
        self._main_fig.canvas.draw_idle()

    def _update_window(self, window_val: Tuple[float, float]):
        # global window_fill
        self._window_fill.remove()
        self._window_fill = self._x_axis.axvspan(*window_val, ymin=0.0, ymax=1.0, alpha=0.15, color='green')
        self._sugg_line.set_ydata(self._calculate_optimal_preped(window_val) * 10)
        self._main_fig.canvas.draw_idle()

    @staticmethod
    def _calculate_difference(t, f, w, calc_threshold=0.5) -> List[float]:
        diffs = list()
        for z in zip(t, f, w):  # time, flow, weight
            if z[1] < 0.1 or z[2] < 0.1:
                diffs.append(0.0)
            else:  # d = weight / flow
                if z[1] > calc_threshold and z[2] > calc_threshold:
                    d = z[2] / z[1]
                else:  # flow or weight is too low
                    d = 0.0
                diffs.append(d)
        return diffs

    @staticmethod
    def _stable_weight_time_begin(t, w, weight_threshold) -> float:
        for z in zip(t, w):  # time, weight
            if z[1] > weight_threshold:
                return z[0]
        return t[0]  # failed

    @staticmethod
    def _estimate_optimal(t, f, w, d, calc_t_window) -> float:
        def mse_windowed(flow_correction: float) -> float:
            c = 0
            accum = 0.0
            for z in filter(lambda _z: calc_t_window[0] <= _z[0] < calc_t_window[1], zip(t, f, w)):
                accum += (z[1] * flow_correction - z[2]) ** 2
                c += 1
            return accum / c
        diff_median = median(filter(lambda v: v > 0.01, d))
        opt = optimize.minimize(mse_windowed, diff_median, method='Nelder-Mead')
        if opt.status == 0:
            return opt.x
        else:
            print('WARNING: optimization failed. using diff median.')
            return diff_median


if __name__ == '__main__':
    file_name = filedialog.askopenfilename()
    with open(file_name) as shot_file:
        if not shot_file.name.endswith('.shot'):
            print('%s doesn\'t seem like a proper shot file.' % shot_file.name)
            exit(2)
        s = Shot.parse(shot_file)
    Analysis(s).show()
