from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import Button
from matplotlib.widgets import Slider
from matplotlib.widgets import RangeSlider
from typing import List, Tuple, Iterable
from scipy import optimize
from scipy.signal import savgol_filter
from scipy.signal import resample
from statistics import median
from functools import partial
import argparse
import scipy.integrate as integration
import matplotlib

from shot import Shot


def eq_within(v1, v2, margin=0.005):
    return v2 - margin < v1 < v2 + margin


class Analysis:
    def __init__(self, s: Shot, weight_threshold: float = 0.8, extraction_threshold: float = 0.2):
        self.shot = s

        smoothed_weight = self._smoothing(s.weight)
        self._stable_weight_t = self._stable_weight_time_begin(s.elapsed, smoothed_weight, weight_threshold)
        self._initial_window = (self._stable_weight_t + 0.5, s.elapsed[-1] - 1.5)
        self._resistance = self._calculate_resistance(extraction_threshold)

        try:  # using 'logged' gravimetric flow (same as before)
            self._tds_effect = self._make_tds_effect(False, extraction_threshold)
        except RuntimeError as e:
            print('WARNING: trying with raw cumulative weight')
            if hasattr(self.shot, 'weight_accum_raw'):  # using 'derived' gravimetric flow
                self._erroneous_weight = self.shot.weight  # backup old weight values
                self._tds_effect = self._make_tds_effect(True, extraction_threshold)
            else:
                raise e

        self._gravimetric_water = [w - tw for w, tw in zip(self.shot.weight, self._tds_effect)]
        self._diffs = self._calculate_difference(s.elapsed, s.flow, self._gravimetric_water, self._stable_weight_t)

        self._calculate_optimal_preped = partial(self._estimate_optimal, s.elapsed, s.flow, s.weight, self._tds_effect, self._diffs)
        self._corr_suggestion = self._calculate_optimal_preped(self._initial_window)

        # temp storage for redrawn objects
        self._main_fig = None
        self._x_axis = None
        self._window_fill = None
        self._flow_plt = None
        self._error_line = None
        self._sugg_line = None
        self._corr_line = None

    def show(self, with_controls: bool = True, verbose: bool = False):
        # graph things...
        self._main_fig, self._x_axis = plt.subplots()
        self._x_axis.set_xlabel('seconds')
        plt.title('Shot at [%s] %.01fg, %.02f%% TDS' % (self.shot.shot_time, self.shot.bw, self.shot.tds))

        time_series = self.shot.elapsed

        self._flow_plt, = plt.plot(time_series, self.shot.flow, label='flow (ml/s)', color='blue', lw=3.5)
        plt.plot(time_series, self.shot.pressure, label='pressure (bar)', color='green')

        # derived things
        if hasattr(self, '_erroneous_weight'):  # only shown when weight curve is derived from raw scale values
            plt.plot(time_series, self._erroneous_weight, label='error weight (g/s)', color='brown', alpha=0.6, linestyle='dashed')
        # tds_ratio = [(tw / w * 100) for tw, w in zip(self._tds_effect, self._smoothing(self.shot.weight, 21))]
        # plt.plot(time_series, tds_ratio, label='TDS ratio (%)')
        plt.plot(time_series, self._gravimetric_water, label='measured water (g/s)', color='brown', lw=2)

        self._error_line, = plt.plot(time_series, [0.0] * len(time_series), label='error (x10)', color='grey')
        error_data = self._error_series(self.shot.flow, self._stable_weight_t, 10)
        self._error_line.set_ydata(error_data)  # little hack to clip large error values

        self._sugg_line = plt.axhline(self._corr_suggestion * 10, label='suggestion (x10)', color='pink', linestyle='dashed')
        self._corr_line = plt.axhline(10.0, label='correction (x10)', color='magenta', linestyle='dotted')

        if verbose:
            plt.plot(time_series, self.shot.weight, label='weight (g/s)', color='orange', linestyle='dashed')
            resistance, = plt.plot(time_series, [0.0] * len(time_series), label='resistance', color='yellow')
            resistance.set_ydata(self._resistance)
            plt.plot(time_series, [v * 10 for v in self._diffs], label='difference (x10)', color='red')
            plt.plot(time_series, [v * 10 for v in self._tds_effect], label='TDS weight (g/s, x10)', color='navy', linestyle='dashed')

        self._window_fill = self._x_axis.axvspan(*self._initial_window, ymin=0.0, ymax=1.0, alpha=0.15, color='green')

        if with_controls:
            # sliders
            plt.subplots_adjust(right=0.85, bottom=0.18)
            # correction slider
            correction_ax = plt.axes([0.87, 0.18, 0.03, 0.65])
            correction_slider = Slider(correction_ax, 'correction\nvalue', orientation='vertical',
                                       valinit=1.0, valmin=0.3, valmax=2.5, valstep=0.01)
            if eq_within(self.shot.current_calibration, 1.0):
                corr_value_fmt = FuncFormatter(lambda v, p: 'x%.02f' % v)
            else:
                corr_value_fmt = FuncFormatter(lambda v, p: 'x%.03f\n(%.02f)' % (self.shot.current_calibration * v, v))
            correction_slider._fmt = corr_value_fmt

            correction_slider.on_changed(partial(Analysis._update_flow, self, self.shot.flow))

            # +- buttons
            plus_button_x = plt.axes([0.92, 0.27, 0.035, 0.06])
            minus_button_x = plt.axes([0.92, 0.2, 0.035, 0.06])
            plus_button = Button(plus_button_x, '▲')
            minus_button = Button(minus_button_x, '▼')
            plus_button.on_clicked(lambda _: correction_slider.set_val(correction_slider.val + 0.01))
            minus_button.on_clicked(lambda _: correction_slider.set_val(correction_slider.val - 0.01))

            # window slider
            window_ax = plt.axes([0.16, 0.05, 0.66, 0.03])
            window_slider = RangeSlider(window_ax, 'opt.\nwindow', valmin=0.0, valmax=time_series[-1], valstep=0.1)
            window_slider.set_val(self._initial_window)
            window_slider.valtext.set_visible(False)

            window_slider.on_changed(partial(Analysis._update_window, self))

        self._x_axis.legend()
        plt.show()

    def _calculate_resistance(self, extraction_threshold: float):
        r = list()
        # using weight, not flow
        for (_w, _p) in zip(self._smoothing(self.shot.weight, 15), self._smoothing(self.shot.pressure)):
            if _w > extraction_threshold:
                r.append(_p ** 0.5 / _w)
            else:
                r.append(0.0)
        return r

    def _error_series(self, target_flow: List[float], from_t: float, magnifier: int) -> List[float]:
        dat = list()
        for z in zip(self.shot.elapsed, self.shot.weight, self._tds_effect, target_flow):
            dat.append(magnifier * abs(z[1] - z[2] - z[3]) if from_t <= z[0] else 0.0)
        return self._smoothing(dat, 15)

    def _make_tds_effect(self, cindy_rescue: bool, extraction_threshold_weight):
        tds_series = list()
        in_extraction = False
        times = list()
        weights = list()
        resists = list()
        if cindy_rescue:
            scale_factor, scaled = self._try_scaling_to_beverage_weight(self.shot.elapsed, self.shot.weight, self.shot.bw)
            self.shot.weight = scaled
            print('scaling with x%.02f, now shot weighs %.02fg' % (scale_factor, integration.trapz(scaled, self.shot.elapsed)))
            if 0:  # old approximation method
                accum_restored = self._from_raw_scale_weight(self.shot.weight_accum_raw, self.shot.elapsed, self.shot.weight, scale_factor)
                self.shot.weight = accum_restored
        for t, w, r in zip(*self._smoothing((self.shot.elapsed, self.shot.weight)), self._resistance):
            # collect weight vals after extraction phase starts
            if not in_extraction and extraction_threshold_weight <= w:
                in_extraction = True
            if not in_extraction:
                tds_series.append(0.0)
            else:
                times.append(t)
                weights.append(w)
                resists.append(r)
        return tds_series + self._guessimate_to_tds_weight(times, weights, resists)

    @staticmethod
    def _try_scaling_to_beverage_weight(elapsed_time: List[float], gravimetric_flow: List[float], beverage_weight: float):
        scale_factor = beverage_weight / integration.trapz(gravimetric_flow, elapsed_time)
        return scale_factor, [scale_factor * v for v in gravimetric_flow]

    @staticmethod
    def _from_raw_scale_weight(raw_weight: List[float], time_elapsed_target: List[float], approximate_to: List[float], scale_hint: float) -> List[float]:
        """
        trimming: infer extraction phase, which best fits to the scaled erroneous gravimetric flow
        **: supposedly, heater test phase / flush phase
        --: extraction phase (including PI stages)
        ***---------------------------------------******** => raw, cumulative weight values
           <trim_begin ---------------- -trim_end>         => inferred extraction phase
           [-------------------------------------]         => scaled baseline (erroneous gravimetric flow)
        """
        COMPARE_CUMULATIVE_WEIGHT = False

        def sqerr_trimmed_scaled_weight(args: List):  # trim_samples_begin, -trim_samples_end, scale_factor
            trim_begin = round(args[0])
            trim_end = round(-args[1])
            scale = args[2]
            trimmed = raw_weight[trim_begin:trim_end]
            # downsample high freq weight samples to number of time samples
            downsampled = resample(trimmed, len(time_elapsed_target), window=40.0)
            scaled_cumulative = integration.cumtrapz([scale * v for v in approximate_to], time_elapsed_target)
            return sum((v0 - v1) ** 2 for (v0, v1) in zip(downsampled, scaled_cumulative))
        # minimize squared error between trimmed, downsampled cumulative weight <> scaled cumulative weight
        # adjusting trimming range(begin, end), and weight correction factor
        trim_opt = optimize.minimize(sqerr_trimmed_scaled_weight, [100, 35, scale_hint], method='Nelder-Mead')
        if trim_opt.status == 0:
            print('Raw gravimetric flow estimation: Trim[%.02f, -%.02f], x%.02f' % (trim_opt.x[0], trim_opt.x[1], trim_opt.x[2]))
        else:
            raise RuntimeError('Optimization for gravimetric flow estimation failed!')
        if COMPARE_CUMULATIVE_WEIGHT:
            scaled = [trim_opt.x[2] * v for v in approximate_to]
            plt.plot(time_elapsed_target, [0.0] + [v for v in integration.cumtrapz(scaled, time_elapsed_target)], label='weight scaled (g/s)')

        weight_trimmed = raw_weight[round(trim_opt.x[0]):round(-trim_opt.x[1])]
        # downsample inferred extraction phase then remove some FFT artifacts
        downsampled = list(map(lambda v: 0 if v < 0.05 else v, resample(weight_trimmed, len(time_elapsed_target), window=42.0)))

        if COMPARE_CUMULATIVE_WEIGHT:
            _t = list()
            for i in range(len(time_elapsed_target)):
                _t.append(time_elapsed_target[-1] / len(time_elapsed_target) * i)
            plt.plot(_t, downsampled, label='derived weight (g/s)')
            plt.legend()

        derivative = list()
        dt = time_elapsed_target[-1] / len(time_elapsed_target)
        for w0, w1 in zip(downsampled[:-1], downsampled[1:]):
            dw = max(0, w1 - w0)
            derivative.append(dw / dt)
        print('WARNING: shot now weighs %.02fg' % integration.trapz(derivative, dx=dt))
        return [0.0] + derivative

    def _guessimate_to_tds_weight(self, t, w, r) -> List[float]:
        target_tds_weight = self.shot.bw * self.shot.tds / 100.0
        t_begin = t[0]
        t_end = t[-1]
        t_span = t_end - t_begin

        def puck_degradation(t: float) -> float:
            return 1.0 - 0.7 / t_span * (t - t_begin)
        integral_cumul = [0.0] + list(v for v in integration.cumtrapz(w, t))
        if not eq_within(self.shot.bw, integral_cumul[-1], self.shot.bw * 0.2):
            raise RuntimeError('Too much error between beverage weight: %.02fg, cumulative weight: %.02fg' % (self.shot.bw, integral_cumul[-1]))

        tds_weight_curve = list()
        for idx, (w0, w1) in enumerate(zip(integral_cumul[:-1], integral_cumul[1:])):
            dw = w1 - w0
            dt = t[idx + 1] - t[idx]
            ratio = dw / integral_cumul[-1]  # take ratio of tds in the current weight "shard"
            tds_weight_curve.append(ratio * target_tds_weight / dt * (r[idx] ** 0.5) * puck_degradation(t[idx]))
        tds_weight_curve = tds_weight_curve
        tds_weight_degraded = integration.simpson(tds_weight_curve, t[1:])
        weight_scale = tds_weight_degraded / target_tds_weight
        return Analysis._smoothing([0.0] + [w / weight_scale for w in tds_weight_curve])

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
        for z in zip(t, *Analysis._smoothing((f, w))):  # time, flow, weight
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
    from tkinter import filedialog, simpledialog
    import tkinter
    matplotlib.use('TkAgg')

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', default=False,
                        help='make the plotting more verbose!')
    mutex_g = parser.add_mutually_exclusive_group()
    mutex_g.add_argument('--visualizer', dest='visualizer_url', action='store', nargs='?', const='?', default=None,
                        help='take shot history from visualizer.coffee')
    mutex_g.add_argument('--file', action='store', nargs='?', const='?', default=None,
                        help='take shot history from a file')
    args = parser.parse_args()

    if args.visualizer_url:
        if args.visualizer_url == '?':
            tkinter.Tk()
            url = simpledialog.askstring('shot URL', 'Enter URL of the shot: (https://...)\t\t\t\t\t')
        else:
            url = args.visualizer_url

        if not url.startswith('http://') and not url.startswith('https://'):
            raise RuntimeError('invalid shot url!')
        s = Shot.parse_visualizer(url)
        Analysis(s).show(True, args.verbose)
        exit(0)

    if args.file is None or args.file == '?':
        file_name = filedialog.askopenfilename()
    else:
        file_name = args.file
    with open(file_name, encoding='utf-8') as shot_file:
        ext = shot_file.name.split('.')[-1]
        if ext not in ['shot', 'json']:
            print('%s doesn\'t seem like a proper shot file.' % shot_file.name)
            exit(2)
        s = Shot.parse(shot_file)
        Analysis(s, args.verbose).show(True, args.verbose)
