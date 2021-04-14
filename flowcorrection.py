from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, RangeSlider
from typing import TextIO, Dict, List
from scipy import optimize
from statistics import median
from functools import partial
import sys
import time
import matplotlib
matplotlib.use('TkAgg')


def extract_shot(shot_file: TextIO) -> Dict[str, List[float]]:
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

    # trim time stamps, with some assertions
    if len(data) != len(target_labels):
        print('WARNING: shot file not extracted properly!')
        return dict()
    shortest_len = min([len(it) for it in data.values()])
    data['espresso_elapsed'] = data['espresso_elapsed'][:shortest_len]
    data['espresso_flow'] = data['espresso_flow'][:shortest_len]
    data['espresso_flow_weight'] = data['espresso_flow_weight'][:shortest_len]
    data['espresso_pressure'] = data['espresso_pressure'][:shortest_len]
    return data


def calculate_difference(t, f, w, calc_threshold=0.5) -> List[float]:
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


def stable_weight_time_begin(t, w, weight_threshold) -> float:
    for z in zip(t, w):  # time, weight
        if z[1] > weight_threshold:
            return z[0]
    return t[0]  # failed


def estimate_optimal(t, f, w, d, calc_t_window) -> float:
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
    if len(sys.argv) != 2:
        print('give me .shot file!')
        exit(1)

    with open(sys.argv[1]) as shot_file:
        if not shot_file.name.endswith('.shot'):
            print('%s doesn\'t seem like a proper shot file.' % shot_file.name)
            exit(2)

        shot_time = time.ctime(int(shot_file.readline().split()[1]))  # first line should be clock.. plz!
        extr = extract_shot(shot_file)

    coffee_time = extr['espresso_elapsed']
    coffee_pressure = extr['espresso_pressure']
    coffee_flow = extr['espresso_flow']
    coffee_weight = extr['espresso_flow_weight']

    diffs = calculate_difference(coffee_time, coffee_flow, coffee_weight)
    stable_weight_t = stable_weight_time_begin(coffee_time, coffee_weight, 0.7)
    initial_window = (stable_weight_t, coffee_time[-1])

    calculate_optimal_preped = partial(estimate_optimal, coffee_time, coffee_flow, coffee_weight, diffs)
    corr_suggestion = calculate_optimal_preped(initial_window)

    # graph things...
    fig, ax = plt.subplots()
    ax.set_xlabel('seconds')
    plt.title('Shot at [%s]' % shot_time)

    flow_plt, = plt.plot(coffee_time, coffee_flow, label='flow (ml/s)', color='blue', lw=3.5)
    plt.plot(coffee_time, coffee_weight, label='weight (g/s)', color='brown', lw=3.5)
    plt.plot(coffee_time, coffee_pressure, label='pressure (bar)', color='green')
    plt.plot(coffee_time, [v * 10 for v in diffs], label='difference (x10)', color='red')
    sugg_line = plt.axhline(corr_suggestion * 10, label='suggestion (x10)', color='pink', linestyle='dashed')
    corr_line = plt.axhline(10.0, label='correction (x10)', color='magenta', linestyle='dotted')

    # sliders
    plt.subplots_adjust(right=0.85, bottom=0.18)
    # correction slider
    correction_ax = plt.axes([0.9, 0.2, 0.03, 0.65])
    correction_slider = Slider(correction_ax, 'correction\nvalue', orientation='vertical',
                               valinit=1.0, valmin=0.3, valmax=2.5, valstep=0.01)

    def update_flow(correction_val: float):
        flow_plt.set_ydata([val * correction_val for val in coffee_flow])
        corr_line.set_ydata(correction_val * 10)
        fig.canvas.draw_idle()
    correction_slider.on_changed(update_flow)

    # window slider
    window_ax = plt.axes([0.16, 0.05, 0.66, 0.03])
    try:
        window_slider = RangeSlider(window_ax, 'opt.\nwindow',
                                    valinit=initial_window, valmin=0.0, valmax=coffee_time[-1], valstep=0.1)
    except:  # FIXME: matplotlib bug: move slider to see the correct calculations
        print('ERROR: matplotlib bug: move slider to see the correct calculations.')
        window_slider = RangeSlider(window_ax, 'opt.\nwindow',
                                    valmin=0.0, valmax=coffee_time[-1], valstep=0.1)
    window_slider.valtext.set_visible(False)

    # to be redrawn on slider move
    window_fill = ax.axvspan(*initial_window, ymin=0.0, ymax=1.0, alpha=0.15, color='green')

    def update_window(window_val):
        global window_fill
        window_fill.remove()
        window_fill = ax.axvspan(*window_val, ymin=0.0, ymax=1.0, alpha=0.15, color='green')
        sugg_line.set_ydata(calculate_optimal_preped(window_val) * 10)
        sugg_line.set_label('asgasd')
        fig.canvas.draw_idle()
    window_slider.on_changed(update_window)

    ax.legend()
    plt.show()
