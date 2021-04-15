from matplotlib.widgets import cbook, AxesWidget
import numpy as np
import math
import locale
import itertools
import matplotlib as mpl
from matplotlib import transforms as mtransforms
from numbers import Number, Integral


class _DummyAxis:
    __name__ = "dummy"

    def __init__(self, minpos=0):
        self.dataLim = mtransforms.Bbox.unit()
        self.viewLim = mtransforms.Bbox.unit()
        self._minpos = minpos

    def get_view_interval(self):
        return self.viewLim.intervalx

    def set_view_interval(self, vmin, vmax):
        self.viewLim.intervalx = vmin, vmax

    def get_minpos(self):
        return self._minpos

    def get_data_interval(self):
        return self.dataLim.intervalx

    def set_data_interval(self, vmin, vmax):
        self.dataLim.intervalx = vmin, vmax

    def get_tick_space(self):
        # Just use the long-standing default of nbins==9
        return 9


class TickHelper:
    axis = None

    def set_axis(self, axis):
        self.axis = axis

    def create_dummy_axis(self, **kwargs):
        if self.axis is None:
            self.axis = _DummyAxis(**kwargs)

    def set_view_interval(self, vmin, vmax):
        self.axis.set_view_interval(vmin, vmax)

    def set_data_interval(self, vmin, vmax):
        self.axis.set_data_interval(vmin, vmax)

    def set_bounds(self, vmin, vmax):
        self.set_view_interval(vmin, vmax)
        self.set_data_interval(vmin, vmax)


class Formatter(TickHelper):
    """
    Create a string based on a tick value and location.
    """
    # some classes want to see all the locs to help format
    # individual ones
    locs = []

    def __call__(self, x, pos=None):
        """
        Return the format for tick value *x* at position pos.
        ``pos=None`` indicates an unspecified location.
        """
        raise NotImplementedError('Derived must override')

    def format_ticks(self, values):
        """Return the tick labels for all the ticks at once."""
        self.set_locs(values)
        return [self(value, i) for i, value in enumerate(values)]

    def format_data(self, value):
        """
        Return the full string representation of the value with the
        position unspecified.
        """
        return self.__call__(value)

    def format_data_short(self, value):
        """
        Return a short string version of the tick value.
        Defaults to the position-independent long value.
        """
        return self.format_data(value)

    def get_offset(self):
        return ''

    def set_locs(self, locs):
        """
        Set the locations of the ticks.
        This method is called before computing the tick labels because some
        formatters need to know all tick locations to do so.
        """
        self.locs = locs

    @staticmethod
    def fix_minus(s):
        """
        Some classes may want to replace a hyphen for minus with the proper
        unicode symbol (U+2212) for typographical correctness.  This is a
        helper method to perform such a replacement when it is enabled via
        :rc:`axes.unicode_minus`.
        """
        return (s.replace('-', '\N{MINUS SIGN}')
                if mpl.rcParams['axes.unicode_minus']
                else s)

    def _set_locator(self, locator):
        """Subclasses may want to override this to set a locator."""
        pass


class ScalarFormatter(Formatter):
    """
    Format tick values as a number.
    Parameters
    ----------
    useOffset : bool or float, default: :rc:`axes.formatter.useoffset`
        Whether to use offset notation. See `.set_useOffset`.
    useMathText : bool, default: :rc:`axes.formatter.use_mathtext`
        Whether to use fancy math formatting. See `.set_useMathText`.
    useLocale : bool, default: :rc:`axes.formatter.use_locale`.
        Whether to use locale settings for decimal sign and positive sign.
        See `.set_useLocale`.
    Notes
    -----
    In addition to the parameters above, the formatting of scientific vs.
    floating point representation can be configured via `.set_scientific`
    and `.set_powerlimits`).
    **Offset notation and scientific notation**
    Offset notation and scientific notation look quite similar at first sight.
    Both split some information from the formatted tick values and display it
    at the end of the axis.
    - The scientific notation splits up the order of magnitude, i.e. a
      multiplicative scaling factor, e.g. ``1e6``.
    - The offset notation separates an additive constant, e.g. ``+1e6``. The
      offset notation label is always prefixed with a ``+`` or ``-`` sign
      and is thus distinguishable from the order of magnitude label.
    The following plot with x limits ``1_000_000`` to ``1_000_010`` illustrates
    the different formatting. Note the labels at the right edge of the x axis.
    .. plot::
        lim = (1_000_000, 1_000_010)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'hspace': 2})
        ax1.set(title='offset_notation', xlim=lim)
        ax2.set(title='scientific notation', xlim=lim)
        ax2.xaxis.get_major_formatter().set_useOffset(False)
        ax3.set(title='floating point notation', xlim=lim)
        ax3.xaxis.get_major_formatter().set_useOffset(False)
        ax3.xaxis.get_major_formatter().set_scientific(False)
    """

    def __init__(self, useOffset=None, useMathText=None, useLocale=None):
        if useOffset is None:
            useOffset = mpl.rcParams['axes.formatter.useoffset']
        self._offset_threshold = \
            mpl.rcParams['axes.formatter.offset_threshold']
        self.set_useOffset(useOffset)
        self._usetex = mpl.rcParams['text.usetex']
        if useMathText is None:
            useMathText = mpl.rcParams['axes.formatter.use_mathtext']
        self.set_useMathText(useMathText)
        self.orderOfMagnitude = 0
        self.format = ''
        self._scientific = True
        self._powerlimits = mpl.rcParams['axes.formatter.limits']
        if useLocale is None:
            useLocale = mpl.rcParams['axes.formatter.use_locale']
        self._useLocale = useLocale

    def get_useOffset(self):
        """
        Return whether automatic mode for offset notation is active.
        This returns True if ``set_useOffset(True)``; it returns False if an
        explicit offset was set, e.g. ``set_useOffset(1000)``.
        See Also
        --------
        ScalarFormatter.set_useOffset
        """
        return self._useOffset

    def set_useOffset(self, val):
        """
        Set whether to use offset notation.
        When formatting a set numbers whose value is large compared to their
        range, the formatter can separate an additive constant. This can
        shorten the formatted numbers so that they are less likely to overlap
        when drawn on an axis.
        Parameters
        ----------
        val : bool or float
            - If False, do not use offset notation.
            - If True (=automatic mode), use offset notation if it can make
              the residual numbers significantly shorter. The exact behavior
              is controlled by :rc:`axes.formatter.offset_threshold`.
            - If a number, force an offset of the given value.
        Examples
        --------
        With active offset notation, the values
        ``100_000, 100_002, 100_004, 100_006, 100_008``
        will be formatted as ``0, 2, 4, 6, 8`` plus an offset ``+1e5``, which
        is written to the edge of the axis.
        """
        if val in [True, False]:
            self.offset = 0
            self._useOffset = val
        else:
            self._useOffset = False
            self.offset = val

    useOffset = property(fget=get_useOffset, fset=set_useOffset)

    def get_useLocale(self):
        """
        Return whether locale settings are used for formatting.
        See Also
        --------
        ScalarFormatter.set_useLocale
        """
        return self._useLocale

    def set_useLocale(self, val):
        """
        Set whether to use locale settings for decimal sign and positive sign.
        Parameters
        ----------
        val : bool or None
            *None* resets to :rc:`axes.formatter.use_locale`.
        """
        if val is None:
            self._useLocale = mpl.rcParams['axes.formatter.use_locale']
        else:
            self._useLocale = val

    useLocale = property(fget=get_useLocale, fset=set_useLocale)

    def _format_maybe_minus_and_locale(self, fmt, arg):
        """
        Format *arg* with *fmt*, applying unicode minus and locale if desired.
        """
        return self.fix_minus(locale.format_string(fmt, (arg,), True)
                              if self._useLocale else fmt % arg)

    def get_useMathText(self):
        """
        Return whether to use fancy math formatting.
        See Also
        --------
        ScalarFormatter.set_useMathText
        """
        return self._useMathText

    def set_useMathText(self, val):
        r"""
        Set whether to use fancy math formatting.
        If active, scientific notation is formatted as :math:`1.2 \times 10^3`.
        Parameters
        ----------
        val : bool or None
            *None* resets to :rc:`axes.formatter.use_mathtext`.
        """
        if val is None:
            self._useMathText = mpl.rcParams['axes.formatter.use_mathtext']
        else:
            self._useMathText = val

    useMathText = property(fget=get_useMathText, fset=set_useMathText)

    def __call__(self, x, pos=None):
        """
        Return the format for tick value *x* at position *pos*.
        """
        if len(self.locs) == 0:
            return ''
        else:
            xp = (x - self.offset) / (10. ** self.orderOfMagnitude)
            if abs(xp) < 1e-8:
                xp = 0
            return self._format_maybe_minus_and_locale(self.format, xp)

    def set_scientific(self, b):
        """
        Turn scientific notation on or off.
        See Also
        --------
        ScalarFormatter.set_powerlimits
        """
        self._scientific = bool(b)

    def set_powerlimits(self, lims):
        r"""
        Set size thresholds for scientific notation.
        Parameters
        ----------
        lims : (int, int)
            A tuple *(min_exp, max_exp)* containing the powers of 10 that
            determine the switchover threshold. For a number representable as
            :math:`a \times 10^\mathrm{exp}`` with :math:`1 <= |a| < 10`,
            scientific notation will be used if ``exp <= min_exp`` or
            ``exp >= max_exp``.
            The default limits are controlled by :rc:`axes.formatter.limits`.
            In particular numbers with *exp* equal to the thresholds are
            written in scientific notation.
            Typically, *min_exp* will be negative and *max_exp* will be
            positive.
            For example, ``formatter.set_powerlimits((-3, 4))`` will provide
            the following formatting:
            :math:`1 \times 10^{-3}, 9.9 \times 10^{-3}, 0.01,`
            :math:`9999, 1 \times 10^4`.
        See Also
        --------
        ScalarFormatter.set_scientific
        """
        if len(lims) != 2:
            raise ValueError("'lims' must be a sequence of length 2")
        self._powerlimits = lims

    def format_data_short(self, value):
        # docstring inherited
        if isinstance(value, np.ma.MaskedArray) and value.mask:
            return ""
        if isinstance(value, Integral):
            fmt = "%d"
        else:
            if getattr(self.axis, "__name__", "") in ["xaxis", "yaxis"]:
                if self.axis.__name__ == "xaxis":
                    axis_trf = self.axis.axes.get_xaxis_transform()
                    axis_inv_trf = axis_trf.inverted()
                    screen_xy = axis_trf.transform((value, 0))
                    neighbor_values = axis_inv_trf.transform(
                        screen_xy + [[-1, 0], [+1, 0]])[:, 0]
                else:  # yaxis:
                    axis_trf = self.axis.axes.get_yaxis_transform()
                    axis_inv_trf = axis_trf.inverted()
                    screen_xy = axis_trf.transform((0, value))
                    neighbor_values = axis_inv_trf.transform(
                        screen_xy + [[0, -1], [0, +1]])[:, 1]
                delta = abs(neighbor_values - value).max()
            else:
                # Rough approximation: no more than 1e4 divisions.
                a, b = self.axis.get_view_interval()
                delta = (b - a) / 1e4
            # If e.g. value = 45.67 and delta = 0.02, then we want to round to
            # 2 digits after the decimal point (floor(log10(0.02)) = -2);
            # 45.67 contributes 2 digits before the decimal point
            # (floor(log10(45.67)) + 1 = 2): the total is 4 significant digits.
            # A value of 0 contributes 1 "digit" before the decimal point.
            sig_digits = max(
                0,
                (math.floor(math.log10(abs(value))) + 1 if value else 1)
                - math.floor(math.log10(delta)))
            fmt = f"%-#.{sig_digits}g"
        return self._format_maybe_minus_and_locale(fmt, value)

    def format_data(self, value):
        # docstring inherited
        e = math.floor(math.log10(abs(value)))
        s = round(value / 10**e, 10)
        exponent = self._format_maybe_minus_and_locale("%d", e)
        significand = self._format_maybe_minus_and_locale(
            "%d" if s % 1 == 0 else "%1.10f", s)
        if e == 0:
            return significand
        elif self._useMathText or self._usetex:
            exponent = "10^{%s}" % exponent
            return (exponent if s == 1  # reformat 1x10^y as 10^y
                    else rf"{significand} \times {exponent}")
        else:
            return f"{significand}e{exponent}"

    def get_offset(self):
        """
        Return scientific notation, plus offset.
        """
        if len(self.locs) == 0:
            return ''
        s = ''
        if self.orderOfMagnitude or self.offset:
            offsetStr = ''
            sciNotStr = ''
            if self.offset:
                offsetStr = self.format_data(self.offset)
                if self.offset > 0:
                    offsetStr = '+' + offsetStr
            if self.orderOfMagnitude:
                if self._usetex or self._useMathText:
                    sciNotStr = self.format_data(10 ** self.orderOfMagnitude)
                else:
                    sciNotStr = '1e%d' % self.orderOfMagnitude
            if self._useMathText or self._usetex:
                if sciNotStr != '':
                    sciNotStr = r'\times\mathdefault{%s}' % sciNotStr
                s = r'$%s\mathdefault{%s}$' % (sciNotStr, offsetStr)
            else:
                s = ''.join((sciNotStr, offsetStr))

        return self.fix_minus(s)

    def set_locs(self, locs):
        # docstring inherited
        self.locs = locs
        if len(self.locs) > 0:
            if self._useOffset:
                self._compute_offset()
            self._set_order_of_magnitude()
            self._set_format()

    def _compute_offset(self):
        locs = self.locs
        # Restrict to visible ticks.
        vmin, vmax = sorted(self.axis.get_view_interval())
        locs = np.asarray(locs)
        locs = locs[(vmin <= locs) & (locs <= vmax)]
        if not len(locs):
            self.offset = 0
            return
        lmin, lmax = locs.min(), locs.max()
        # Only use offset if there are at least two ticks and every tick has
        # the same sign.
        if lmin == lmax or lmin <= 0 <= lmax:
            self.offset = 0
            return
        # min, max comparing absolute values (we want division to round towards
        # zero so we work on absolute values).
        abs_min, abs_max = sorted([abs(float(lmin)), abs(float(lmax))])
        sign = math.copysign(1, lmin)
        # What is the smallest power of ten such that abs_min and abs_max are
        # equal up to that precision?
        # Note: Internally using oom instead of 10 ** oom avoids some numerical
        # accuracy issues.
        oom_max = np.ceil(math.log10(abs_max))
        oom = 1 + next(oom for oom in itertools.count(oom_max, -1)
                       if abs_min // 10 ** oom != abs_max // 10 ** oom)
        if (abs_max - abs_min) / 10 ** oom <= 1e-2:
            # Handle the case of straddling a multiple of a large power of ten
            # (relative to the span).
            # What is the smallest power of ten such that abs_min and abs_max
            # are no more than 1 apart at that precision?
            oom = 1 + next(oom for oom in itertools.count(oom_max, -1)
                           if abs_max // 10 ** oom - abs_min // 10 ** oom > 1)
        # Only use offset if it saves at least _offset_threshold digits.
        n = self._offset_threshold - 1
        self.offset = (sign * (abs_max // 10 ** oom) * 10 ** oom
                       if abs_max // 10 ** oom >= 10**n
                       else 0)

    def _set_order_of_magnitude(self):
        # if scientific notation is to be used, find the appropriate exponent
        # if using an numerical offset, find the exponent after applying the
        # offset. When lower power limit = upper <> 0, use provided exponent.
        if not self._scientific:
            self.orderOfMagnitude = 0
            return
        if self._powerlimits[0] == self._powerlimits[1] != 0:
            # fixed scaling when lower power limit = upper <> 0.
            self.orderOfMagnitude = self._powerlimits[0]
            return
        # restrict to visible ticks
        vmin, vmax = sorted(self.axis.get_view_interval())
        locs = np.asarray(self.locs)
        locs = locs[(vmin <= locs) & (locs <= vmax)]
        locs = np.abs(locs)
        if not len(locs):
            self.orderOfMagnitude = 0
            return
        if self.offset:
            oom = math.floor(math.log10(vmax - vmin))
        else:
            if locs[0] > locs[-1]:
                val = locs[0]
            else:
                val = locs[-1]
            if val == 0:
                oom = 0
            else:
                oom = math.floor(math.log10(val))
        if oom <= self._powerlimits[0]:
            self.orderOfMagnitude = oom
        elif oom >= self._powerlimits[1]:
            self.orderOfMagnitude = oom
        else:
            self.orderOfMagnitude = 0

    def _set_format(self):
        # set the format string to format all the ticklabels
        if len(self.locs) < 2:
            # Temporarily augment the locations with the axis end points.
            _locs = [*self.locs, *self.axis.get_view_interval()]
        else:
            _locs = self.locs
        locs = (np.asarray(_locs) - self.offset) / 10. ** self.orderOfMagnitude
        loc_range = np.ptp(locs)
        # Curvilinear coordinates can yield two identical points.
        if loc_range == 0:
            loc_range = np.max(np.abs(locs))
        # Both points might be zero.
        if loc_range == 0:
            loc_range = 1
        if len(self.locs) < 2:
            # We needed the end points only for the loc_range calculation.
            locs = locs[:-2]
        loc_range_oom = int(math.floor(math.log10(loc_range)))
        # first estimate:
        sigfigs = max(0, 3 - loc_range_oom)
        # refined estimate:
        thresh = 1e-3 * 10 ** loc_range_oom
        while sigfigs >= 0:
            if np.abs(locs - np.round(locs, decimals=sigfigs)).max() < thresh:
                sigfigs -= 1
            else:
                break
        sigfigs += 1
        self.format = '%1.' + str(sigfigs) + 'f'
        if self._usetex or self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format


class SliderBase(AxesWidget):
    """
    The base class for constructing Slider widgets. Not intended for direct
    usage.

    For the slider to remain responsive you must maintain a reference to it.
    """
    def __init__(self, ax, orientation, closedmin, closedmax,
                 valmin, valmax, valfmt, dragging, valstep):
        if ax.name == '3d':
            raise ValueError('Sliders cannot be added to 3D Axes')

        super().__init__(ax)

        self.orientation = orientation
        self.closedmin = closedmin
        self.closedmax = closedmax
        self.valmin = valmin
        self.valmax = valmax
        self.valstep = valstep
        self.drag_active = False
        self.valfmt = valfmt

        if orientation == "vertical":
            ax.set_ylim((valmin, valmax))
            axis = ax.yaxis
        else:
            ax.set_xlim((valmin, valmax))
            axis = ax.xaxis

        self._fmt = axis.get_major_formatter()
        if not isinstance(self._fmt, ScalarFormatter):
            self._fmt = ScalarFormatter()
            self._fmt.set_axis(axis)
        self._fmt.set_useOffset(False)  # No additive offset.
        self._fmt.set_useMathText(True)  # x sign before multiplicative offset.

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_navigate(False)
        self.connect_event("button_press_event", self._update)
        self.connect_event("button_release_event", self._update)
        if dragging:
            self.connect_event("motion_notify_event", self._update)
        self._observers = cbook.CallbackRegistry()

    def _stepped_value(self, val):
        """Return *val* coerced to closest number in the ``valstep`` grid."""
        if isinstance(self.valstep, Number):
            val = (self.valmin
                   + round((val - self.valmin) / self.valstep) * self.valstep)
        elif self.valstep is not None:
            valstep = np.asanyarray(self.valstep)
            if valstep.ndim != 1:
                raise ValueError(
                    f"valstep must have 1 dimension but has {valstep.ndim}"
                )
            val = valstep[np.argmin(np.abs(valstep - val))]
        return val

    def disconnect(self, cid):
        """
        Remove the observer with connection id *cid*.

        Parameters
        ----------
        cid : int
            Connection id of the observer to be removed.
        """
        self._observers.disconnect(cid)

    def reset(self):
        """Reset the slider to the initial value."""
        if self.val != self.valinit:
            self.set_val(self.valinit)


class RangeSlider(SliderBase):
    """
    A slider representing a range of floating point values. Defines the min and
    max of the range via the *val* attribute as a tuple of (min, max).

    Create a slider that defines a range contained within [*valmin*, *valmax*]
    in axes *ax*. For the slider to remain responsive you must maintain a
    reference to it. Call :meth:`on_changed` to connect to the slider event.

    Attributes
    ----------
    val : tuple of float
        Slider value.
    """

    def __init__(
            self,
            ax,
            label,
            valmin,
            valmax,
            valinit=None,
            valfmt=None,
            closedmin=True,
            closedmax=True,
            dragging=True,
            valstep=None,
            orientation="horizontal",
            **kwargs,
    ):
        """
        Parameters
        ----------
        ax : Axes
            The Axes to put the slider in.

        label : str
            Slider label.

        valmin : float
            The minimum value of the slider.

        valmax : float
            The maximum value of the slider.

        valinit : tuple of float or None, default: None
            The initial positions of the slider. If None the initial positions
            will be at the 25th and 75th percentiles of the range.

        valfmt : str, default: None
            %-format string used to format the slider values.  If None, a
            `.ScalarFormatter` is used instead.

        closedmin : bool, default: True
            Whether the slider interval is closed on the bottom.

        closedmax : bool, default: True
            Whether the slider interval is closed on the top.

        dragging : bool, default: True
            If True the slider can be dragged by the mouse.

        valstep : float, default: None
            If given, the slider will snap to multiples of *valstep*.

        orientation : {'horizontal', 'vertical'}, default: 'horizontal'
            The orientation of the slider.

        Notes
        -----
        Additional kwargs are passed on to ``self.poly`` which is the
        `~matplotlib.patches.Rectangle` that draws the slider knob.  See the
        `.Rectangle` documentation for valid property names (``facecolor``,
        ``edgecolor``, ``alpha``, etc.).
        """
        super().__init__(ax, orientation, closedmin, closedmax,
                         valmin, valmax, valfmt, dragging, valstep)

        self.val = valinit
        if valinit is None:
            # Place at the 25th and 75th percentiles
            extent = valmax - valmin
            valinit = np.array(
                [valmin + extent * 0.25, valmin + extent * 0.75]
            )
        else:
            valinit = self._value_in_bounds(valinit)
        self.valinit = valinit
        if orientation == "vertical":
            self.poly = ax.axhspan(valinit[0], valinit[1], 0, 1, **kwargs)
        else:
            self.poly = ax.axvspan(valinit[0], valinit[1], 0, 1, **kwargs)

        if orientation == "vertical":
            self.label = ax.text(
                0.5,
                1.02,
                label,
                transform=ax.transAxes,
                verticalalignment="bottom",
                horizontalalignment="center",
            )

            self.valtext = ax.text(
                0.5,
                -0.02,
                self._format(valinit),
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="center",
            )
        else:
            self.label = ax.text(
                -0.02,
                0.5,
                label,
                transform=ax.transAxes,
                verticalalignment="center",
                horizontalalignment="right",
            )

            self.valtext = ax.text(
                1.02,
                0.5,
                self._format(valinit),
                transform=ax.transAxes,
                verticalalignment="center",
                horizontalalignment="left",
            )

        self.set_val(valinit)

    def _min_in_bounds(self, min):
        """Ensure the new min value is between valmin and self.val[1]."""
        if min <= self.valmin:
            if not self.closedmin:
                return self.val[0]
            min = self.valmin

        if min > self.val[1]:
            min = self.val[1]
        return self._stepped_value(min)

    def _max_in_bounds(self, max):
        """Ensure the new max value is between valmax and self.val[0]."""
        if max >= self.valmax:
            if not self.closedmax:
                return self.val[1]
            max = self.valmax

        if max <= self.val[0]:
            max = self.val[0]
        return self._stepped_value(max)

    def _value_in_bounds(self, val):
        return (self._min_in_bounds(val[0]), self._max_in_bounds(val[1]))

    def _update_val_from_pos(self, pos):
        """Update the slider value based on a given position."""
        idx = np.argmin(np.abs(self.val - pos))
        if idx == 0:
            val = self._min_in_bounds(pos)
            self.set_min(val)
        else:
            val = self._max_in_bounds(pos)
            self.set_max(val)

    def _update(self, event):
        """Update the slider position."""
        if self.ignore(event) or event.button != 1:
            return

        if event.name == "button_press_event" and event.inaxes == self.ax:
            self.drag_active = True
            event.canvas.grab_mouse(self.ax)

        if not self.drag_active:
            return

        elif (event.name == "button_release_event") or (
                event.name == "button_press_event" and event.inaxes != self.ax
        ):
            self.drag_active = False
            event.canvas.release_mouse(self.ax)
            return
        if self.orientation == "vertical":
            self._update_val_from_pos(event.ydata)
        else:
            self._update_val_from_pos(event.xdata)

    def _format(self, val):
        """Pretty-print *val*."""
        if self.valfmt is not None:
            return f"({self.valfmt % val[0]}, {self.valfmt % val[1]})"
        else:
            _, s1, s2, _ = self._fmt.format_ticks(
                [self.valmin, *val, self.valmax]
            )
            # fmt.get_offset is actually the multiplicative factor, if any.
            s1 += self._fmt.get_offset()
            s2 += self._fmt.get_offset()
            # Use f string to avoid issues with backslashes when cast to a str
            return f"({s1}, {s2})"

    def set_min(self, min):
        """
        Set the lower value of the slider to *min*.

        Parameters
        ----------
        min : float
        """
        self.set_val((min, self.val[1]))

    def set_max(self, max):
        """
        Set the lower value of the slider to *max*.

        Parameters
        ----------
        max : float
        """
        self.set_val((self.val[0], max))

    def set_val(self, val):
        """
        Set slider value to *val*.

        Parameters
        ----------
        val : tuple or array-like of float
        """
        val = np.sort(np.asanyarray(val))
        if val.shape != (2,):
            raise ValueError(
                f"val must have shape (2,) but has shape {val.shape}"
            )
        val[0] = self._min_in_bounds(val[0])
        val[1] = self._max_in_bounds(val[1])
        xy = self.poly.xy
        if self.orientation == "vertical":
            xy[0] = 0, val[0]
            xy[1] = 0, val[1]
            xy[2] = 1, val[1]
            xy[3] = 1, val[0]
            xy[4] = 0, val[0]
        else:
            xy[0] = val[0], 0
            xy[1] = val[0], 1
            xy[2] = val[1], 1
            xy[3] = val[1], 0
            xy[4] = val[0], 0
        self.poly.xy = xy
        self.valtext.set_text(self._format(val))
        if self.drawon:
            self.ax.figure.canvas.draw_idle()
        self.val = val
        if self.eventson:
            self._observers.process("changed", val)

    def on_changed(self, func):
        """
        Connect *func* as callback function to changes of the slider value.

        Parameters
        ----------
        func : callable
            Function to call when slider is changed. The function
            must accept a numpy array with shape (2,) as its argument.

        Returns
        -------
        int
            Connection id (which can be used to disconnect *func*).
        """
        return self._observers.connect('changed', lambda val: func(val))