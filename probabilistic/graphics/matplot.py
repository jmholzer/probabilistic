from datetime import datetime
from typing import Optional, Tuple, Union

from matplotlib import pyplot
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from numpy import array, linspace

from probabilistic.core import calculate_quartiles

from labellines import labelLines


import warnings

pyplot.rcParams['axes.autolimit_mode'] = 'round_numbers'


def generate_cdf_figure(
    density_function: Tuple[array],
    security_ticker: str,
    estimate_date: datetime,
    *,
    current_price: Optional[Union[float, bool]] = False,
    quartiles: Optional[bool] = False,
) -> Figure:
    """Create a Matplotlib Figure object of a PDF

    Useful for drawing a graph using Streamlit

    Returns:
        A Matplotlib Figure object of the generated graph
    """
    fig, ax = pyplot.subplots()
    ax.plot(density_function[0], density_function[1])
    ax.set_title(
        f"Cumulative Density Function of the price of"
        f"\n{security_ticker} on {estimate_date}"
    )

    # add axis titles
    ax.set_xlabel(f"Price")
    ax.set_ylabel("Probability")

    # format x-axis
    ax.tick_params(axis='x', which='minor', bottom=False)

    # format y-axis
    ax.set_ylim([0, 1])
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.set_yticks(linspace(0, 1, 11))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    if current_price:
        pass
    if quartiles:
        quartile_bounds = calculate_quartiles(density_function)
        x_start, x_end = ax.get_xlim()
        y_start, y_end = ax.get_ylim()
        for k, v in quartile_bounds.items():
            xmax = (v - x_start) / (x_end - x_start)
            ymax = (k - y_start) / (y_end - y_start)
            ax.axvline(x=v, ymax=ymax, color="black", linestyle="--", label=f"{v:.0f}")
            ax.axhline(y=k, xmax=xmax, color="black", linestyle="--")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labelLines(ax.get_lines(), align=False, zorder=2.5)

    return fig
