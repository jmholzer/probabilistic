from typing import Optional, Tuple

from matplotlib import pyplot
from matplotlib.figure import Figure
from numpy import array

from probabilistic.core import calculate_quartiles

pyplot.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))


def generate_figure(
    density_function: Tuple[array],
    title: Optional[str] = None,
    *,
    quartiles: Optional[bool] = False
) -> Figure:
    """Create a Matplotlib Figure object of a PDF

    Useful for drawing a graph using Streamlit

    Returns:
        A Matplotlib Figure object of the generated graph
    """
    fig, ax = pyplot.subplots()
    ax.scatter(x=density_function[0], y=density_function[1], s=0.1)
    ax.set_title(title)

    if quartiles:
        quartile_bounds = calculate_quartiles(density_function)
        for k, v in quartile_bounds.items():
            ax.axvspan(xmin=0, xmax=k, ymin=0, ymax=v)

    return fig
