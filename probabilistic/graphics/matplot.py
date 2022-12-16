from matplotlib import pyplot
from matplotlib.figure import Figure
from typing import Tuple, Optional

from numpy import array


def draw_figure(density_function: Tuple[array], title: Optional[str] = None) -> None:
    """Draw a density function using matplotlib

    Returns:
        None
    """
    pyplot.scatter(x=density_function[0], y=density_function[1], s=0.1)
    pyplot.title(title)
    pyplot.show()


def generate_figure(density_function: Tuple[array], title: Optional[str] = None) -> Figure:
    """Create a Matplotlib Figure object of a PDF

    Useful for drawing a graph using Streamlit

    Returns:
        A Matplotlib Figure object of the generated graph
    """
    fig, ax = pyplot.subplots()
    ax.scatter(x=density_function[0], y=density_function[1], s=0.1)
    ax.set_title(title)
    return fig
