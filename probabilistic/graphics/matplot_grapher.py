from matplotlib import pyplot

from probabilistic.graphics import AbstractGrapher

from typing import Any


class MatplotGrapher(AbstractGrapher):
    """Implementation of a Grapher that renders a PDF using matplotlib

    Attributes:
        _pdf: a tuple of arrays containing PDF data (X, y)
        _title: the title of the graph to draw

    Methods:
        draw_pdf
        generate_pdf_figure
    """

    def draw_pdf(self) -> None:
        """Draw a PDF

        Returns:
            None
        """
        pyplot.scatter(x=self._pdf[0], y=self._pdf[1], s=0.1)
        pyplot.title(self._title)
        pyplot.show()

    def generate_pdf_figure(self) -> Any:
        """Create a Matplotlib Figure object of a PDF

        Useful for drawing a graph using Streamlit

        Returns:
            A Matplotlib Figure object of the generated graph
        """
        fig, ax = pyplot.subplots()
        ax.scatter(x=self._pdf[0], y=self._pdf[1], s=0.1)
        return fig
