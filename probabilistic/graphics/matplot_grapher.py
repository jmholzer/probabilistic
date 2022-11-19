from matplotlib import pyplot

from probabilistic.graphics import AbstractGrapher


class MatplotGrapher(AbstractGrapher):
    """Implementation of a Grapher that renders a PDF using matplotlib

    Attributes:
        _pdf: a tuple of arrays containing PDF data (X, y)
        _title: the title of the graph to draw

    Methods:
        draw
    """

    def draw(self) -> None:
        """Create a graph of a PDF

        Args:
            pdf: a tuple containing the x-axis values (index 0) and y-axis values
                (index 1) of the generated PDF

        Returns:
            None
        """
        pyplot.scatter(x=self._pdf[0], y=self._pdf[1][:2500], s=0.1)
        pyplot.title(self._title)
        pyplot.show()
