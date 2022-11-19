from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class AbstractGrapher(ABC):
    """Abstract class for Graphers -- objects of which create graphs
    of generated PDFS

    Attributes:
        _pdf: a tuple of arrays containing PDF data (X, y)
        _title: the title of the graph to draw

    Methods:
        draw
    """

    def __init__(self, pdf: Tuple[np.array], title: Optional[str] = None) -> None:
        self._pdf = pdf
        self._title = title

    @abstractmethod
    def draw(self, pdf: Tuple[np.array]) -> None:
        """Create a graph of a PDF

        Args:
            pdf: a tuple containing the x-axis values (index 0) and y-axis values
                (index 1) of the generated PDF

        Returns:
            None
        """
        pass
