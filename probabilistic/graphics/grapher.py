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
    def draw_pdf(self) -> None:
        """Draw a graph of a PDF

        Returns:
            None
        """
        pass

    @abstractmethod
    def generate_pdf_figure(self):
        """Create a Matplotlib Figure object of a PDF

        Useful for drawing a graph using Streamlit

        Returns:
            A Matplotlib Figure object of the generated graph
        """
        pass
