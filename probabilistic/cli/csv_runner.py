from probabilistic.core import calculate_pdf
from probabilistic.graphics import draw_figure
from probabilistic.io import CSVReader


def run(input_csv_path: str, current_price: float, days_forward: int) -> None:
    """Run probabilistic using the data from a CSV file as input

    Args:
        input_csv_path: the path of the input CSV file
        current_price: the current price of the security
        days_forward: the number of days in the future to estimate the
            price probability density at

    Returns:
        None
    """
    reader = CSVReader()
    options_data = reader.read(input_csv_path)
    pdf = calculate_pdf(options_data, current_price, days_forward)
    draw_figure(pdf)
