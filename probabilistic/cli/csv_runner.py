from probabilistic.core import calculate_pdf, calculate_cdf
from probabilistic.io import CSVReader
import pandas as pd
from traitlets import Bool
from typing import Optional


def run(input_csv_path: str, current_price: float, days_forward: int, save_to_csv: Bool, output_csv_path: Optional[str] = None) -> None:
    """Run probabilistic using the data from a CSV file as input

    Args:
        input_csv_path: the path of the input CSV file
        current_price: the current price of the security
        days_forward: the number of days in the future to estimate the
            price probability density at
        save_to_csv: whether to save result as a csv. If False, then function will return a DataFrame
        output_csv_path: the path to save the output csv results 

    Returns:
        the output csv results containing 3 columns: Price, PDF, CDF
    """
    reader = CSVReader()
    options_data = reader.read(input_csv_path)
    pdf_point_arrays = calculate_pdf(options_data, current_price, days_forward)
    cdf_point_arrays = calculate_cdf(pdf_point_arrays)

    priceP, densityP = pdf_point_arrays
    priceC, densityC = cdf_point_arrays

    # Convert results to DataFrame
    df = pd.DataFrame({
        'Price': priceP,
        'PDF': densityP,
        'CDF': densityC
    })

    # Write the DataFrame to a CSV file
    if save_to_csv:
        df.to_csv(output_csv_path, index=False)
    else:
        return df