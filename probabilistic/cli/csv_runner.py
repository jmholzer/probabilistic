from probabilistic.io import CSVReader


def run(input_csv_path: str) -> None:
    """Run probabilistic using the data from a CSV file as input

    Args:
        input_csv_path: the path of the input CSV file

    Returns:
        None
    """
    reader = CSVReader()
    reader.read(input_csv_path)
