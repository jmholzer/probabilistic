import pathlib

import click
from click_default_group import DefaultGroup

from . import csv_runner
from .utils import python_call


@click.group(
    cls=DefaultGroup, name="probabilistic", default="run", default_if_no_args=True
)
def cli():
    """Defines a click group for the whole project"""
    pass


@cli.command()
@click.option("--csv", "input_csv_path")
@click.option("--current-price", "current_price")
@click.option("--days-forward", "days_forward")
def calculate(input_csv_path: str, current_price: float, days_forward: int) -> None:
    """The CLI endpoint for running probabilistic end-to-end

    Args:
        input_csv_path: the path to the input CSV file
        current_price: the current price of the security
        days_forward: the number of days in the future to estimate the
            price probability density at

    Returns:
        None
    """
    if input_csv_path:
        # TODO: Get rid of this casting in a neat way
        csv_runner.run(input_csv_path, float(current_price), int(days_forward))


@cli.command()
def run() -> None:
    """The CLI endpoint for running the probabilistic interface"""
    root_path = pathlib.Path(__file__).parent.parent.resolve()
    interface_path = root_path / pathlib.Path("dashboard/interface.py")
    python_call("streamlit", ("run", str(interface_path)))


def main() -> None:
    cli()
