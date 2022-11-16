import click

import csv_runner


@click.group(name="probabilistic")
def cli():
    """Defines a click group for the whole project
    """
    pass


@cli.command()
@click.option("--csv", "input_csv_path")
def calculate(input_csv_path: str) -> None:
    if input_csv_path:
        csv_runner.run(input_csv_path)


def main() -> None:
    cli()
