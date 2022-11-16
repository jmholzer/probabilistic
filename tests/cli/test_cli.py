from probabilistic.cli import cli
from probabilistic.cli import csv_runner

from click.testing import CliRunner


def test_calculate_csv(mocker):
    runner = CliRunner()
    mocker.patch('probabilistic.cli.csv_runner.run')
    runner.invoke(cli.calculate, ['--csv', "dummy_path"])
    csv_runner.run.assert_called_once_with("dummy_path")
