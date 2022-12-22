from click.testing import CliRunner

from probabilistic.cli import cli, csv_runner


def test_calculate_csv(mocker):
    runner = CliRunner()
    mocker.patch("probabilistic.cli.csv_runner.run")
    runner.invoke(cli.calculate, ["--csv", "dummy_path"])
    csv_runner.run.assert_called_once_with("dummy_path")
