from click.testing import CliRunner

from oipd.cli import cli, generate_pdf


def test_calculate_csv(mocker):
    runner = CliRunner()
    mocker.patch("oipd.cli.generate_pdf.run")
    runner.invoke(cli.calculate, ["--csv", "dummy_path"])
    generate_pdf.run.assert_called_once_with("dummy_path")
