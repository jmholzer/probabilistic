from typing import List, Iterable
import sys
import click
import subprocess
import shlex


def call(cmd: List[str], **kwargs):  # pragma: no cover
    """Run a subprocess command and raise if it fails.
    Args:
        cmd: List of command parts.
        **kwargs: Optional keyword arguments passed to `subprocess.run`.
    Raises:
        click.exceptions.Exit: If `subprocess.run` returns non-zero code.
    """
    click.echo(" ".join(shlex.quote(c) for c in cmd))
    # pylint: disable=subprocess-run-check
    code = subprocess.run(cmd, **kwargs).returncode
    if code:
        raise click.exceptions.Exit(code=code)


def python_call(module: str, arguments: Iterable[str], **kwargs):  # pragma: no cover
    """Run a subprocess command that invokes a Python module."""
    call([sys.executable, "-m", module] + list(arguments), **kwargs)
