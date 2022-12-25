from typing import Iterable
import sys
import click
import subprocess
import shlex


def python_call(module: str, arguments: Iterable[str], **kwargs):
    """Run a subprocess command that invokes a Python module.

    Arguments:
        module: The module to invoke.
        arguments: The arguments to pass to the module.
        **kwargs: Additional keyword arguments to pass to subprocess.run.

    Raises:
        subprocess.CalledProcessError: If the subprocess call fails.
    """
    command = [sys.executable, "-m", module] + list(arguments)
    click.echo(" ".join(shlex.quote(cmd) for cmd in command))
    return_code = subprocess.run(command, **kwargs).returncode
    if return_code == 1:
        raise click.exceptions.Exit(code=return_code)
