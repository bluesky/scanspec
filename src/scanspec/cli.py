"""Interface for ``python -m scanspec``."""

import logging
import string

import click

# Need this so we can eval() below
from .regions import *  # noqa
from .specs import *  # noqa


@click.group(invoke_without_command=True)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(
        ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], case_sensitive=False
    ),
)
@click.version_option(prog_name="scanspec", message="%(version)s")
@click.pass_context
def cli(ctx: click.Context, log_level: str):
    """Top level scanspec command line interface."""
    level = getattr(logging, log_level.upper(), None)
    logging.basicConfig(format="%(levelname)s:%(message)s", level=level)

    # if no command is supplied, print the help message
    if ctx.invoked_subcommand is None:
        # We need to prove that cli has been converted to a command
        # by the click decorator to keep pyright happy.
        assert isinstance(cli, click.Command)
        click.echo(cli.get_help(ctx))


@cli.command()
@click.argument("spec")
def plot(spec: str):
    """Plot a scanspec."""
    from scanspec.plot import plot_spec

    for letter in string.ascii_lowercase:
        locals()[letter] = letter
    eval_spec = eval(spec)
    plot_spec(eval_spec)


@cli.command()
@click.option("--cors", is_flag=True)
@click.option(
    "--port", default=8080, help="The port that the scanspec service will be hosted on."
)
def service(cors: bool, port: int):
    """Run up a REST service."""
    from scanspec.service import run_app

    run_app(cors, port)


@cli.command()
def schema():
    """Print the OpenAPI schema for the service."""
    from scanspec.service import scanspec_schema_text

    click.echo(scanspec_schema_text())
