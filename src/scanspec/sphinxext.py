"""An example_spec directive."""

from contextlib import contextmanager
from typing import Any, cast

from docutils.statemachine import StringList
from matplotlib.sphinxext import plot_directive
from sphinx.application import Sphinx

from . import __version__


@contextmanager
def always_create_figures():
    """Force matplotlib PlotDirective to always create figures.

    This is needed even if source rst hasn't changed, as we often use
    example_spec from within docstrings
    """

    def always_true(
        original: Any,
        derived: Any,
        includes: Any = None,
    ) -> bool:
        return True

    # Type ignored because we never manipulate this object
    orig_f = plot_directive.out_of_date  # type: ignore
    # Patch the plot directive so it thinks all sources are out of date
    plot_directive.out_of_date = always_true
    try:
        yield
    finally:
        plot_directive.out_of_date = orig_f


class ExampleSpecDirective(plot_directive.PlotDirective):
    """Runs `plot_spec` on the ``spec`` definied in the content."""

    def run(self) -> Any:
        """Run the directive."""
        self.content = StringList(
            ["# Example Spec", "", "from scanspec.plot import plot_spec"]
            + [str(x) for x in self.content]
            + ["plot_spec(spec)"]
        )
        with always_create_figures():
            return cast(Any, super().run())


def setup(app: Sphinx):
    """Setup this extension in sphinx."""
    app.add_directive("example_spec", ExampleSpecDirective)

    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
