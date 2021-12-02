from contextlib import contextmanager

from matplotlib.sphinxext import plot_directive

from ._version_git import __version__


@contextmanager
def always_create_figures():
    """Force matplotlib PlotDirective to always create figures.

    This is needed even if source rst hasn't changed, as we often use
    example_spec from within docstrings
    """
    orig_f = plot_directive.out_of_date
    # Patch the plot directive so it thinks all sources are out of date
    plot_directive.out_of_date = lambda *args, **kwargs: True
    try:
        yield
    finally:
        plot_directive.out_of_date = orig_f


class ExampleSpecDirective(plot_directive.PlotDirective):
    """Runs `plot_spec` on the ``spec`` definied in the content."""

    def run(self):
        self.content = (
            ["# Example Spec", "", "from scanspec.plot import plot_spec"]
            + [str(x) for x in self.content]
            + ["plot_spec(spec)"]
        )
        with always_create_figures():
            return super().run()


def setup(app):
    """Setup this extension in sphinx."""
    app.add_directive("example_spec", ExampleSpecDirective)

    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
