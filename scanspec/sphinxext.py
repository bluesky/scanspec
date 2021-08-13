from matplotlib.sphinxext.plot_directive import PlotDirective

from ._version_git import __version__


class ExampleSpecDirective(PlotDirective):
    def run(self):
        self.content = (
            ["# Example Spec", "", "from scanspec.plot import plot_spec"]
            + [str(x) for x in self.content]
            + ["plot_spec(spec)"]
        )
        return super().run()


def setup(app):
    app.add_directive("example_spec", ExampleSpecDirective)

    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
