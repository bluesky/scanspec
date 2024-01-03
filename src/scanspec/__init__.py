"""The top level scanspec module"""
from importlib.metadata import version

#: The version as calculated by setuptools_scm
__version__ = version("scanspec")
del version

__all__ = ["__version__", "core", "specs", "regions", "plot", "service"]
