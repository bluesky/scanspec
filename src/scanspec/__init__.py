from importlib.metadata import version

__version__ = version("scanspec")
del version

__all__ = ["__version__", "specs", "regions"]
