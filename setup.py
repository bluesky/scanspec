# type: ignore
from setuptools import setup

setup(
    use_scm_version={
        "write_to": "src/scanspec/_version.py"
    },
    setup_requires=["setuptools>=45,<57", "setuptools_scm[toml]>=6.2", "wheel==0.33.1"],
)
