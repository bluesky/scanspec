import subprocess
import sys

from scanspec import __version__


def test_cli_version():
    cmd = [sys.executable, "-m", "scanspec", "--version"]
    assert subprocess.check_output(cmd).decode().strip() == __version__
