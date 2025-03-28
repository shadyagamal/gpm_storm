import contextlib
import os
from importlib.metadata import PackageNotFoundError, version

# Get version
with contextlib.suppress(PackageNotFoundError):
    __version__ = version("gpm_api")
