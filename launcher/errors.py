"""One exception type for every failure the user can act on."""

from __future__ import annotations


class LauncherError(Exception):
    """A failure with a message that names the thing, the cause and the next command."""

    code = 1


class UsageError(LauncherError):
    """The command line itself is wrong."""

    code = 2
