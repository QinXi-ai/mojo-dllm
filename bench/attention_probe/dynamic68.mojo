"""Separate executable so a rejected kernel cannot poison other cases."""
from probe import run


def main() raises:
    run[128, True]("dynamic68")
