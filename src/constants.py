"""
constants.py
------------
Module for loading universal and system constants from a JSON file.
"""

import json


def loadConstants(filePath: str) -> dict:
    """
    Load physical/system constants from a JSON file.

    Parameters
    ----------
    filePath : str
        Path to the JSON file containing constants.

    Returns
    -------
    dict
        Dictionary with constants and their values.

    Example
    -------
    constants = loadConstants("data/constants.json")
    print(constants["G"])  # prints 2.959122082855911e-4
    """
    with open(filePath, "r") as f:
        constants = json.load(f)
    return constants
