"""
reader.py
----------
Module responsible for reading planetary data from JSON files.
The data is used as input for secular perturbation calculations.
"""

import json
import pandas as pd


def loadPlanets(filePath: str) -> pd.DataFrame:
    """
    Load planetary data from a JSON file into a pandas DataFrame.

    Parameters
    ----------
    filePath : str
        Path to the JSON file containing planetary data.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the planet parameters with columns:
        ['name', 'mass', 'a', 'e', 'I', 'omega', 'Omega']

    Raises
    ------
    ValueError
        If the JSON structure is invalid or required fields are missing.
    """
    # Read JSON file
    with open(filePath, "r") as f:
        data = json.load(f)

    if "planets" not in data:
        raise ValueError("Invalid JSON format: missing 'planets' key.")

    # Convert list of planets to DataFrame
    df = pd.DataFrame(data["planets"])

    # Required fields
    requiredColumns = ["name", "mass", "a", "e", "I", "omega", "Omega"]

    for col in requiredColumns:
        if col not in df.columns:
            raise ValueError(f"Missing required field in planet data: {col}")

    # Ensure correct data types
    df = df.astype({
        "name": str,
        "mass": float,
        "a": float,
        "e": float,
        "I": float,
        "omega": float,
        "Omega": float
    })

    return df
