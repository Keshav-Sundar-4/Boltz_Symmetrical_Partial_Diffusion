from pathlib import Path

import yaml
from rdkit.Chem.rdchem import Mol

from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.types import Target


def parse_yaml(path: Path, ccd: dict[str, Mol]) -> tuple[Target, dict]:
    """Parse a Boltz input yaml / json.

    Parameters
    ----------
    path : Path
        Path to the YAML input format.
    components : Dict
        Dictionary of CCD components.

    Returns
    -------
    Target
        The parsed target.
    dict
        The raw YAML data.

    """
    with path.open("r") as file:
        data = yaml.safe_load(file)

    name = path.stem

    # Parse the Boltz schema
    target = parse_boltz_schema(name, data, ccd)
    return target, data

def get_symmetry_type(data: dict) -> str:
    """
    Extracts the symmetry type from the YAML data.

    Args:
        data: The raw YAML data as a dictionary.

    Returns:
        The symmetry type (str), or "NA" if not found.
    """
    return data.get("symmetry", "NA")

def get_radius(data: dict) -> float:
    """
    Extracts the radius from the YAML data.

    Args:
        data: The raw YAML data as a dictionary.

    Returns:
        The radius (float), or 0.0 if not found.
    """
    return data.get("radius", 0.0)
