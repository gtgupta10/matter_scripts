"""
Step 3: Unified Surface Slab Generator — Flyte Workflows (FIXED VERSION)
=========================================================
A production-grade Flyte workflow mirroring the bulk builder architecture,
supporting multiple slab-generation modes with multi-termination sweeps.

FIXES:
- Better handling of hexagonal structures (wurtzite, etc.)
- Fallback mechanisms for difficult Miller indices
- Improved error handling and user guidance
- Automatic parameter adjustment for problematic orientations

Input Modes (UI JSON Examples):
  0. Shortcut: {"element": "Cu", "crystal": "fcc", "miller": "1 1 1",
                "vacuum": 15.0, "min_slab_size": 8.0}
  0b. Binary: {"prototype": "GaN", "elements": "Ga,N", "miller": "0 0 0 1",
               "vacuum": 15.0}
  1. Miller  : {"miller": "1 1 1", "input_file": "<POSCAR/CIF/extxyz>",
                "min_slab_size": 8.0, "vacuum": 15.0}
  2. MP/COD  : {"mp_id": "mp-30",  "miller": "1 0 0", "vacuum": 15.0}
               {"cod_id": 1010942, "miller": "1 1 0", "vacuum": 12.0}
               {"formula": "TiO2",   "miller": "1 1 0", "vacuum": 15.0,
                "max_e_hull": 0.05}

All miller-based modes share the same termination sweep:
  - symmetrize=True  → only symmetric slabs
  - all_terminations=True  → return every non-equivalent termination
  - shift override → pin a specific termination by fractional shift value

Outputs per slab:
  - POSCAR  (VASP format)
  - extxyz  (ASE extended XYZ)
  - CIF
  - JSON metadata  (formula, miller, termination index, shift, thickness,
                    surface_area, n_sites, spacegroup, vacuum, cell)
"""

from __future__ import annotations
from typing import NamedTuple, Optional, List, Dict
from dataclasses import dataclass, field
import os, gzip, base64, urllib.request, tempfile
import numpy as np
from flytekit import task, workflow, Resources
from flytekit.types.file import FlyteFile

# ── Materials Project API key ─────────────────────────────────────────────────
MP_API_KEY = os.environ.get("MP_API_KEY", "your_api_key_here")

# ── Shared task resources ─────────────────────────────────────────────────────
_RESOURCES = Resources(cpu="2", mem="8Gi", gpu="0")
_IMAGE = "guptag13/slab:v5"

# ── Single-element lattice param table (a, c) in Å — mirrors bulk builder ────
KNOWN_ALAT = {
    "Pd": (3.859, None), "Pt": (3.924, None), "Au": (4.078, None),
    "Ag": (4.086, None), "Cu": (3.615, None), "Ni": (3.524, None),
    "Al": (4.046, None), "Fe": (2.856, None), "W":  (3.165, None),
    "Mo": (3.147, None), "Rh": (3.803, None), "Ir": (3.840, None),
    "Ti": (2.951, 4.684), "Zr": (3.232, 5.148), "Mg": (3.209, 5.211),
    "Co": (2.507, 4.069), "Zn": (2.665, 4.947), "Ru": (2.706, 4.282),
    "Os": (2.734, 4.319), "Re": (2.761, 4.456), "Hf": (3.196, 5.051),
}

# ── Binary/multi-element prototype table ─────────────────────────────────────
KNOWN_BINARY = {
    # ── Rocksalt (SG 225, Fm-3m): A@4a(0,0,0)  B@4b(0.5,0.5,0.5) ───────
    "rocksalt":   {"sg":225, "n_species":2, "coords":[[0,0,0],[0.5,0.5,0.5]],                "a":4.210, "c":None},
    "NaCl":       {"sg":225, "n_species":2, "coords":[[0,0,0],[0.5,0.5,0.5]],                "a":5.640, "c":None},
    "MgO":        {"sg":225, "n_species":2, "coords":[[0,0,0],[0.5,0.5,0.5]],                "a":4.211, "c":None},
    "NiO":        {"sg":225, "n_species":2, "coords":[[0,0,0],[0.5,0.5,0.5]],                "a":4.177, "c":None},
    "FeO":        {"sg":225, "n_species":2, "coords":[[0,0,0],[0.5,0.5,0.5]],                "a":4.302, "c":None},
    "TiN":        {"sg":225, "n_species":2, "coords":[[0,0,0],[0.5,0.5,0.5]],                "a":4.240, "c":None},
    "ZrN":        {"sg":225, "n_species":2, "coords":[[0,0,0],[0.5,0.5,0.5]],                "a":4.585, "c":None},
    # ── Zinc-blende (SG 216, F-43m): A@4a(0,0,0)  B@4c(0.25,0.25,0.25) ─
    "zincblende": {"sg":216, "n_species":2, "coords":[[0,0,0],[0.25,0.25,0.25]],             "a":5.420, "c":None},
    "GaAs":       {"sg":216, "n_species":2, "coords":[[0,0,0],[0.25,0.25,0.25]],             "a":5.653, "c":None},
    "ZnS":        {"sg":216, "n_species":2, "coords":[[0,0,0],[0.25,0.25,0.25]],             "a":5.420, "c":None},
    "InP":        {"sg":216, "n_species":2, "coords":[[0,0,0],[0.25,0.25,0.25]],             "a":5.869, "c":None},
    # ── CsCl / B2 (SG 221, Pm-3m): A@1a(0,0,0)  B@1b(0.5,0.5,0.5) ──────
    "CsCl":       {"sg":221, "n_species":2, "coords":[[0,0,0],[0.5,0.5,0.5]],                "a":4.123, "c":None},
    "B2":         {"sg":221, "n_species":2, "coords":[[0,0,0],[0.5,0.5,0.5]],                "a":3.000, "c":None},
    # ── Fluorite (SG 225, Fm-3m): A@4a(0,0,0)  B@8c(0.25,0.25,0.25) ────
    "fluorite":   {"sg":225, "n_species":2, "coords":[[0,0,0],[0.25,0.25,0.25]],             "a":5.462, "c":None},
    "CaF2":       {"sg":225, "n_species":2, "coords":[[0,0,0],[0.25,0.25,0.25]],             "a":5.462, "c":None},
    "CeO2":       {"sg":225, "n_species":2, "coords":[[0,0,0],[0.25,0.25,0.25]],             "a":5.411, "c":None},
    "ZrO2":       {"sg":225, "n_species":2, "coords":[[0,0,0],[0.25,0.25,0.25]],             "a":5.070, "c":None},
    # ── Rutile (SG 136, P4₂/mnm): A@2a(0,0,0)  B@4f(u,u,0) ─────────────
    "rutile":     {"sg":136, "n_species":2, "coords":[[0,0,0],[0.305,0.305,0]],              "a":4.594, "c":2.959},
    "TiO2":       {"sg":136, "n_species":2, "coords":[[0,0,0],[0.305,0.305,0]],              "a":4.594, "c":2.959},
    "SnO2":       {"sg":136, "n_species":2, "coords":[[0,0,0],[0.307,0.307,0]],              "a":4.737, "c":3.186},
    "RuO2":       {"sg":136, "n_species":2, "coords":[[0,0,0],[0.306,0.306,0]],              "a":4.491, "c":3.107},
    "IrO2":       {"sg":136, "n_species":2, "coords":[[0,0,0],[0.305,0.305,0]],              "a":4.498, "c":3.154},
    # ── Wurtzite (SG 186, P6₃mc): A@2b(1/3,2/3,0)  B@2b(1/3,2/3,u) ─────
    "wurtzite":   {"sg":186, "n_species":2, "coords":[[1/3,2/3,0.0],[1/3,2/3,0.375]],       "a":3.250, "c":5.207},
    "ZnO":        {"sg":186, "n_species":2, "coords":[[1/3,2/3,0.0],[1/3,2/3,0.375]],       "a":3.250, "c":5.207},
    "GaN":        {"sg":186, "n_species":2, "coords":[[1/3,2/3,0.0],[1/3,2/3,0.375]],       "a":3.189, "c":5.185},
    "AlN":        {"sg":186, "n_species":2, "coords":[[1/3,2/3,0.0],[1/3,2/3,0.375]],       "a":3.111, "c":4.978},
    "InN":        {"sg":186, "n_species":2, "coords":[[1/3,2/3,0.0],[1/3,2/3,0.375]],       "a":3.545, "c":5.703},
    # ── Corundum (SG 167, R-3c): A@12c(0,0,z)  B@18e(x,0,0.25) ──────────
    "corundum":   {"sg":167, "n_species":2, "coords":[[0,0,0.352],[0.306,0,0.25]],           "a":4.758, "c":12.991},
    "Al2O3":      {"sg":167, "n_species":2, "coords":[[0,0,0.352],[0.306,0,0.25]],           "a":4.758, "c":12.991},
    "Fe2O3":      {"sg":167, "n_species":2, "coords":[[0,0,0.355],[0.306,0,0.25]],           "a":5.034, "c":13.752},
    "Cr2O3":      {"sg":167, "n_species":2, "coords":[[0,0,0.348],[0.306,0,0.25]],           "a":4.958, "c":13.584},
    # ── Perovskite (SG 221, Pm-3m): A@1a  B@1b  O@3c ─────────────────────
    "perovskite": {"sg":221, "n_species":3, "coords":[[0,0,0],[0.5,0.5,0.5],[0.5,0.5,0.0]], "a":3.905, "c":None},
    "SrTiO3":     {"sg":221, "n_species":3, "coords":[[0,0,0],[0.5,0.5,0.5],[0.5,0.5,0.0]], "a":3.905, "c":None},
    "BaTiO3":     {"sg":221, "n_species":3, "coords":[[0,0,0],[0.5,0.5,0.5],[0.5,0.5,0.0]], "a":4.006, "c":None},
    # ── L12 (SG 221, Pm-3m): A@3c(0.5,0.5,0)  B@1a(0,0,0) ───────────────
    "L12":        {"sg":221, "n_species":2, "coords":[[0.5,0.5,0.0],[0,0,0]],                "a":3.572, "c":None},
    "Ni3Al":      {"sg":221, "n_species":2, "coords":[[0.5,0.5,0.0],[0,0,0]],                "a":3.572, "c":None},
    "Pt3Al":      {"sg":221, "n_species":2, "coords":[[0.5,0.5,0.0],[0,0,0]],                "a":3.876, "c":None},
    # ── L10 (SG 123, P4/mmm): A@1a(0,0,0)  B@1c(0.5,0.5,0.5) ────────────
    "L10":        {"sg":123, "n_species":2, "coords":[[0,0,0],[0.5,0.5,0.5]],                "a":3.850, "c":3.713},
    "FePt":       {"sg":123, "n_species":2, "coords":[[0,0,0],[0.5,0.5,0.5]],                "a":3.850, "c":3.713},
    "CoPt":       {"sg":123, "n_species":2, "coords":[[0,0,0],[0.5,0.5,0.5]],                "a":3.810, "c":3.690},
    # ── NiAs (SG 194, P6₃/mmc): A@2a(0,0,0)  B@2c(1/3,2/3,0.25) ─────────
    "NiAs":       {"sg":194, "n_species":2, "coords":[[0,0,0],[1/3,2/3,0.25]],               "a":3.619, "c":5.034},
}


# ═════════════════════════════════════════════════════════════════════════════
# Input Dataclasses
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ShortcutSlabParams:
    """One-liner slab generation: element + crystal structure + Miller index."""
    element: str = "Cu"
    crystal: str = "fcc"
    miller: str = "1 1 1"
    alat: Optional[float] = None
    alat_c: Optional[float] = None
    min_slab_size: float = 8.0
    vacuum: float = 15.0
    center_slab: bool = True
    in_unit_planes: bool = False
    symmetrize: bool = True
    all_terminations: bool = True
    shift: Optional[float] = None
    max_normal_search: int = 2  # Increased default for better handling
    primitive: bool = True
    reorient_lattice: bool = True


@dataclass
class BinarySlabParams:
    """One-liner slab for binary/ternary compounds using KNOWN_BINARY table."""
    prototype:        str   = "TiO2"
    elements:         str   = "Ti,O"
    miller:           str   = "1 0 1"
    a:                Optional[float] = None
    c:                Optional[float] = None
    min_slab_size:    float = 10.0
    vacuum:           float = 15.0
    center_slab:      bool  = True
    in_unit_planes:   bool  = False
    symmetrize:       bool  = True
    all_terminations: bool  = True
    shift:            Optional[float] = None
    max_normal_search: int  = 2  # Increased default for better handling
    primitive:        bool  = True
    reorient_lattice: bool  = True


@dataclass
class MillerParams:
    """Cleave a slab from any structure file using Miller indices."""
    miller: str = "1 1 1"
    min_slab_size: float = 8.0
    vacuum: float = 15.0
    center_slab: bool = True
    in_unit_planes: bool = False
    symmetrize: bool = True
    all_terminations: bool = True
    shift: Optional[float] = None
    tol: float = 0.1
    max_normal_search: int = 2  # Increased default
    primitive: bool = True
    reorient_lattice: bool = True


@dataclass
class MPSlabParams:
    """Fetch bulk from Materials Project, then cleave with Miller indices."""
    mp_id: Optional[str] = None
    formula: Optional[str] = None
    is_stable: bool = False
    max_e_hull: float = 0.1
    conventional_cell: bool = True
    miller: str = "1 0 0"
    min_slab_size: float = 8.0
    vacuum: float = 15.0
    center_slab: bool = True
    in_unit_planes: bool = False
    symmetrize: bool = True
    all_terminations: bool = True
    shift: Optional[float] = None
    max_normal_search: int = 2  # Increased default
    primitive: bool = True
    reorient_lattice: bool = True


@dataclass
class CODSlabParams:
    """Fetch bulk from COD, then cleave with Miller indices."""
    cod_id: int = 1010942
    miller: str = "1 0 0"
    min_slab_size: float = 8.0
    vacuum: float = 15.0
    center_slab: bool = True
    in_unit_planes: bool = False
    symmetrize: bool = True
    all_terminations: bool = True
    shift: Optional[float] = None
    max_normal_search: int = 2  # Increased default
    primitive: bool = True
    reorient_lattice: bool = True


@dataclass
class SupercellOptions:
    """In-plane supercell expansion applied after slab generation."""
    nx: int = 1
    ny: int = 1
    supercell_matrix: Optional[str] = None


@dataclass
class SlabOutputConfig:
    """Output format and prefix settings."""
    output_prefix: str = "slab"
    output_formats: str = "extxyz,vasp,cif"
    write_all_terminations: bool = True


# ═════════════════════════════════════════════════════════════════════════════
# Output NamedTuple
# ═════════════════════════════════════════════════════════════════════════════

class SlabOutput(NamedTuple):
    """Primary output = first (or only) termination as FlyteFile."""
    primary_slab_file: FlyteFile
    all_termination_files: List[FlyteFile]
    termination_metadata: List[dict]
    formula: str
    num_atoms: int
    n_terminations: int
    miller: str
    slab_thickness: float
    surface_area: float
    spacegroup_info: str
    encoded_xyz: Dict[str, str]
    metadata: dict


# ═════════════════════════════════════════════════════════════════════════════
# Parse helpers
# ═════════════════════════════════════════════════════════════════════════════

def _parse_miller(miller_str: str) -> tuple:
    """Parse Miller indices, accepting both 3-index and 4-index notation."""
    parts = miller_str.replace(",", " ").split()
    if len(parts) == 4:
        h, k, i, l = (int(x) for x in parts)
        if i != -(h + k):
            print(f"  Warning: Miller-Bravais i={i} != -(h+k)={-(h+k)}, "
                  f"using (h,k,l)=({h},{k},{l})")
        return (h, k, l)
    elif len(parts) == 3:
        return tuple(int(x) for x in parts)
    else:
        raise ValueError(
            f"Miller indices must be 3 integers '1 1 1' or "
            f"4-index Miller-Bravais '0 0 0 1', got: '{miller_str}'"
        )


def _is_hexagonal_spacegroup(sg_number: int) -> bool:
    """Check if spacegroup is hexagonal/trigonal for special handling."""
    hexagonal_ranges = [
        (143, 167),  # Trigonal (R, H)
        (168, 194),  # Hexagonal
    ]
    return any(start <= sg_number <= end for start, end in hexagonal_ranges)


# ═════════════════════════════════════════════════════════════════════════════
# Core: pymatgen SlabGenerator wrapper with fallback mechanisms
# ═════════════════════════════════════════════════════════════════════════════

def _run_slabgenerator_with_fallback(
    structure,          # pymatgen Structure
    miller: tuple,
    min_slab_size: float,
    vacuum: float,
    center_slab: bool,
    in_unit_planes: bool,
    symmetrize: bool,
    all_terminations: bool,
    shift_override: Optional[float],
    max_normal_search: int,
    primitive: bool,
    reorient_lattice: bool,
) -> list:
    """
    Enhanced SlabGenerator with multiple fallback strategies.
    Returns a list of pymatgen Slab objects.
    """
    from pymatgen.core.surface import SlabGenerator
    
    # Strategy 1: Try with original parameters
    print(f"  Attempting slab generation for Miller {miller}...")
    
    strategies = [
        {
            "name": "Original parameters",
            "symmetrize": symmetrize,
            "max_normal_search": max_normal_search,
            "min_slab_size": min_slab_size,
            "in_unit_planes": in_unit_planes,
        },
        {
            "name": "Increased max_normal_search",
            "symmetrize": symmetrize,
            "max_normal_search": max(5, max_normal_search + 2),
            "min_slab_size": min_slab_size,
            "in_unit_planes": in_unit_planes,
        },
        {
            "name": "Larger min_slab_size",
            "symmetrize": symmetrize,
            "max_normal_search": max_normal_search,
            "min_slab_size": min_slab_size * 1.5,
            "in_unit_planes": in_unit_planes,
        },
        {
            "name": "Symmetrize=False",
            "symmetrize": False,
            "max_normal_search": max_normal_search,
            "min_slab_size": min_slab_size,
            "in_unit_planes": in_unit_planes,
        },
        {
            "name": "Unit planes mode",
            "symmetrize": symmetrize,
            "max_normal_search": max_normal_search,
            "min_slab_size": max(3, int(min_slab_size / structure.lattice.c * 2)),
            "in_unit_planes": True,
        },
    ]
    
    # Special handling for hexagonal structures with (0,0,1) orientation
    sg = structure.get_space_group_info()[1]
    if _is_hexagonal_spacegroup(sg) and miller == (0, 0, 1):
        print(f"  Detected hexagonal structure (SG {sg}) with (0,0,1) orientation")
        strategies.insert(1, {
            "name": "Hexagonal (0,0,1) special",
            "symmetrize": False,
            "max_normal_search": 10,
            "min_slab_size": max(15.0, min_slab_size * 2),
            "in_unit_planes": False,
        })
    
    for i, strategy in enumerate(strategies):
        try:
            print(f"    Strategy {i+1}: {strategy['name']}")
            
            sg = SlabGenerator(
                initial_structure=structure,
                miller_index=miller,
                min_slab_size=strategy["min_slab_size"],
                min_vacuum_size=vacuum,
                center_slab=center_slab,
                in_unit_planes=strategy["in_unit_planes"],
                max_normal_search=strategy["max_normal_search"],
                primitive=primitive,
                reorient_lattice=reorient_lattice,
            )

            if shift_override is not None:
                slab = sg.get_slab(shift=shift_override, tol=0.1)
                slabs = [slab] if slab is not None else []
            elif all_terminations:
                slabs = sg.get_slabs(symmetrize=strategy["symmetrize"])
            else:
                slabs = sg.get_slabs(symmetrize=strategy["symmetrize"])[:1]

            if slabs:
                print(f"    ✓ Success! Generated {len(slabs)} slab(s)")
                return slabs
            else:
                print(f"    ✗ No slabs generated")
                
        except Exception as e:
            print(f"    ✗ Failed: {str(e)[:100]}...")
            continue
    
    # If all strategies fail, provide helpful error message
    raise RuntimeError(
        f"Failed to generate slabs for Miller index {miller} after trying {len(strategies)} strategies.\n"
        f"Structure: {structure.formula} (SG {sg})\n"
        f"Troubleshooting suggestions:\n"
        f"  1. Try a different Miller index (e.g., '1 0 0' or '1 1 0')\n"
        f"  2. Increase min_slab_size significantly (e.g., 20.0 Å)\n"
        f"  3. Set symmetrize=False\n"
        f"  4. Use in_unit_planes=True with integer layer count\n"
        f"  5. Try shift=0.0 or shift=0.5 to pin specific terminations"
    )


# ═════════════════════════════════════════════════════════════════════════════
# ASE ↔ pymatgen converters
# ═════════════════════════════════════════════════════════════════════════════

def _ase_to_pmg(atoms):
    from pymatgen.io.ase import AseAtomsAdaptor
    return AseAtomsAdaptor.get_structure(atoms)


def _pmg_to_ase(structure):
    from pymatgen.io.ase import AseAtomsAdaptor
    return AseAtomsAdaptor.get_atoms(structure)


def _pmg_from_file(path: str):
    from pymatgen.core import Structure
    return Structure.from_file(path)


# ═════════════════════════════════════════════════════════════════════════════
# Mode 0: Shortcut — element + crystal + miller → slab
# ═════════════════════════════════════════════════════════════════════════════

def _build_shortcut_slab(params: ShortcutSlabParams) -> tuple:
    """Build bulk from KNOWN_ALAT, then cleave with SlabGenerator."""
    from ase.build import bulk as ase_bulk

    # Resolve lattice parameters
    known_a, known_c = KNOWN_ALAT.get(params.element, (None, None))
    a = params.alat or known_a
    c = params.alat_c or known_c

    if not a:
        raise ValueError(
            f"No lattice parameter for '{params.element}'. "
            "Provide alat or add to KNOWN_ALAT."
        )
    if params.crystal == "hcp" and not c:
        raise ValueError(
            f"HCP requires alat_c for '{params.element}'. "
            "Provide alat-c or add to KNOWN_ALAT."
        )

    kw = {"crystalstructure": params.crystal, "a": a}
    if params.crystal == "hcp":
        kw["covera"] = c / a

    atoms = ase_bulk(params.element, **kw)
    print(f"  Shortcut bulk: {params.element} {params.crystal.upper()}  "
          f"a={a:.4f} Å" + (f"  c={c:.4f} Å" if c else ""))

    structure = _ase_to_pmg(atoms)
    miller = _parse_miller(params.miller)

    slabs = _run_slabgenerator_with_fallback(
        structure=structure,
        miller=miller,
        min_slab_size=params.min_slab_size,
        vacuum=params.vacuum,
        center_slab=params.center_slab,
        in_unit_planes=params.in_unit_planes,
        symmetrize=params.symmetrize,
        all_terminations=params.all_terminations,
        shift_override=params.shift,
        max_normal_search=params.max_normal_search,
        primitive=params.primitive,
        reorient_lattice=params.reorient_lattice,
    )
    return slabs, miller


# ═════════════════════════════════════════════════════════════════════════════
# Mode 0b: Binary shortcut — prototype + elements + miller → slab
# ═════════════════════════════════════════════════════════════════════════════

def _build_binary_slab(params: BinarySlabParams) -> tuple:
    """Build bulk from KNOWN_BINARY using pymatgen's Structure.from_spacegroup()."""
    from pymatgen.core import Structure, Lattice

    proto_key = params.prototype.strip()
    if proto_key not in KNOWN_BINARY:
        available = ", ".join(sorted(KNOWN_BINARY.keys()))
        raise ValueError(
            f"Unknown prototype '{proto_key}'.\n"
            f"Available: {available}"
        )

    proto   = KNOWN_BINARY[proto_key]
    sg      = proto["sg"]
    coords  = proto["coords"]
    n_sites = proto["n_species"]
    a_val   = params.a or proto["a"]
    c_val   = params.c or proto["c"]

    # Parse element symbols
    elems = [e.strip() for e in params.elements.split(",")]
    if len(elems) != n_sites:
        raise ValueError(
            f"Prototype '{proto_key}' has {n_sites} unique Wyckoff sites "
            f"but {len(elems)} elements were given.\n"
            f"Provide exactly {n_sites} comma-separated symbols."
        )

    # Build lattice
    if   sg in (225, 216, 221, 123):   # cubic or tetragonal
        if c_val and abs(c_val - a_val) > 0.01:
            lattice = Lattice.tetragonal(a_val, c_val)
        else:
            lattice = Lattice.cubic(a_val)
    elif sg in (186, 194):             # hexagonal
        if not c_val:
            raise ValueError(f"Prototype '{proto_key}' (SG {sg}) requires c.")
        lattice = Lattice.hexagonal(a_val, c_val)
    elif sg == 167:                    # rhombohedral → hexagonal setting
        if not c_val:
            raise ValueError(f"Prototype '{proto_key}' (SG {sg}) requires c.")
        lattice = Lattice.hexagonal(a_val, c_val)
    elif sg == 136:                    # tetragonal
        if not c_val:
            raise ValueError(f"Prototype '{proto_key}' (SG {sg}) requires c.")
        lattice = Lattice.tetragonal(a_val, c_val)
    else:
        lattice = Lattice.cubic(a_val)

    # Build structure
    species = elems
    frac    = coords

    structure = Structure.from_spacegroup(
        sg=sg,
        lattice=lattice,
        species=species,
        coords=frac,
    )

    print(f"\n  Binary shortcut: {params.prototype}  →  {structure.formula}")
    print(f"    Spacegroup  : {sg}")
    print(f"    Lattice     : a={a_val:.4f} Å" + (f"  c={c_val:.4f} Å" if c_val else ""))
    print(f"    Sites       : {structure.num_sites}")
    print(f"    Elements    : {elems}  →  {len(coords)} Wyckoff sites")

    miller = _parse_miller(params.miller)
    slabs  = _run_slabgenerator_with_fallback(
        structure=structure,
        miller=miller,
        min_slab_size=params.min_slab_size,
        vacuum=params.vacuum,
        center_slab=params.center_slab,
        in_unit_planes=params.in_unit_planes,
        symmetrize=params.symmetrize,
        all_terminations=params.all_terminations,
        shift_override=params.shift,
        max_normal_search=params.max_normal_search,
        primitive=params.primitive,
        reorient_lattice=params.reorient_lattice,
    )
    return slabs, miller


# ═════════════════════════════════════════════════════════════════════════════
# Mode 1: Miller from file
# ═════════════════════════════════════════════════════════════════════════════

def _build_miller_slabs(input_path: str, params: MillerParams) -> list:
    """Load any structure file, run SlabGenerator, return list of pymatgen Slabs."""
    structure = _pmg_from_file(input_path)
    miller = _parse_miller(params.miller)
    return _run_slabgenerator_with_fallback(
        structure=structure,
        miller=miller,
        min_slab_size=params.min_slab_size,
        vacuum=params.vacuum,
        center_slab=params.center_slab,
        in_unit_planes=params.in_unit_planes,
        symmetrize=params.symmetrize,
        all_terminations=params.all_terminations,
        shift_override=params.shift,
        max_normal_search=params.max_normal_search,
        primitive=params.primitive,
        reorient_lattice=params.reorient_lattice,
    ), miller


# ═════════════════════════════════════════════════════════════════════════════
# Mode 2: MP → slab
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_mp_structure(mp_id: str, conventional_cell: bool):
    try:
        from mp_api.client import MPRester
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError:
        raise ImportError("Run: pip install mp-api pymatgen")
    with MPRester(MP_API_KEY) as mpr:
        structure = mpr.get_structure_by_material_id(
            mp_id, conventional_unit_cell=conventional_cell
        )
    return structure


def _fetch_mp_search_structure(
    formula: str,
    is_stable: bool,
    max_e_hull: float,
    conventional_cell: bool,
):
    try:
        from mp_api.client import MPRester
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError:
        raise ImportError("Run: pip install mp-api pymatgen")

    query_kwargs = dict(
        formula=formula,
        energy_above_hull=(0, max_e_hull),
        fields=["material_id", "energy_above_hull", "is_stable", "structure"],
    )
    if is_stable:
        query_kwargs["is_stable"] = True

    with MPRester(MP_API_KEY) as mpr:
        try:
            results = mpr.materials.summary.search(**query_kwargs)
        except TypeError:
            query_kwargs["chemsys_formula"] = query_kwargs.pop("formula")
            results = mpr.materials.summary.search(**query_kwargs)

    if not results:
        raise ValueError(
            f"No MP entries for formula='{formula}' with e_hull <= {max_e_hull}"
        )
    results = sorted(results, key=lambda r: r.energy_above_hull)
    best = results[0]
    print(f"  MP Search: picked {best.material_id}  "
          f"e_hull={best.energy_above_hull:.4f} eV/atom")

    structure = best.structure
    if conventional_cell:
        structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
    return structure


def _build_mp_slabs(params: MPSlabParams) -> tuple:
    if params.mp_id:
        structure = _fetch_mp_structure(params.mp_id, params.conventional_cell)
    elif params.formula:
        structure = _fetch_mp_search_structure(
            params.formula, params.is_stable,
            params.max_e_hull, params.conventional_cell,
        )
    else:
        raise ValueError("MPSlabParams requires either mp_id or formula.")

    miller = _parse_miller(params.miller)
    slabs = _run_slabgenerator_with_fallback(
        structure=structure,
        miller=miller,
        min_slab_size=params.min_slab_size,
        vacuum=params.vacuum,
        center_slab=params.center_slab,
        in_unit_planes=params.in_unit_planes,
        symmetrize=params.symmetrize,
        all_terminations=params.all_terminations,
        shift_override=params.shift,
        max_normal_search=params.max_normal_search,
        primitive=params.primitive,
        reorient_lattice=params.reorient_lattice,
    )
    return slabs, miller


# ═════════════════════════════════════════════════════════════════════════════
# Mode 3: COD → slab
# ═════════════════════════════════════════════════════════════════════════════

def _build_cod_slabs(params: CODSlabParams) -> tuple:
    import warnings
    from ase.io import read

    url = f"https://www.crystallography.net/cod/{params.cod_id}.cif"
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
        urllib.request.urlretrieve(url, f.name)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="crystal system.*is not interpreted",
                category=UserWarning,
            )
            atoms = read(f.name)

    structure = _ase_to_pmg(atoms)
    miller = _parse_miller(params.miller)
    slabs = _run_slabgenerator_with_fallback(
        structure=structure,
        miller=miller,
        min_slab_size=params.min_slab_size,
        vacuum=params.vacuum,
        center_slab=params.center_slab,
        in_unit_planes=params.in_unit_planes,
        symmetrize=params.symmetrize,
        all_terminations=params.all_terminations,
        shift_override=params.shift,
        max_normal_search=params.max_normal_search,
        primitive=params.primitive,
        reorient_lattice=params.reorient_lattice,
    )
    return slabs, miller


# ═════════════════════════════════════════════════════════════════════════════
# In-plane supercell expansion
# ═════════════════════════════════════════════════════════════════════════════

def _apply_supercell(atoms, opts: SupercellOptions):
    """Apply in-plane supercell expansion. nz is always forced to 1."""
    from ase.build import make_supercell

    if opts.supercell_matrix:
        mat = [int(x.strip()) for x in opts.supercell_matrix.split(",")]
        if len(mat) != 9:
            raise ValueError("supercell_matrix must have exactly 9 integers")
        M = np.array(mat, dtype=int).reshape(3, 3)

        # Enforce nz = 1
        if M[2, 0] != 0 or M[2, 1] != 0 or M[0, 2] != 0 or M[1, 2] != 0:
            print("  Warning: supercell_matrix z off-diagonal entries zeroed")
            M[2, 0] = M[2, 1] = M[0, 2] = M[1, 2] = 0
        if M[2, 2] != 1:
            print(f"  Warning: supercell_matrix M[2,2]={M[2,2]} forced to 1")
            M[2, 2] = 1

        det = int(round(abs(np.linalg.det(M))))
        n_after = len(atoms) * det
        print(f"  Supercell matrix {M.tolist()} → {n_after} atoms")
        atoms = make_supercell(atoms, M)
        nx_used, ny_used = int(M[0, 0]), int(M[1, 1])
        return atoms, (nx_used, ny_used, 1)

    else:
        nx = max(1, int(opts.nx))
        ny = max(1, int(opts.ny))
        if nx == 1 and ny == 1:
            print(f"  Supercell 1×1 (no expansion)")
        else:
            print(f"  In-plane supercell {nx}×{ny} → {len(atoms) * nx * ny} atoms")
        atoms = atoms.repeat([nx, ny, 1])
        return atoms, (nx, ny, 1)


# ═════════════════════════════════════════════════════════════════════════════
# Helper functions for output
# ═════════════════════════════════════════════════════════════════════════════

def _slab_thickness(slab) -> float:
    """Robust slab thickness in Å."""
    z_coords = [site.coords[2] for site in slab]
    return float(max(z_coords) - min(z_coords))


def _spacegroup_info(atoms, symprec: float = 0.1) -> str:
    try:
        import spglib
        cell_data = (
            atoms.get_cell().T,
            atoms.get_scaled_positions(),
            atoms.get_atomic_numbers(),
        )
        ds = spglib.get_symmetry_dataset(cell_data, symprec=symprec)
        return f"{ds['international']},{ds['number']}"
    except Exception:
        return "unknown"


def _write_one_slab(
    atoms,
    prefix: str,
    term_idx: int,
    miller_label: str,
    miller: tuple,
    vacuum: float,
    mode: str,
    formats: list,
    extra_meta: dict,
    supercell_used: tuple = (1, 1, 1),
) -> dict:
    """Write one slab to disk in requested formats."""
    from ase.io import write

    label = f"{prefix}_{miller_label}_t{term_idx}"
    cell = atoms.get_cell()
    formula = atoms.get_chemical_formula()
    n_sites = len(atoms)
    volume = atoms.get_volume()
    sg_info = _spacegroup_info(atoms)

    # Calculate slab thickness and surface area
    z_coords = atoms.get_positions()[:, 2]
    thickness = float(max(z_coords) - min(z_coords))
    a_vec, b_vec = cell[0], cell[1]
    surface_area = float(np.linalg.norm(np.cross(a_vec, b_vec)))

    format_map = {
        "extxyz": (f"{label}.extxyz", "extxyz"),
        "xyz": (f"{label}.xyz", "xyz"),
        "vasp": (f"{label}_POSCAR", "vasp"),
        "cif": (f"{label}.cif", "cif"),
        "json": (f"{label}.json", "json"),
    }

    written_files = {}
    encoded_xyz = {}

    for fmt in formats:
        if fmt not in format_map:
            print(f"  Warning: unknown format '{fmt}', skipping")
            continue
        fname, ase_fmt = format_map[fmt]
        write(fname, atoms, format=ase_fmt)
        written_files[fmt] = fname
        if fmt in ("extxyz", "xyz"):
            with open(fname, "rb") as fh:
                encoded_xyz[fmt] = base64.b64encode(
                    gzip.compress(fh.read())
                ).decode()

    meta = {
        "source_mode": mode,
        "formula": formula,
        "miller": list(miller) if miller else None,
        "termination_index": term_idx,
        "n_sites": n_sites,
        "supercell": list(supercell_used),
        "vacuum_requested": vacuum,
        "slab_thickness_A": round(thickness, 4),
        "surface_area_A2": round(surface_area, 4),
        "volume_A3": round(volume, 4),
        "spacegroup": sg_info,
        "cell": cell.tolist(),
        "files": written_files,
        **extra_meta,
    }

    print(f"  Termination {term_idx:>2d} | {formula:>20s} | "
          f"{n_sites:>4d} atoms | "
          f"thickness={thickness:.2f} Å | area={surface_area:.2f} Å²")

    return {"metadata": meta, "encoded_xyz": encoded_xyz, "files": written_files}


def _finalise_slabs(
    slab_list,
    miller: tuple,
    mode: str,
    vacuum: float,
    supercell_opts: SupercellOptions,
    out_config: SlabOutputConfig,
) -> SlabOutput:
    """Unified finalisation for all modes."""
    miller_label = f"m{''.join(str(h) for h in miller)}"
    prefix = out_config.output_prefix
    formats = [f.strip() for f in out_config.output_formats.split(",")]
    n_terms = len(slab_list)

    print(f"\n  {'='*60}")
    print(f"  Mode        : {mode.upper()}")
    print(f"  Miller      : {miller}")
    print(f"  Terminations: {n_terms}")
    print(f"  {'='*60}")

    all_metadata = []
    all_files = []
    primary_atoms = None
    primary_meta = {}
    primary_encoded = {}

    for idx, slab_obj in enumerate(slab_list):
        # Convert to ASE Atoms
        if hasattr(slab_obj, "get_cell"):
            atoms = slab_obj
            extra = {}
        else:
            atoms = _pmg_to_ase(slab_obj)
            extra = {
                "pmg_shift": float(slab_obj.shift) if hasattr(slab_obj, "shift") else None,
                "pmg_scale_factor": (
                    slab_obj.scale_factor.tolist()
                    if hasattr(slab_obj, "scale_factor")
                    and hasattr(slab_obj.scale_factor, "tolist")
                    else None
                ),
            }

        # Apply in-plane supercell
        atoms, supercell_used = _apply_supercell(atoms, supercell_opts)

        # Write files
        write_this = (idx == 0) or out_config.write_all_terminations
        if write_this:
            result = _write_one_slab(
                atoms=atoms,
                prefix=prefix,
                term_idx=idx,
                miller_label=miller_label,
                miller=miller,
                vacuum=vacuum,
                mode=mode,
                formats=formats,
                extra_meta=extra,
                supercell_used=supercell_used,
            )
            all_metadata.append(result["metadata"])
            for fmt, fpath in result["files"].items():
                all_files.append(FlyteFile(fpath))

            if idx == 0:
                primary_atoms = atoms
                primary_meta = result["metadata"]
                primary_encoded = result["encoded_xyz"]

    # Primary slab summary
    primary_poscar = next(
        (FlyteFile(m["files"]["vasp"]) for m in all_metadata
         if "vasp" in m.get("files", {})),
        all_files[0] if all_files else FlyteFile("/dev/null"),
    )

    return SlabOutput(
        primary_slab_file=primary_poscar,
        all_termination_files=all_files,
        termination_metadata=all_metadata,
        formula=primary_meta.get("formula", ""),
        num_atoms=primary_meta.get("n_sites", 0),
        n_terminations=n_terms,
        miller=str(miller),
        slab_thickness=primary_meta.get("slab_thickness_A", 0.0),
        surface_area=primary_meta.get("surface_area_A2", 0.0),
        spacegroup_info=primary_meta.get("spacegroup", "unknown"),
        encoded_xyz=primary_encoded,
        metadata=primary_meta,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Flyte Task
# ═════════════════════════════════════════════════════════════════════════════

@task(container_image=_IMAGE, requests=_RESOURCES, limits=_RESOURCES)
def generate_slab_task(
    shortcut_params: Optional[ShortcutSlabParams] = None,
    binary_params:   Optional[BinarySlabParams]   = None,
    miller_params:   Optional[MillerParams]        = None,
    mp_params:       Optional[MPSlabParams]        = None,
    cod_params:      Optional[CODSlabParams]       = None,
    input_file:      Optional[FlyteFile]           = None,
    supercell:       SupercellOptions              = SupercellOptions(),
    out_config:      SlabOutputConfig              = SlabOutputConfig(),
) -> SlabOutput:

    slab_list = []
    miller    = None
    mode      = ""
    vacuum    = 15.0

    # Mode 0a: Shortcut (single element)
    if shortcut_params is not None:
        slabs, miller = _build_shortcut_slab(shortcut_params)
        slab_list = slabs
        mode   = "shortcut"
        vacuum = shortcut_params.vacuum

    # Mode 0b: Binary shortcut (compound prototype)
    elif binary_params is not None:
        slabs, miller = _build_binary_slab(binary_params)
        slab_list = slabs
        mode   = "binary"
        vacuum = binary_params.vacuum

    # Mode 1: Miller from file
    elif miller_params is not None:
        if input_file is None:
            raise ValueError("miller_params mode requires input_file.")
        local_path = input_file.download()
        slabs, miller = _build_miller_slabs(local_path, miller_params)
        slab_list = slabs
        mode = "miller_file"
        vacuum = miller_params.vacuum

    # Mode 2: MP → slab
    elif mp_params is not None:
        slabs, miller = _build_mp_slabs(mp_params)
        slab_list = slabs
        mode = "mp_slab"
        vacuum = mp_params.vacuum

    # Mode 3: COD → slab
    elif cod_params is not None:
        slabs, miller = _build_cod_slabs(cod_params)
        slab_list = slabs
        mode = "cod_slab"
        vacuum = cod_params.vacuum

    else:
        raise ValueError(
            "No input mode selected. Provide one of: "
            "shortcut_params, binary_params, miller_params, mp_params, cod_params."
        )

    return _finalise_slabs(
        slab_list=slab_list,
        miller=miller,
        mode=mode,
        vacuum=vacuum,
        supercell_opts=supercell,
        out_config=out_config,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Flyte Workflow
# ═════════════════════════════════════════════════════════════════════════════

@workflow
def generate_slab_workflow(
    shortcut_params: Optional[ShortcutSlabParams] = None,
    binary_params:   Optional[BinarySlabParams]   = None,
    miller_params:   Optional[MillerParams]        = None,
    mp_params:       Optional[MPSlabParams]        = None,
    cod_params:      Optional[CODSlabParams]       = None,
    input_file:      Optional[FlyteFile]           = None,
    supercell:       SupercellOptions              = SupercellOptions(),
    out_config:      SlabOutputConfig              = SlabOutputConfig(),
) -> SlabOutput:
    return generate_slab_task(
        shortcut_params=shortcut_params,
        binary_params=binary_params,
        miller_params=miller_params,
        mp_params=mp_params,
        cod_params=cod_params,
        input_file=input_file,
        supercell=supercell,
        out_config=out_config,
    )


# ═════════════════════════════════════════════════════════════════════════════
# CLI — test each mode from terminal
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Slab Generator — FIXED VERSION with better error handling",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
EXAMPLES:
  # Mode 0b — Binary shortcut: prototype + elements (no file, no API key)
  python slab_gen.py --mode binary --prototype TiO2 --elements Ti,O --miller "1 0 1" --vacuum 15
  python slab_gen.py --mode binary --prototype MgO --elements Mg,O --miller "1 0 0" --vacuum 15
  python slab_gen.py --mode binary --prototype ZnO --elements Zn,O --miller "0 0 0 1" --vacuum 15
  python slab_gen.py --mode binary --prototype GaN --elements Ga,N --miller "0 0 0 1" --vacuum 15
  python slab_gen.py --mode binary --prototype Al2O3 --elements Al,O --miller "0 0 0 1" --vacuum 15
  python slab_gen.py --mode binary --prototype SrTiO3 --elements Sr,Ti,O --miller "1 0 0" --vacuum 15
  python slab_gen.py --mode binary --prototype TiO2 --elements Ti,O --miller "1 0 1" --nx 2 --ny 2 --no-all-terms --prefix TiO2_101

  python 03_slabgeneration.py --mode shortcut --element Cu --crystal fcc --miller "1 1 1"
  python 03_slabgeneration.py --mode shortcut --element Pd --crystal fcc --miller "1 1 0" --vacuum 20
  python 03_slabgeneration.py --mode shortcut --element Ti --crystal hcp --miller "0 0 0 1"
  python 03_slabgeneration.py --mode shortcut --element Fe --crystal bcc --miller "1 1 0" --nx 2 --ny 2
  python 03_slabgeneration.py --mode shortcut --element Cu --crystal fcc --miller "1 1 1" --nx 3 --ny 3 --no-all-terms
  python 03_slabgeneration.py --mode shortcut --element Ni --crystal fcc --miller "1 0 0" --alat 3.52 --vacuum 18

  # Mode 1 — Miller from file (all terminations)
  python 03_slabgeneration.py --mode miller --input-file POSCAR --miller "1 1 1"
  python 03_slabgeneration.py --mode miller --input-file bulk.cif --miller "1 1 0" --vacuum 20 --no-all-terms

  # Mode 1 — pin a specific termination shift
  python 03_slabgeneration.py --mode miller --input-file POSCAR --miller "1 1 1" --shift 0.25

  # Mode 3 — MP by ID (needs: export MP_API_KEY=your_key)
  python 03_slabgeneration.py --mode mp --mp-id mp-30 --miller "1 0 0"

  # Mode 3 — MP by formula search
  python 03_slabgeneration.py --mode mp --formula TiO2 --miller "1 1 0" --max-e-hull 0.05

  # Mode 4 — COD
  python 03_slabgeneration.py --mode cod --cod-id 1010942 --miller "1 0 0"

  # In-plane supercell — use --nx and --ny (nz is always fixed to 1)
  python 03_slabgeneration.py --mode shortcut --element Cu --crystal fcc --miller "1 1 1" --nx 2 --ny 2
  python 03_slabgeneration.py --mode miller --input-file POSCAR --miller "1 1 0" --nx 3 --ny 2

  # Advanced supercell matrix (nz row auto-enforced to (0,0,1))
  python 03_slabgeneration.py --mode miller --input-file POSCAR --miller "1 1 1" --supercell-matrix 2,0,0,0,2,0,0,0,1

  # Custom output formats and prefix
  python 03_slabgeneration.py --mode shortcut --element Pd --crystal fcc --miller "1 1 1" --formats extxyz,vasp --prefix Pd111
"""
    )

    parser.add_argument(
        "--mode", required=True,
        choices=["shortcut", "binary", "miller", "mp", "cod"],
        help="Slab generation mode",
    )

    # Mode 0 — Shortcut
    parser.add_argument("--element", default="Cu", help="Chemical symbol (shortcut mode)")
    parser.add_argument("--crystal", default="fcc", help="Crystal structure (shortcut mode)")
    parser.add_argument("--alat", type=float, default=None, help="Override lattice param a (Å)")
    parser.add_argument("--alat-c", type=float, default=None, help="Override c for hcp (Å)")

    # Mode 0b — Binary shortcut
    parser.add_argument("--prototype", default="TiO2", help="Compound prototype key")
    parser.add_argument("--elements", default="Ti,O", help="Comma-separated element symbols")
    parser.add_argument("--a", type=float, default=None, help="Override lattice param a (Å)")
    parser.add_argument("--c", type=float, default=None, help="Override lattice param c (Å)")

    # Mode 1 — Miller
    parser.add_argument("--input-file", default=None, help="Bulk structure file")
    parser.add_argument("--miller", default="1 1 1", help="Miller indices")
    parser.add_argument("--min-slab-size", type=float, default=8.0, help="Min slab thickness (Å)")
    parser.add_argument("--vacuum", type=float, default=15.0, help="Vacuum thickness (Å)")
    parser.add_argument("--no-center", action="store_true", help="Do not center slab")
    parser.add_argument("--in-unit-planes", action="store_true")
    parser.add_argument("--no-symmetrize", action="store_true")
    parser.add_argument("--no-all-terms", action="store_true", help="Only generate first termination")
    parser.add_argument("--shift", type=float, default=None, help="Pin fractional shift")
    parser.add_argument("--max-normal-search", type=int, default=2)
    parser.add_argument("--no-primitive", action="store_true")
    parser.add_argument("--no-reorient", action="store_true")

    # Mode 2 — MP
    parser.add_argument("--mp-id", default=None)
    parser.add_argument("--formula", default=None)
    parser.add_argument("--is-stable", action="store_true")
    parser.add_argument("--max-e-hull", type=float, default=0.1)
    parser.add_argument("--no-conventional", action="store_true")

    # Mode 3 — COD
    parser.add_argument("--cod-id", type=int, default=1010942)

    # Supercell
    parser.add_argument("--nx", type=int, default=1, help="In-plane supercell repetitions along a")
    parser.add_argument("--ny", type=int, default=1, help="In-plane supercell repetitions along b")
    parser.add_argument("--supercell-matrix", default=None, help="Advanced: 9 comma-separated ints")

    # Output
    parser.add_argument("--prefix", default="slab")
    parser.add_argument("--formats", default="extxyz,vasp,cif")
    parser.add_argument("--no-write-all-terms", action="store_true")

    args = parser.parse_args()

    SC = SupercellOptions(
        nx=args.nx,
        ny=args.ny,
        supercell_matrix=args.supercell_matrix,
    )
    OUT = SlabOutputConfig(
        output_prefix=args.prefix,
        output_formats=args.formats,
        write_all_terminations=not args.no_write_all_terms,
    )
    kwargs = dict(supercell=SC, out_config=OUT)

    if args.mode == "shortcut":
        kwargs["shortcut_params"] = ShortcutSlabParams(
            element=args.element,
            crystal=args.crystal,
            miller=args.miller,
            alat=args.alat,
            alat_c=args.alat_c,
            min_slab_size=args.min_slab_size,
            vacuum=args.vacuum,
            center_slab=not args.no_center,
            in_unit_planes=args.in_unit_planes,
            symmetrize=not args.no_symmetrize,
            all_terminations=not args.no_all_terms,
            shift=args.shift,
            max_normal_search=args.max_normal_search,
            primitive=not args.no_primitive,
            reorient_lattice=not args.no_reorient,
        )

    elif args.mode == "binary":
        kwargs["binary_params"] = BinarySlabParams(
            prototype=args.prototype,
            elements=args.elements,
            miller=args.miller,
            a=args.a,
            c=args.c,
            min_slab_size=args.min_slab_size,
            vacuum=args.vacuum,
            center_slab=not args.no_center,
            in_unit_planes=args.in_unit_planes,
            symmetrize=not args.no_symmetrize,
            all_terminations=not args.no_all_terms,
            shift=args.shift,
            max_normal_search=args.max_normal_search,
            primitive=not args.no_primitive,
            reorient_lattice=not args.no_reorient,
        )

    elif args.mode == "miller":
        if not args.input_file:
            parser.error("--input-file required for --mode miller")
        kwargs["miller_params"] = MillerParams(
            miller=args.miller,
            min_slab_size=args.min_slab_size,
            vacuum=args.vacuum,
            center_slab=not args.no_center,
            in_unit_planes=args.in_unit_planes,
            symmetrize=not args.no_symmetrize,
            all_terminations=not args.no_all_terms,
            shift=args.shift,
            max_normal_search=args.max_normal_search,
            primitive=not args.no_primitive,
            reorient_lattice=not args.no_reorient,
        )
        kwargs["input_file"] = FlyteFile(args.input_file)

    elif args.mode == "mp":
        if not args.mp_id and not args.formula:
            parser.error("--mp-id or --formula required for --mode mp")
        kwargs["mp_params"] = MPSlabParams(
            mp_id=args.mp_id,
            formula=args.formula,
            is_stable=args.is_stable,
            max_e_hull=args.max_e_hull,
            conventional_cell=not args.no_conventional,
            miller=args.miller,
            min_slab_size=args.min_slab_size,
            vacuum=args.vacuum,
            center_slab=not args.no_center,
            in_unit_planes=args.in_unit_planes,
            symmetrize=not args.no_symmetrize,
            all_terminations=not args.no_all_terms,
            shift=args.shift,
            max_normal_search=args.max_normal_search,
            primitive=not args.no_primitive,
            reorient_lattice=not args.no_reorient,
        )

    elif args.mode == "cod":
        kwargs["cod_params"] = CODSlabParams(
            cod_id=args.cod_id,
            miller=args.miller,
            min_slab_size=args.min_slab_size,
            vacuum=args.vacuum,
            center_slab=not args.no_center,
            in_unit_planes=args.in_unit_planes,
            symmetrize=not args.no_symmetrize,
            all_terminations=not args.no_all_terms,
            shift=args.shift,
            max_normal_search=args.max_normal_search,
            primitive=not args.no_primitive,
            reorient_lattice=not args.no_reorient,
        )

    print("\n" + "=" * 60)
    print(f"  Mode: {args.mode.upper()} (FIXED VERSION)")
    print("=" * 60)

    try:
        r = generate_slab_workflow(**kwargs)

        print(f"\n  ✓ SUCCESS {r.formula}")
        print(f"      Miller         : {r.miller}")
        print(f"      Terminations   : {r.n_terminations}")
        print(f"      Atoms (t=0)    : {r.num_atoms}")
        print(f"      Thickness (t=0): {r.slab_thickness:.3f} Å")
        print(f"      Surface area   : {r.surface_area:.3f} Å²")
        print(f"      Spacegroup     : {r.spacegroup_info}\n")
        
    except Exception as e:
        print(f"\n  ✗ FAILED: {str(e)}")
        print(f"\n  Troubleshooting suggestions:")
        print(f"  1. Try a different Miller index (e.g., '1 0 0' instead of '0 0 0 1')")
        print(f"  2. Increase --min-slab-size (e.g., 20.0)")
        print(f"  3. Add --no-symmetrize flag")
        print(f"  4. Add --max-normal-search 5")
        print(f"  5. Try --shift 0.0 or --shift 0.5")
        exit(1)

