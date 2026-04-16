"""
Step 5: Adsorbate Placement — Flyte Workflow
=============================================
Places an adsorbate molecule on a surface slab at ONE representative
site of the user-chosen site type (ontop | bridge | hollow).

pymatgen finds all symmetry-inequivalent sites; we take only the first
(most symmetric) one for the selected type — no symmetry-equivalent
duplicates are generated.

Features
--------
• accepts CIF / POSCAR / extxyz slab files
• user picks site type: ontop | bridge | hollow
  → exactly ONE representative site is used
• user picks orientation: vertical | horizontal | tilted | all
• single height or comma-separated height sweep
• CO / OH / H adsorbates built-in, plus any molecule from ase.build.molecule()
  (H2O, NH3, CO2, CH4, NO, C2H2, C6H6, …) or any single element symbol
• per-structure JSON manifest
• Flyte-friendly outputs with FlyteFile + encoded_xyz

Orientation definitions
-----------------------
  vertical   — molecule axis along surface normal (z-up)
  horizontal — molecule axis flat in the xy-plane (90° tilt from normal)
  tilted     — molecule axis at 45° to surface normal
  all        — all three orientations

Input examples (CLI):
  --adsorbate CO --site-type ontop   --orientation vertical   --height 2.0
  --adsorbate CO --site-type ontop   --orientation all        --height 1.9,2.0,2.1
  --adsorbate OH --site-type bridge  --orientation tilted     --height 2.0
  --adsorbate H  --site-type hollow  --orientation vertical   --height 1.8

Output files produced (site-type=ontop, orientation=all, height=2.0):
  ads_CO_ontop_h2.00_vertical/POSCAR
  ads_CO_ontop_h2.00_horizontal/POSCAR
  ads_CO_ontop_h2.00_tilted/POSCAR
"""

from __future__ import annotations
from typing import NamedTuple, List, Dict
from dataclasses import dataclass
import json, gzip, base64
from pathlib import Path

import numpy as np

from flytekit import task, workflow, Resources
from flytekit.types.file import FlyteFile

from pymatgen.core import Molecule, Structure
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms as AseAtoms
from ase.build import molecule as ase_molecule


# ── Flyte resources ───────────────────────────────────────────────────────────
_RESOURCES = Resources(cpu="2", mem="8Gi", gpu="0")
_IMAGE     = "guptag13/slab:v5"


# ── Fallback adsorbate library (used only when ASE doesn't know the molecule) ─
# These cover the most common surface-science adsorbates.  Any molecule that
# ase.build.molecule() recognises is loaded from there instead and takes
# priority over this dict.
_FALLBACK_ADSORBATES: Dict[str, Molecule] = {
    "CO": Molecule(["C", "O"], [[0, 0, 0], [0, 0, 1.15]]),
    "OH": Molecule(["O", "H"], [[0, 0, 0], [0, 0, 0.97]]),
    "H":  Molecule(["H"],      [[0, 0, 0]]),
}


def get_adsorbate(name: str) -> Molecule:
    """
    Load an adsorbate molecule by name.

    Resolution order
    ----------------
    1. ase.build.molecule(name)  — covers ~100 common molecules
       (CO, CO2, H2O, NH3, CH4, NO, OH, H2, N2, C2H2, C6H6, …)
    2. Built-in fallback dict    — CO, OH, H with hand-tuned bond lengths
    3. Single-element symbol     — treat *name* as an atomic symbol (e.g. "O", "N")

    Raises ValueError if none of the above succeed.

    Examples
    --------
    >>> get_adsorbate("CO")      # from ASE
    >>> get_adsorbate("H2O")     # from ASE
    >>> get_adsorbate("NH3")     # from ASE
    >>> get_adsorbate("O")       # single atom fallback
    >>> get_adsorbate("custom")  # raises ValueError with helpful message
    """
    # 1. Try ASE molecule database first
    try:
        atoms = ase_molecule(name)
        # Centre molecule at origin so anchor atom starts at [0,0,0]
        atoms.positions -= atoms.positions[0]
        mol = _ase_to_pmg(atoms)
        print(f"  Adsorbate '{name}' loaded from ase.build.molecule  "
              f"({len(mol)} atom(s))")
        return mol
    except (KeyError, Exception):
        pass

    # 2. Fallback dict (hand-tuned geometries)
    if name in _FALLBACK_ADSORBATES:
        mol = _FALLBACK_ADSORBATES[name]
        print(f"  Adsorbate '{name}' loaded from built-in fallback library  "
              f"({len(mol)} atom(s))")
        return mol

    # 3. Single-element atom (e.g. "O", "N", "S")
    try:
        mol = Molecule([name], [[0, 0, 0]])
        print(f"  Adsorbate '{name}' treated as single atom  (1 atom)")
        return mol
    except Exception:
        pass

    raise ValueError(
        f"Unknown adsorbate '{name}'.\n"
        f"  • Pass any name recognised by ase.build.molecule()  "
        f"(CO, H2O, NH3, CO2, CH4, NO, C6H6, …)\n"
        f"  • Or an element symbol for a single atom  (O, N, S, …)\n"
        f"  • Built-in fallbacks: {list(_FALLBACK_ADSORBATES)}"
    )

# ── Site type options ─────────────────────────────────────────────────────────
SITE_TYPE_KEYS = ("ontop", "bridge", "hollow")

# ── Orientation options ───────────────────────────────────────────────────────
ORIENTATION_KEYS = ("vertical", "horizontal", "tilted", "all")

# rotation axis + tilt angle (degrees) + description for each orientation
_ORIENT_SPEC: Dict[str, tuple] = {
    "vertical":   ([1, 0, 0],   0, "axis along surface normal (z-up)"),
    "horizontal": ([1, 0, 0],  90, "axis flat in xy-plane"),
    "tilted":     ([1, 0, 0],  45, "axis at 45° to surface normal"),
}

# ── Output format map ─────────────────────────────────────────────────────────
FORMAT_MAP: Dict[str, tuple] = {
    "vasp":   ("_POSCAR", "poscar"),
    "cif":    (".cif",    "cif"),
    "extxyz": (".extxyz", "extxyz"),
    "xyz":    (".xyz",    "xyz"),
}


# =============================================================================
# Input / Output dataclasses
# =============================================================================

@dataclass
class AdsorbateParams:
    """
    adsorbate     : any ase.build.molecule name, element symbol, or built-in key
    site_type     : ontop | bridge | hollow
    site_element  : element(s) nearest to the desired site - for alloys / MXenes.
                    Leave blank ("") to accept the first site of *site_type*
                    regardless of which element is underneath.
                    Examples:
                      "Mo"     -> ontop above a Mo atom
                      "Ti"     -> ontop above a Ti atom
                      "Mo,Ti"  -> bridge between Mo and Ti
                      "Mo,Mo"  -> bridge between two Mo atoms
                    Matched by comparing the sorted nearest-surface-atom elements
                    to the site against the sorted input list.
    height        : adsorption height(s) in Angstrom above topmost slab atom.
                    Single value  -> "2.0"
                    Height sweep  -> "1.9,2.0,2.1"
    orientation   : vertical | horizontal | tilted | all
    """
    adsorbate:    str = "CO"
    site_type:    str = "ontop"   # ontop | bridge | hollow
    site_element: str = ""        # "" = any; "Mo" = ontop-Mo; "Mo,Ti" = bridge-Mo/Ti
    height:       str = "2.0"     # "2.0"  or  "1.9,2.0,2.1"
    orientation:  str = "vertical"   # vertical | horizontal | tilted | all


@dataclass
class AdsorbateOutputConfig:
    output_prefix:  str  = "ads"
    output_formats: str  = "vasp,cif"   # comma-separated
    write_metadata: bool = True


# =============================================================================
# Output NamedTuple
# =============================================================================

class AdsorbateOutput(NamedTuple):
    primary_structure:   FlyteFile
    all_structure_files: List[FlyteFile]
    manifest:            List[dict]
    host_formula:        str
    adsorbate_name:      str
    n_structures:        int
    summary:             str
    encoded_xyz:         Dict[str, str]
    metadata:            dict


# =============================================================================
# Helpers
# =============================================================================

def _parse_heights(height_str: str) -> List[float]:
    """'2.0' → [2.0]   |   '1.9,2.0,2.1' → [1.9, 2.0, 2.1]"""
    return [float(x.strip()) for x in height_str.replace(",", " ").split()]


def _load_structure(path: str) -> Structure:
    try:
        return Structure.from_file(path)
    except Exception:
        from ase.io import read as ase_read
        return AseAtomsAdaptor.get_structure(ase_read(path))


def _encode_file(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(gzip.compress(fh.read())).decode()


def _pmg_to_ase(mol: Molecule) -> AseAtoms:
    return AseAtoms(
        symbols=[str(s.specie) for s in mol],
        positions=mol.cart_coords,
    )


def _ase_to_pmg(atoms: AseAtoms) -> Molecule:
    return Molecule(
        species=list(atoms.get_chemical_symbols()),
        coords=atoms.get_positions().tolist(),
    )


# =============================================================================
# Rotation (Rodrigues)
# =============================================================================

def _rotate_molecule(atoms: AseAtoms, axis: list, angle_deg: float) -> AseAtoms:
    """Rotate ASE Atoms around *axis* by *angle_deg*; returns a new copy."""
    atoms  = atoms.copy()
    angle  = np.deg2rad(angle_deg)
    axis   = np.asarray(axis, dtype=float)
    axis  /= np.linalg.norm(axis)
    K = np.array([
        [ 0,        -axis[2],  axis[1]],
        [ axis[2],   0,       -axis[0]],
        [-axis[1],   axis[0],  0      ],
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    atoms.positions = atoms.positions @ R.T
    return atoms


# =============================================================================
# Orientation builder
# =============================================================================

def _build_oriented_molecules(
    base_mol:    Molecule,
    orientation: str,           # "vertical" | "horizontal" | "tilted" | "all"
) -> List[tuple]:               # [(label, Molecule), ...]
    """
    Return a list of (label, Molecule) for the requested orientation.
    Single-atom adsorbates (H) always return just [("vertical", mol)]
    because rotation is physically meaningless for them.
    """
    if len(base_mol) == 1:
        return [("vertical", base_mol.copy())]

    keys = list(_ORIENT_SPEC.keys()) if orientation == "all" else [orientation]
    base_ase = _pmg_to_ase(base_mol)

    result = []
    for key in keys:
        axis, angle, desc = _ORIENT_SPEC[key]
        rotated = _rotate_molecule(base_ase, axis, angle)
        result.append((key, _ase_to_pmg(rotated)))
        print(f"  Orientation [{key}]: {desc}")
    return result


# =============================================================================
# Surface-element catalogue  (alloy / MXene aware)
# =============================================================================

def _surface_elements(slab: Structure, z_tol: float = 0.5) -> List[str]:
    """Return element symbols of surface-layer atoms (topmost z-layer)."""
    z_max = max(s.coords[2] for s in slab)
    return [str(s.specie) for s in slab if s.coords[2] >= z_max - z_tol]


def _site_nearest_elements(
    slab:        Structure,
    site_coords: np.ndarray,
    n_nearest:   int,
    z_tol:       float = 3.0,
) -> List[str]:
    """
    Return the sorted element symbols of the *n_nearest* surface atoms to
    *site_coords* (only atoms within *z_tol* Ang of the surface are considered).
    """
    z_max = max(s.coords[2] for s in slab)
    surf  = [(s, np.linalg.norm(s.coords[:2] - site_coords[:2]))
             for s in slab if s.coords[2] >= z_max - z_tol]
    surf.sort(key=lambda x: x[1])
    return sorted(str(s.specie) for s, _ in surf[:n_nearest])


def _n_nearest_for_site_type(site_type: str) -> int:
    """How many nearest atoms define the site chemistry."""
    return {"ontop": 1, "bridge": 2, "hollow": 3}.get(site_type, 1)


def list_available_sites(slab: Structure) -> Dict[str, List[str]]:
    """
    Return a human-readable dict of all symmetry-inequivalent sites with
    their element labels — useful for interactive site selection.

    Example output::
        {
          "ontop":  ["ontop-Mo", "ontop-Ti", "ontop-N"],
          "bridge": ["bridge-Mo/Mo", "bridge-Mo/Ti", "bridge-Ti/N"],
          "hollow": ["hollow-Mo/Mo/Ti", "hollow-Mo/Ti/N"],
        }
    """
    asf       = AdsorbateSiteFinder(slab)
    all_sites = asf.find_adsorption_sites()
    n_near    = _n_nearest_for_site_type

    result: Dict[str, List[str]] = {}
    for stype, coords_list in all_sites.items():
        if stype == "all":
            continue
        labels = []
        for coords in coords_list:
            elems = _site_nearest_elements(slab, np.array(coords), n_near(stype))
            labels.append(f"{stype}-{'/'.join(elems)}")
        result[stype] = labels
    return result


def _get_representative_site(
    slab:         Structure,
    asf:          AdsorbateSiteFinder,
    site_type:    str,
    site_element: str,              # "" | "Mo" | "Mo,Ti" | ...
) -> np.ndarray:
    """
    Return coordinates of the ONE representative site that matches
    *site_type* and (optionally) *site_element*.

    site_element matching
    ---------------------
    The sorted list of nearest-surface-atom elements at each candidate site
    is compared against the sorted, comma-split *site_element* input.
    Empty string skips the element filter and takes the first site.

    Raises ValueError with a full list of available labelled sites if no
    match is found.
    """
    all_sites = asf.find_adsorption_sites()

    if site_type not in all_sites or len(all_sites[site_type]) == 0:
        available = list_available_sites(slab)
        raise ValueError(
            f"Site type '{site_type}' not found on this surface.\n"
            f"Available sites: {available}"
        )

    n_near      = _n_nearest_for_site_type(site_type)
    want_elems = (
        sorted(e.strip() for e in site_element.replace(" ", ",").split(",") if e.strip())
        if site_element.strip() else []
    )

    for coords in all_sites[site_type]:
        coords = np.array(coords)
        if not want_elems:
            # No element filter — take the very first (most symmetric) site
            elems = _site_nearest_elements(slab, coords, n_near)
            label = f"{site_type}-{'/'.join(elems)}"
            print(f"  Selected site: {label} at {coords.round(3)}")
            return coords
        # Element-filtered match
        elems = _site_nearest_elements(slab, coords, n_near)
        if elems == want_elems:
            label = f"{site_type}-{'/'.join(elems)}"
            print(f"  Selected site: {label} at {coords.round(3)}")
            return coords

    # Nothing matched — show what IS available
    available = list_available_sites(slab)
    site_lines = "\n".join(f"  {k}: {v}" for k, v in available.items())
    raise ValueError(
        f"No '{site_type}' site with element(s) '{site_element}' found.\n"
        f"Available labelled sites on this surface:\n"
        + site_lines
    )


# =============================================================================
# Core placement
# =============================================================================

def _place_adsorbates(
    slab:           Structure,
    adsorbate:      Molecule,
    adsorbate_name: str,
    heights:        List[float],
    site_type:      str,
    site_element:   str,
    orientation:    str,
    out_prefix:     str,
    formats:        List[str],
) -> List[dict]:
    """
    Place adsorbate at the single representative site of *site_type*
    (filtered by *site_element* if given) for every height x orientation.

    Total structures = len(heights) x n_orientations
    """
    invalid = [f for f in formats if f not in FORMAT_MAP]
    if invalid:
        raise ValueError(f"Unknown format(s): {invalid}. Choose from: {list(FORMAT_MAP)}")

    asf        = AdsorbateSiteFinder(slab)
    slab_z_max = max(site.coords[2] for site in slab)
    n_slab     = len(slab)

    # 1. One representative site (element-aware)
    site_coords = _get_representative_site(slab, asf, site_type, site_element)

    # 2. Oriented molecule variants
    oriented_mols = _build_oriented_molecules(adsorbate, orientation)

    manifest: List[dict] = []

    for h in heights:
        for orient_label, orient_mol in oriented_mols:

            elem_label = site_element.replace(",", "-") if site_element.strip() else "any"
            tag = f"{out_prefix}_{adsorbate_name}_{site_type}-{elem_label}_h{h:.2f}_{orient_label}"

            # Place molecule at representative site
            struct = asf.add_adsorbate(orient_mol, site_coords, translate=False)

            # Shift adsorbate anchor to slab_z_max + h
            ads_indices  = list(range(n_slab, len(struct)))
            anchor_z_now = struct[n_slab].coords[2]
            dz           = (slab_z_max + h) - anchor_z_now
            struct.translate_sites(ads_indices, [0, 0, dz], frac_coords=False)

            # Write output files
            written:  Dict[str, str] = {}
            run_dir = Path(tag)
            run_dir.mkdir(exist_ok=True)

            for fmt in formats:
                suffix, pmg_fmt = FORMAT_MAP[fmt]
                fname  = "POSCAR" if fmt == "vasp" else f"{tag}{suffix}"
                fpath  = run_dir / fname
                struct.to(fmt=pmg_fmt, filename=str(fpath))
                written[fmt] = str(fpath)

            # Shortest adsorbate–slab bond
            dists    = [struct.get_distance(n_slab, i) for i in range(n_slab)]
            min_dist = round(min(dists), 4)

            info = {
                "tag":              tag,
                "site_type":        site_type,
                "site_element":     site_element or "any",
                "site_coords":      site_coords.tolist(),
                "height_input":     h,
                "orientation":      orient_label,
                "actual_bond_dist": min_dist,
                "files":            written,
            }
            manifest.append(info)

            files_str = "  ".join(f"{k}={v}" for k, v in written.items())
            print(
                f"  [OK] {tag} | "
                f"h={h:.2f} Å  orient={orient_label} | "
                f"bond={min_dist:.3f} Å | {files_str}"
            )

    return manifest


# =============================================================================
# Flyte Task
# =============================================================================

@task(container_image=_IMAGE, requests=_RESOURCES, limits=_RESOURCES)
def place_adsorbate_task(
    slab_file:  FlyteFile,
    ads_params: AdsorbateParams       = AdsorbateParams(),
    out_config: AdsorbateOutputConfig = AdsorbateOutputConfig(),
) -> AdsorbateOutput:

    local_path = slab_file.download()
    slab       = _load_structure(local_path)
    formula    = slab.composition.reduced_formula

    # Validate site_type and orientation; adsorbate is validated inside get_adsorbate()
    if ads_params.site_type not in SITE_TYPE_KEYS:
        raise ValueError(f"Unknown site_type '{ads_params.site_type}'. Choose from: {SITE_TYPE_KEYS}")
    if ads_params.orientation not in ORIENTATION_KEYS:
        raise ValueError(f"Unknown orientation '{ads_params.orientation}'. Choose from: {ORIENTATION_KEYS}")

    adsorbate = get_adsorbate(ads_params.adsorbate)
    heights   = _parse_heights(ads_params.height)
    formats   = [f.strip() for f in out_config.output_formats.split(",")]

    orient_labels = (
        list(_ORIENT_SPEC.keys()) if ads_params.orientation == "all"
        else [ads_params.orientation]
    )

    # List all labelled sites — especially useful for alloys/MXenes
    available_sites = list_available_sites(slab)

    print(f"\n  {'='*60}")
    print(f"  Adsorbate placement")
    print(f"  Slab         : {formula}  ({len(slab)} atoms)")
    print(f"  Adsorbate    : {ads_params.adsorbate}")
    print(f"  Site type    : {ads_params.site_type}  [1 representative site]")
    print(f"  Site element : {ads_params.site_element or 'any (first available)'}")
    print(f"  Height(s)    : {heights} Ang")
    print(f"  Orientation  : {ads_params.orientation}  -> {orient_labels}")
    print(f"  Formats      : {formats}")
    print(f"  Total files  : {len(heights) * len(orient_labels)} structure(s)")
    print(f"  Available labelled sites on this surface:")
    for stype, labels in available_sites.items():
        print(f"    {stype}: {labels}")
    print(f"  {'='*60}\n")

    manifest = _place_adsorbates(
        slab           = slab,
        adsorbate      = adsorbate,
        adsorbate_name = ads_params.adsorbate,
        heights        = heights,
        site_type      = ads_params.site_type,
        site_element   = ads_params.site_element,
        orientation    = ads_params.orientation,
        out_prefix     = out_config.output_prefix,
        formats        = formats,
    )

    if not manifest:
        raise RuntimeError("No structures generated. Check site_type is present on this surface.")

    all_files: List[FlyteFile] = []
    for entry in manifest:
        path = entry["files"].get("vasp") or next(iter(entry["files"].values()))
        all_files.append(FlyteFile(path))

    primary     = all_files[0]
    encoded_xyz: Dict[str, str] = {}
    if "extxyz" in manifest[0]["files"]:
        encoded_xyz["extxyz"] = _encode_file(manifest[0]["files"]["extxyz"])

    if out_config.write_metadata:
        manifest_path = Path(f"{out_config.output_prefix}_manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"\n  Manifest written: {manifest_path}")

    meta = {
        "host_formula":    formula,
        "n_slab_atoms":    len(slab),
        "adsorbate":       ads_params.adsorbate,
        "site_type":       ads_params.site_type,
        "site_element":    ads_params.site_element or "any",
        "available_sites": available_sites,
        "heights":         heights,
        "orientation":     ads_params.orientation,
        "formats":         formats,
        "n_structures":    len(manifest),
    }

    summary = (
        f"{len(manifest)} structure(s) | "
        f"{ads_params.adsorbate} on {formula} | "
        f"site={ads_params.site_type} | "
        f"height(s)={heights} Å | "
        f"orientation={ads_params.orientation}"
    )
    print(f"\n  Summary: {summary}\n")

    return AdsorbateOutput(
        primary_structure   = primary,
        all_structure_files = all_files,
        manifest            = manifest,
        host_formula        = formula,
        adsorbate_name      = ads_params.adsorbate,
        n_structures        = len(manifest),
        summary             = summary,
        encoded_xyz         = encoded_xyz,
        metadata            = meta,
    )


# =============================================================================
# Flyte Workflow
# =============================================================================

@workflow
def place_adsorbate_workflow(
    slab_file:  FlyteFile,
    ads_params: AdsorbateParams       = AdsorbateParams(),
    out_config: AdsorbateOutputConfig = AdsorbateOutputConfig(),
) -> AdsorbateOutput:
    return place_adsorbate_task(
        slab_file  = slab_file,
        ads_params = ads_params,
        out_config = out_config,
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Place an adsorbate on a slab at one representative site.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
SITE TYPE  (pick one — 1 representative site used automatically)
  ontop    adsorb directly above a surface atom
  bridge   adsorb between two neighbouring surface atoms
  hollow   adsorb above the centre of a surface hollow

ORIENTATION  (pick one or all)
  vertical    molecule axis along surface normal (z-up)
  horizontal  molecule axis flat in xy-plane
  tilted      molecule axis at 45° to surface normal
  all         all three above

HEIGHT
  Single:  --height 2.0
  Sweep:   --height 1.9,2.0,2.1

EXAMPLES
  # CO vertical on ontop, single height → 1 structure
  python matter_adsorbate.py --slab POSCAR --adsorbate CO \\
      --site-type ontop --orientation vertical --height 2.0 --formats vasp cif

  # CO all orientations on ontop, single height → 3 structures
  python matter_adsorbate.py --slab POSCAR --adsorbate CO \\
      --site-type ontop --orientation all --height 2.0 --formats vasp cif extxyz

  # OH tilted on bridge, height sweep → 3 structures
  python matter_adsorbate.py --slab slab.cif --adsorbate OH \\
      --site-type bridge --orientation tilted --height 1.8,2.0,2.2

  # H on hollow, vertical, single height → 1 structure
  python matter_adsorbate.py --slab POSCAR --adsorbate H \\
      --site-type hollow --orientation vertical --height 1.8

OUTPUT STRUCTURE  (site-type=ontop, orientation=all, height=2.0)
  ads_CO_ontop_h2.00_vertical/POSCAR
  ads_CO_ontop_h2.00_horizontal/POSCAR
  ads_CO_ontop_h2.00_tilted/POSCAR
""",
    )

    parser.add_argument("--slab",        required=True,
                        help="Slab file (POSCAR / CIF / extxyz)")
    parser.add_argument("--adsorbate",   default="CO",
                        help=(
                            "Adsorbate molecule name — anything ase.build.molecule() knows "
                            "(CO, H2O, NH3, CO2, CH4, NO, C6H6, …), "
                            "a single element symbol (O, N, S, …), "
                            "or a built-in fallback (CO, OH, H).  Default: CO"
                        ))
    parser.add_argument("--site-type",    default="ontop",
                        choices=SITE_TYPE_KEYS,
                        help="Site type to adsorb on (default: ontop)")
    parser.add_argument("--site-element", default="",
                        help=(
                            "Element(s) nearest to the desired site — for alloys/MXenes. "
                            "Leave blank to use the first available site of --site-type. "
                            "Examples: 'Mo' (ontop-Mo), 'Mo,Ti' (bridge-Mo/Ti), 'Ti' (ontop-Ti). "
                            "Run without this flag first to see available labelled sites printed "
                            "to stdout, then re-run with the exact element string you want."
                        ))
    parser.add_argument("--height",      default="2.0",
                        help='Height in Å — "2.0" or sweep "1.9,2.0,2.1" (default: 2.0)')
    parser.add_argument("--orientation", default="vertical",
                        choices=ORIENTATION_KEYS,
                        help="Molecular orientation (default: vertical)")
    parser.add_argument("--formats",     nargs="+", default=["vasp"],
                        choices=list(FORMAT_MAP),
                        help="Output formats: vasp cif extxyz xyz (default: vasp)")
    parser.add_argument("--prefix",      default="ads",
                        help="Output file prefix (default: ads)")

    args = parser.parse_args()

    result = place_adsorbate_workflow(
        slab_file  = FlyteFile(args.slab),
        ads_params = AdsorbateParams(
            adsorbate    = args.adsorbate,
            site_type    = args.site_type,
            site_element = args.site_element,
            height       = args.height,
            orientation  = args.orientation,
        ),
        out_config = AdsorbateOutputConfig(
            output_prefix  = args.prefix,
            output_formats = ",".join(args.formats),
        ),
    )

    print(f"  OK  {result.host_formula}")
    print(f"      Adsorbate  : {result.adsorbate_name}")
    print(f"      Structures : {result.n_structures}")
    print(f"      Summary    : {result.summary}\n")