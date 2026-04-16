"""
Step 4: Defect Surface Generator — Flyte Workflows
===================================================
A production-grade Flyte workflow for creating point defects on surfaces,
following the same architecture as 03_slabgeneration.py.

Defect Types:
  0. Vacancy      : remove one atom of a chosen species/site
  1. Substitution : replace one atom with a different species
  2. Interstitial : insert an atom at a high-symmetry or custom fractional position

Input:
  - POSCAR / CIF / extxyz surface slab file  (output from 03_slabgeneration.py)

Strategy (pymatgen-based, no external defect libraries required):
  - Vacancy / Substitution: uses pymatgen's SymmetricSiteList to find
    symmetrically non-equivalent sites, generates ALL unique defects
    (one per inequivalent site) in a single run.
  - Interstitial: uses VoronoiInterstitialGenerator to find geometrically
    distinct insertion sites automatically, or accepts a user-specified
    fractional coordinate.

Outputs per defect:
  - POSCAR  (VASP format)
  - extxyz  (ASE extended XYZ)
  - CIF
  - JSON metadata  (defect_type, host_formula, defect_species, site_index,
                    site_coords, wyckoff, n_sites, charge_state)


Usage (CLI):
  python defect_gen.py --mode vacancy     --input-file slab_POSCAR --species Ti
  python defect_gen.py --mode substitution --input-file slab_POSCAR --species Ti --sub-species Zr
  python defect_gen.py --mode interstitial --input-file slab_POSCAR --int-species H
  python defect_gen.py --mode interstitial --input-file slab_POSCAR --int-species O --frac-coords "0.5,0.5,0.8"
  python defect_gen.py --mode all          --input-file slab_POSCAR --species Ti --sub-species Zr --int-species H
"""

from __future__ import annotations
from typing import NamedTuple, Optional, List, Dict
from dataclasses import dataclass, field
import os, gzip, base64, json, tempfile
import numpy as np
from flytekit import task, workflow, Resources
from flytekit.types.file import FlyteFile

# ── Shared task resources (mirrors slab generator) ───────────────────────────
_RESOURCES = Resources(cpu="2", mem="8Gi", gpu="0")
_IMAGE     = "guptag13/slab:v5"


# ═════════════════════════════════════════════════════════════════════════════
# Input Dataclasses
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class VacancyParams:
    """
    Generate all symmetrically non-equivalent vacancy defects for a given species.

    species      : element to remove, e.g. "Ti" — leave blank ("") to generate
                   vacancies for ALL species in the slab
    n_jobs       : max number of inequivalent vacancies to generate
                   (0 = all, default 0)
    charge_states: comma-separated charge states to annotate, e.g. "0,-1,-2"
                   (informational only — no charge correction applied here)
    tol          : site-equivalence tolerance in Å (default 0.1)
    """
    species:       str   = ""          # "" → all species
    n_jobs:        int   = 0           # 0 = all inequivalent sites
    charge_states: str   = "0"
    tol:           float = 0.1


@dataclass
class SubstitutionParams:
    """
    Generate substitution defects replacing 'species' with 'sub_species'.

    Two modes controlled by replace_all:

      replace_all=False (default) — per-site mode:
        One output structure per symmetrically inequivalent site.
        Standard point-defect approach for DFT calculations.
        e.g. Al2O3 slab with 12 inequivalent Al → 12 output files,
             each with exactly one Al replaced by Mg.

      replace_all=True — bulk replacement mode:
        One output structure with EVERY occurrence of 'species'
        replaced by 'sub_species' simultaneously.
        e.g. Al2O3 → MgxOy (all Al replaced by Mg in one shot).
        Useful for alloy/doped-compound surface models.

    species     : host element to replace, e.g. "Al"
    sub_species : substituting element,    e.g. "Mg"
    replace_all : if True, replace all occurrences in one structure
    n_jobs      : max inequivalent sites in per-site mode (0 = all, ignored if replace_all)
    charge_states: comma-separated charge states to annotate
    tol         : site-equivalence tolerance in Å (per-site mode only)
    """
    species:       str   = "Al"
    sub_species:   str   = "Mg"
    replace_all:   bool  = False
    n_jobs:        int   = 0
    charge_states: str   = "0"
    tol:           float = 0.1


@dataclass
class InterstitialParams:
    """
    Generate interstitial defects by inserting an atom at high-symmetry voids.

    int_species  : species to insert, e.g. "H", "O", "Li", "N"
    frac_coords  : if set, insert at this specific fractional position
                   e.g. "0.5,0.5,0.75" — skips auto Voronoi search
    n_jobs       : max number of Voronoi sites to generate (0 = all)
    min_dist     : minimum distance from inserted atom to any host atom (Å)
                   sites closer than this are discarded (default 1.0)
    charge_states: comma-separated charge states to annotate
    """
    int_species:   str            = "H"
    frac_coords:   Optional[str]  = None    # "x,y,z" or None for auto Voronoi
    n_jobs:        int            = 0
    min_dist:      float          = 1.0
    charge_states: str            = "0"



@dataclass
class DefectOutputConfig:
    """Output format and file naming settings."""
    output_prefix:  str  = "defect"
    output_formats: str  = "extxyz,vasp,cif"   # comma-separated
    write_metadata: bool = True


# ═════════════════════════════════════════════════════════════════════════════
# Output NamedTuple
# ═════════════════════════════════════════════════════════════════════════════

class DefectOutput(NamedTuple):
    """
    primary_defect_file  : POSCAR of the first (or only) defect structure.
    all_defect_files     : list of FlyteFile for every generated defect POSCAR.
    defect_metadata      : list of metadata dicts, one per defect.
    host_formula         : chemical formula of the host slab.
    n_defects            : total number of defect structures generated.
    defect_type          : "vacancy" | "substitution" | "interstitial" | "all"
    summary              : human-readable summary string.
    encoded_xyz          : base64+gzip extxyz of primary defect.
    metadata             : full metadata dict for primary defect.
    """
    primary_defect_file: FlyteFile
    all_defect_files:    List[FlyteFile]
    defect_metadata:     List[dict]
    host_formula:        str
    n_defects:           int
    defect_type:         str
    summary:             str
    encoded_xyz:         Dict[str, str]
    metadata:            dict


# ═════════════════════════════════════════════════════════════════════════════
# Parse / IO helpers
# ═════════════════════════════════════════════════════════════════════════════

def _load_structure(path: str):
    """Load POSCAR / CIF / extxyz → pymatgen Structure."""
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.io import read as ase_read

    ext = os.path.splitext(path)[1].lower()
    # Try pymatgen native first (handles POSCAR, CIF well)
    try:
        return Structure.from_file(path)
    except Exception:
        pass
    # Fallback: ASE → pymatgen (good for extxyz)
    atoms = ase_read(path)
    return AseAtomsAdaptor.get_structure(atoms)


def _pmg_to_ase(structure):
    """pymatgen Structure → ASE Atoms."""
    from pymatgen.io.ase import AseAtomsAdaptor
    return AseAtomsAdaptor.get_atoms(structure)




def _get_inequivalent_sites(structure, species: str, tol: float = 0.1):
    """
    Return indices of symmetrically non-equivalent sites for a given species.
    Uses pymatgen SpacegroupAnalyzer to group equivalent sites.
    If species is empty string, returns inequivalent sites for ALL species.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    try:
        spa    = SpacegroupAnalyzer(structure, symprec=tol)
        symm   = spa.get_symmetrized_structure()
        groups = symm.equivalent_indices   # list of lists of equivalent site indices
    except Exception:
        # If symmetry determination fails (common for low-symmetry slabs),
        # treat every site as inequivalent
        print("  Warning: symmetry analysis failed — treating all sites as inequivalent")
        groups = [[i] for i in range(len(structure))]

    # Pick one representative per group, filtered by species
    ineq = []
    for group in groups:
        rep_idx = group[0]
        el = structure[rep_idx].species_string
        if species == "" or el == species:
            ineq.append(rep_idx)
    return ineq


def _spacegroup_info(structure) -> str:
    """Return 'symbol,number' string."""
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        spa = SpacegroupAnalyzer(structure, symprec=0.1)
        ds  = spa.get_symmetry_dataset()
        return f"{ds['international']},{ds['number']}"
    except Exception:
        return "unknown,0"


def _write_defect(
    structure,
    label:      str,
    mode:       str,
    defect_idx: int,
    meta_extra: dict,
    out_config: DefectOutputConfig,
):
    """
    Write one defect structure to disk in requested formats.
    Returns {"metadata": {...}, "encoded_xyz": {...}, "files": {...}}
    """
    from ase.io import write

    atoms   = _pmg_to_ase(structure)
    formula = structure.composition.reduced_formula
    n_sites = len(structure)
    cell    = atoms.get_cell().array
    sg_info = _spacegroup_info(structure)

    formats = [f.strip().lower() for f in out_config.output_formats.split(",")]
    format_map = {
        "extxyz": (f"{label}.extxyz", "extxyz"),
        "xyz":    (f"{label}.xyz",    "xyz"),
        "vasp":   (f"{label}_POSCAR", "vasp"),
        "cif":    (f"{label}.cif",    "cif"),
    }

    written_files = {}
    encoded_xyz   = {}

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
        "defect_type":     mode,
        "defect_index":    defect_idx,
        "host_formula":    meta_extra.get("host_formula", ""),
        "defect_formula":  formula,
        "n_sites":         n_sites,
        "spacegroup":      sg_info,
        "cell":            cell.tolist(),
        "files":           written_files,
        **meta_extra,
    }

    if out_config.write_metadata:
        with open(f"{label}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    print(f"  Defect {defect_idx:>3d} | {mode:<14s} | {formula:<20s} | {n_sites:>4d} atoms")
    return {"metadata": meta, "encoded_xyz": encoded_xyz, "files": written_files}


# ═════════════════════════════════════════════════════════════════════════════
# Mode 0: Vacancy
# ═════════════════════════════════════════════════════════════════════════════

def _build_vacancies(
    host: object,          # pymatgen Structure
    params: VacancyParams,
    host_formula: str,
    out_config: DefectOutputConfig,
    prefix: str,
) -> List[dict]:
    """
    Generate one defect structure per symmetrically inequivalent site of params.species.

    Algorithm:
      1. Find all inequivalent site indices for the target species.
      2. For each site: deep-copy host → remove site → write outputs.
    """
    results = []
    ineq_indices = _get_inequivalent_sites(host, params.species, params.tol)

    if not ineq_indices:
        sp_str = params.species or "any"
        print(f"  Warning: no inequivalent sites found for species='{sp_str}'")
        return results

    max_jobs = params.n_jobs if params.n_jobs > 0 else len(ineq_indices)
    charge_states = [int(c.strip()) for c in params.charge_states.split(",")]

    print(f"\n  Vacancy: {len(ineq_indices)} inequivalent site(s) "
          f"for species='{params.species or 'ALL'}'")

    for job_i, site_idx in enumerate(ineq_indices[:max_jobs]):
        site     = host[site_idx]
        el       = site.species_string
        frac     = site.frac_coords.tolist()
        cart     = site.coords.tolist()

        defect   = host.copy()
        defect.remove_sites([site_idx])

        label = (f"{prefix}_vac_{el}_site{site_idx}")
        res = _write_defect(
            structure=defect,
            label=label,
            mode="vacancy",
            defect_idx=job_i,
            meta_extra={
                "host_formula":    host_formula,
                "removed_species": el,
                "site_index_host": site_idx,
                "site_frac_coords":frac,
                "site_cart_coords":cart,
                "charge_states":   charge_states,
            },
            out_config=out_config,
        )
        results.append(res)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Mode 1: Substitution
# ═════════════════════════════════════════════════════════════════════════════

def _build_substitutions(
    host: object,
    params: SubstitutionParams,
    host_formula: str,
    out_config: DefectOutputConfig,
    prefix: str,
) -> List[dict]:
    """
    Replace each inequivalent site of params.species with params.sub_species.

    Algorithm:
      1. Find inequivalent site indices for target species.
      2. For each site: deep-copy host → replace species → write outputs.
    """
    from pymatgen.core import Element

    results      = []
    ineq_indices = _get_inequivalent_sites(host, params.species, params.tol)

    if not ineq_indices:
        print(f"  Warning: no inequivalent sites for species='{params.species}'")
        return results

    max_jobs     = params.n_jobs if params.n_jobs > 0 else len(ineq_indices)
    charge_states = [int(c.strip()) for c in params.charge_states.split(",")]
    sub_el       = Element(params.sub_species)

    print(f"\n  Substitution: {len(ineq_indices)} inequivalent site(s) "
          f"'{params.species}'→'{params.sub_species}'")

    for job_i, site_idx in enumerate(ineq_indices[:max_jobs]):
        site   = host[site_idx]
        el     = site.species_string
        frac   = site.frac_coords.tolist()
        cart   = site.coords.tolist()

        defect = host.copy()
        defect.replace(site_idx, sub_el)

        label = f"{prefix}_sub_{el}→{params.sub_species}_site{site_idx}"
        res = _write_defect(
            structure=defect,
            label=label,
            mode="substitution",
            defect_idx=job_i,
            meta_extra={
                "host_formula":      host_formula,
                "replaced_species":  el,
                "sub_species":       params.sub_species,
                "site_index_host":   site_idx,
                "site_frac_coords":  frac,
                "site_cart_coords":  cart,
                "charge_states":     charge_states,
            },
            out_config=out_config,
        )
        results.append(res)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Mode 2: Interstitial
# ═════════════════════════════════════════════════════════════════════════════

def _build_interstitials(
    host: object,
    params: InterstitialParams,
    host_formula: str,
    out_config: DefectOutputConfig,
    prefix: str,
) -> List[dict]:
    """
    Insert params.int_species at high-symmetry void sites.

    Two strategies:
      A. Manual: if params.frac_coords is set → insert at that single position.
      B. Auto  : use pymatgen VoronoiInterstitialGenerator to enumerate
                 all geometrically distinct void sites, filtered by min_dist.

    Algorithm (auto):
      1. VoronoiInterstitialGenerator finds candidate void centres.
      2. Filter: discard sites closer than params.min_dist to any host atom.
      3. For each surviving site: deep-copy host → append species → write.
    """
    from pymatgen.core import Element, PeriodicSite

    results       = []
    charge_states = [int(c.strip()) for c in params.charge_states.split(",")]
    int_el        = Element(params.int_species)

    # ── Strategy A: user-specified fractional coordinate ─────────────────
    if params.frac_coords is not None:
        try:
            fc = [float(v.strip()) for v in params.frac_coords.split(",")]
            if len(fc) != 3:
                raise ValueError
        except ValueError:
            raise ValueError(
                f"frac_coords must be 3 floats, e.g. '0.5,0.5,0.75', "
                f"got: '{params.frac_coords}'"
            )

        # distance check — use PeriodicSite.distance(other) not get_distance(i, site)
        site     = PeriodicSite(int_el, fc, host.lattice)
        host_sites = list(host)
        min_d    = min(site.distance(s) for s in host_sites)
        if min_d < params.min_dist:
            print(f"  Warning: interstitial at {fc} is only {min_d:.3f} Å "
                  f"from host atom (min_dist={params.min_dist} Å) — still inserting")

        defect = host.copy()
        defect.append(int_el, fc, coords_are_cartesian=False)

        label = f"{prefix}_int_{params.int_species}_manual"
        res = _write_defect(
            structure=defect,
            label=label,
            mode="interstitial",
            defect_idx=0,
            meta_extra={
                "host_formula":       host_formula,
                "inserted_species":   params.int_species,
                "insertion_strategy": "manual",
                "frac_coords":        fc,
                "min_dist_to_host":   round(min_d, 4),
                "charge_states":      charge_states,
            },
            out_config=out_config,
        )
        results.append(res)
        return results

    # ── Strategy B: auto Voronoi void search ─────────────────────────────
    print(f"\n  Interstitial (auto Voronoi): inserting '{params.int_species}' "
          f"into {host_formula}")

    try:
        from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
        gen      = VoronoiInterstitialGenerator(min_dist=params.min_dist)
        sites    = list(gen.generate(host, {params.int_species: 1}))
    except ImportError:
        # Fallback if pymatgen-analysis-defects not installed:
        # find void centres manually via Voronoi nodes
        sites = _voronoi_fallback(host, params.int_species, params.min_dist)

    if not sites:
        print(f"  Warning: no valid interstitial sites found "
              f"(min_dist={params.min_dist} Å may be too large)")
        return results

    max_jobs = params.n_jobs if params.n_jobs > 0 else len(sites)
    print(f"  Found {len(sites)} Voronoi interstitial site(s), "
          f"generating {min(max_jobs, len(sites))}")

    for job_i, site_obj in enumerate(sites[:max_jobs]):
        # VoronoiInterstitialGenerator returns Interstitial defect objects
        # with .defect_structure (pymatgen Structure) and .site (PeriodicSite)
        try:
            defect_struct = site_obj.defect_structure
            fc            = site_obj.site.frac_coords.tolist()
            cart          = site_obj.site.coords.tolist()
        except AttributeError:
            # Fallback format: site_obj is a PeriodicSite directly
            defect = host.copy()
            defect.append(int_el, site_obj.frac_coords, coords_are_cartesian=False)
            defect_struct = defect
            fc   = site_obj.frac_coords.tolist()
            cart = site_obj.coords.tolist()

        label = f"{prefix}_int_{params.int_species}_voronoi{job_i}"
        res = _write_defect(
            structure=defect_struct,
            label=label,
            mode="interstitial",
            defect_idx=job_i,
            meta_extra={
                "host_formula":       host_formula,
                "inserted_species":   params.int_species,
                "insertion_strategy": "voronoi",
                "frac_coords":        fc,
                "cart_coords":        cart,
                "charge_states":      charge_states,
            },
            out_config=out_config,
        )
        results.append(res)

    return results


def _voronoi_fallback(host, int_species: str, min_dist: float):
    """
    Fallback interstitial site finder using scipy Voronoi on Cartesian coords.
    Works without pymatgen-analysis-defects.

    Strategy:
      1. Build image-padded Cartesian point cloud (1 shell of PBC images)
         using only xy-images for slabs (no z-images — vacuum layer is empty).
      2. Run scipy Voronoi → collect vertices inside the unit cell.
      3. Filter: discard vertices closer than min_dist to any host atom.
         Uses PeriodicSite.distance() — correct PBC-aware distance, not
         get_distance(i, j) which takes two INTEGER indices.
      4. Deduplicate vertices within 0.5 Å of each other.
    """
    from scipy.spatial import Voronoi
    from pymatgen.core import PeriodicSite, Element

    print("  (using scipy Voronoi fallback — install pymatgen-analysis-defects "
          "for more robust site enumeration)")

    # ── Build image-padded Cartesian point cloud ──────────────────────────
    # For slabs: only tile in xy (dx, dy ∈ {-1,0,1}), NOT z — the vacuum
    # region is empty and z-images just flood it with spurious void centres.
    cart_points = []
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for site in host:
                shifted_frac = site.frac_coords + np.array([dx, dy, 0])
                cart_points.append(
                    np.dot(shifted_frac, host.lattice.matrix)
                )
    cart_all = np.array(cart_points)

    vor = Voronoi(cart_all)

    # ── Collect vertices inside the unit cell (0 ≤ frac < 1) ─────────────
    inv_lat = np.linalg.inv(host.lattice.matrix)
    candidates = []
    for v in vor.vertices:
        fc = np.dot(v, inv_lat)
        if np.all(fc >= 0) and np.all(fc < 1):
            candidates.append(fc)

    print(f"  Voronoi: {len(candidates)} candidate void sites inside cell")

    # ── Filter by min_dist using PeriodicSite.distance() ─────────────────
    # PeriodicSite.distance(other) is PBC-aware and takes a PeriodicSite,
    # NOT an integer index — this was the bug in the previous version.
    el    = Element(int_species)
    valid = []
    host_sites = list(host)   # materialise once, avoids repeated __getitem__

    for fc in candidates:
        ps   = PeriodicSite(el, fc, host.lattice)
        dmin = min(ps.distance(s) for s in host_sites)
        if dmin >= min_dist:
            valid.append(ps)

    print(f"  After min_dist={min_dist} Å filter: {len(valid)} site(s) remain")

    # ── Deduplicate: remove sites within 0.5 Å of each other ─────────────
    unique = []
    for ps in valid:
        too_close = any(
            np.linalg.norm(ps.coords - u.coords) < 0.5
            for u in unique
        )
        if not too_close:
            unique.append(ps)

    print(f"  After deduplication: {len(unique)} unique interstitial site(s)")
    return unique


# ═════════════════════════════════════════════════════════════════════════════
# Finalise: collect all results → DefectOutput
# ═════════════════════════════════════════════════════════════════════════════

def _finalise_defects(
    all_results:  List[dict],
    defect_type:  str,
    host_formula: str,
    out_config:   DefectOutputConfig,
) -> DefectOutput:
    """
    Package all defect results into a DefectOutput NamedTuple.
    """
    if not all_results:
        raise ValueError(
            f"No defect structures were generated for mode='{defect_type}'. "
            f"Check species names and input structure."
        )

    n = len(all_results)
    primary = all_results[0]

    # Primary POSCAR
    primary_poscar_path = primary["files"].get("vasp") or list(primary["files"].values())[0]

    # All POSCAR files
    all_files = []
    for res in all_results:
        p = res["files"].get("vasp") or list(res["files"].values())[0]
        all_files.append(FlyteFile(p))

    all_meta   = [r["metadata"]    for r in all_results]
    enc_xyz    = primary.get("encoded_xyz", {})

    summary = (
        f"Defect type    : {defect_type}\n"
        f"Host formula   : {host_formula}\n"
        f"N defects      : {n}\n"
        f"Output prefix  : {out_config.output_prefix}\n"
        f"Formats        : {out_config.output_formats}\n"
    )
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(summary)

    return DefectOutput(
        primary_defect_file=FlyteFile(primary_poscar_path),
        all_defect_files=all_files,
        defect_metadata=all_meta,
        host_formula=host_formula,
        n_defects=n,
        defect_type=defect_type,
        summary=summary,
        encoded_xyz=enc_xyz,
        metadata=primary["metadata"],
    )


# ═════════════════════════════════════════════════════════════════════════════
# Flyte Task
# ═════════════════════════════════════════════════════════════════════════════

@task(container_image=_IMAGE, requests=_RESOURCES, limits=_RESOURCES)
def generate_defect_task(
    input_file:        FlyteFile,
    vacancy_params:    Optional[VacancyParams]      = None,
    sub_params:        Optional[SubstitutionParams] = None,
    int_params:        Optional[InterstitialParams] = None,
    out_config:        DefectOutputConfig           = DefectOutputConfig(),
) -> DefectOutput:
    """
    Main Flyte task. Accepts one or more defect param sets.
    If multiple are set, all types are generated and concatenated
    (useful for --mode all).

    Input precedence:
      vacancy_params    → vacancy defects only
      sub_params        → substitution defects only
      int_params        → interstitial defects only
      all three set     → all defect types combined
    """
    print(f"\n{'='*60}")
    print(f"  Mode: DEFECT GENERATION")
    print(f"{'='*60}")

    # ── Load structure ────────────────────────────────────────────────────
    local_path = input_file.download()
    host_raw   = _load_structure(local_path)

    print(f"\n  Host structure loaded")
    print(f"    Formula   : {host_raw.composition.reduced_formula}")
    print(f"    Sites     : {len(host_raw)}")
    host_formula = host_raw.composition.reduced_formula

    prefix    = out_config.output_prefix
    all_res   = []
    mode_used = []

    # ── Vacancy ──────────────────────────────────────────────────────────
    if vacancy_params is not None:
        print(f"\n{'─'*60}")
        print(f"  Generating VACANCY defects")
        print(f"{'─'*60}")
        res = _build_vacancies(host_raw, vacancy_params, host_formula, out_config,
                               prefix=f"{prefix}_v")
        all_res.extend(res)
        mode_used.append("vacancy")

    # ── Substitution ─────────────────────────────────────────────────────
    if sub_params is not None:
        print(f"\n{'─'*60}")
        print(f"  Generating SUBSTITUTION defects")
        print(f"{'─'*60}")
        res = _build_substitutions(host_raw, sub_params, host_formula, out_config,
                                   prefix=f"{prefix}_s")
        all_res.extend(res)
        mode_used.append("substitution")

    # ── Interstitial ─────────────────────────────────────────────────────
    if int_params is not None:
        print(f"\n{'─'*60}")
        print(f"  Generating INTERSTITIAL defects")
        print(f"{'─'*60}")
        res = _build_interstitials(host_raw, int_params, host_formula, out_config,
                                   prefix=f"{prefix}_i")
        all_res.extend(res)
        mode_used.append("interstitial")

    if not mode_used:
        raise ValueError(
            "No defect params provided. Supply at least one of: "
            "vacancy_params, sub_params, int_params."
        )

    defect_type = "+".join(mode_used) if len(mode_used) > 1 else mode_used[0]
    return _finalise_defects(all_res, defect_type, host_formula, out_config)


# ═════════════════════════════════════════════════════════════════════════════
# Flyte Workflow
# ═════════════════════════════════════════════════════════════════════════════

@workflow
def generate_defect_workflow(
    input_file:     FlyteFile,
    vacancy_params: Optional[VacancyParams]      = None,
    sub_params:     Optional[SubstitutionParams] = None,
    int_params:     Optional[InterstitialParams] = None,
    out_config:     DefectOutputConfig           = DefectOutputConfig(),
) -> DefectOutput:
    return generate_defect_task(
        input_file=input_file,
        vacancy_params=vacancy_params,
        sub_params=sub_params,
        int_params=int_params,
        out_config=out_config,
    )


# ═════════════════════════════════════════════════════════════════════════════
# CLI — test each mode from terminal
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    epilog = """
Examples:

  # Vacancy — all inequivalent Ti sites
  python defect_gen.py --mode vacancy --input-file slab_POSCAR --species Ti

  # Vacancy — all species in slab
  python defect_gen.py --mode vacancy --input-file slab_POSCAR --species ""

  # Substitution — Ti → Zr
  python defect_gen.py --mode substitution --input-file slab_POSCAR --species Ti --sub-species Zr

  # Interstitial — auto Voronoi (H)
  python defect_gen.py --mode interstitial --input-file slab_POSCAR --int-species H

  # Interstitial — specific fractional coordinate
  python defect_gen.py --mode interstitial --input-file slab_POSCAR --int-species O --frac-coords "0.5,0.5,0.8"

  # All three defect types in one run
  python defect_gen.py --mode all --input-file slab_POSCAR --species Ti --sub-species Zr --int-species H

  # Charge states annotated in metadata
  python defect_gen.py --mode vacancy --input-file slab_POSCAR --species Ti --charge-states "0,-1,-2,-3"

  # CIF input
  python defect_gen.py --mode substitution --input-file ZnO_slab.cif --species Zn --sub-species Mg

  # extxyz input
  python defect_gen.py --mode interstitial --input-file surface.extxyz --int-species Li --min-dist 1.5
"""

    parser = argparse.ArgumentParser(
        description="Defect Surface Generator (Vacancy / Substitution / Interstitial)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )

    parser.add_argument("--mode", required=True,
                        choices=["vacancy", "substitution", "interstitial", "all"],
                        help="Defect type to generate")
    parser.add_argument("--input-file", required=True,
                        help="Host slab structure file (POSCAR, CIF, extxyz)")

    # Vacancy / Substitution shared
    parser.add_argument("--species", default="",
                        help="Host species to remove/replace, e.g. 'Ti' "
                             "(vacancy/substitution). Empty = all species.")
    parser.add_argument("--n-jobs", type=int, default=0,
                        help="Max inequivalent defects per type (0=all)")
    parser.add_argument("--tol", type=float, default=0.1,
                        help="Site equivalence tolerance in Å (default 0.1)")
    parser.add_argument("--charge-states", default="0",
                        help="Charge states to annotate in metadata, e.g. '0,-1,-2'")

    # Substitution
    parser.add_argument("--sub-species", default="Zr",
                        help="Substituting species for substitution mode")

    # Interstitial
    parser.add_argument("--int-species", default="H",
                        help="Species to insert for interstitial mode")
    parser.add_argument("--frac-coords", default=None,
                        help="Manual insertion fractional coords, e.g. '0.5,0.5,0.75'")
    parser.add_argument("--min-dist", type=float, default=1.0,
                        help="Min distance from inserted atom to host atoms (Å)")


    # Output
    parser.add_argument("--prefix", default="defect", help="Output file prefix")
    parser.add_argument("--formats", default="extxyz,vasp,cif",
                        help="Comma-separated output formats (extxyz,vasp,cif)")

    args = parser.parse_args()

    # ── Build param objects ───────────────────────────────────────────────
    OUT = DefectOutputConfig(output_prefix=args.prefix, output_formats=args.formats)

    kwargs = dict(
        input_file=FlyteFile(args.input_file),
        out_config=OUT,
    )

    if args.mode in ("vacancy", "all"):
        kwargs["vacancy_params"] = VacancyParams(
            species=args.species,
            n_jobs=args.n_jobs,
            charge_states=args.charge_states,
            tol=args.tol,
        )

    if args.mode in ("substitution", "all"):
        kwargs["sub_params"] = SubstitutionParams(
            species=args.species,
            sub_species=args.sub_species,
            n_jobs=args.n_jobs,
            charge_states=args.charge_states,
            tol=args.tol,
        )

    if args.mode in ("interstitial", "all"):
        kwargs["int_params"] = InterstitialParams(
            int_species=args.int_species,
            frac_coords=args.frac_coords,
            n_jobs=args.n_jobs,
            min_dist=args.min_dist,
            charge_states=args.charge_states,
        )

    r = generate_defect_workflow(**kwargs)

    print(f"\n{'='*60}")
    print(f"  OK  {r.host_formula}")
    print(f"      Defect type  : {r.defect_type}")
    print(f"      N defects    : {r.n_defects}")
    print(f"      Primary file : {r.primary_defect_file.path}")
    print(f"{'='*60}")