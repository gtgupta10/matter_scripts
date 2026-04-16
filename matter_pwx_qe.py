"""
Step 4: Quantum ESPRESSO Input Generator — Flyte Workflow
=========================================================

Creates QE input files for:

    SCF
    NSCF
    DOS
    Bands
    Relax
    VC-Relax

Features
--------
• accepts CIF / POSCAR / extxyz structures
• automatic slab vs bulk detection
• automatic k-point selection
• pseudopotential auto-download
• PSLibrary / SSSP family support
• Flyte-friendly outputs
"""

from __future__ import annotations
from typing import NamedTuple, Optional, Dict
from dataclasses import dataclass
import os, json, gzip, base64
from pathlib import Path

import numpy as np
import requests

from flytekit import task, workflow, Resources
from flytekit.types.file import FlyteFile

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io.espresso import write_espresso_in
from ase.io import read as ase_read, write as ase_write


# ----------------------------------------------------------
# Flyte resources
# ----------------------------------------------------------

_RESOURCES = Resources(cpu="2", mem="8Gi", gpu="0")
_IMAGE = "guptag13/slab:v5"


# ==========================================================
# Input dataclasses
# ==========================================================

@dataclass
class QECalculation:

    """Calculation type"""

    mode: str = "scf"  
    # scf | nscf | dos | bands | relax | vc-relax


@dataclass
class QEKpoints:

    """Kpoint settings"""

    kx: Optional[int] = None
    ky: Optional[int] = None
    kz: Optional[int] = None

    auto_detect: bool = True


@dataclass
class QEPseudo:

    """Pseudopotential source"""

    family: str = "pslibrary"

    base_url: str = (
        "https://pseudopotentials.quantum-espresso.org/upf_files/"
    )

    suffix: str = ".pbe-spn-kjpaw_psl.1.0.0.UPF"


@dataclass
class QEParameters:
    """
    Quantum ESPRESSO parameters (similar role to INCAR in VASP)
    Covers most common parameters used in SCF / Relax / VC-Relax workflows.
    """

    # -------------------------------------------------
    # CONTROL namelist
    # -------------------------------------------------

    calculation: str = "scf"     # scf | relax | vc-relax | nscf | bands
    restart_mode: str = "from_scratch"
    wf_collect: bool = True
    tstress: bool = True
    tprnfor: bool = True
    verbosity: str = "low"

    etot_conv_thr: float = 1e-4
    forc_conv_thr: float = 1e-3


    # -------------------------------------------------
    # SYSTEM namelist
    # -------------------------------------------------

    ecutwfc: float = 40
    ecutrho: float = 320

    occupations: str = "smearing"
    smearing: str = "mp"
    degauss: float = 0.02

    input_dft: Optional[str] = None   # PBE / PBEsol / BLYP
    vdw_corr: Optional[str] = None    # grimme-d2 / grimme-d3

    nspin: int = 1

    lda_plus_u: bool = False
    Hubbard_U: Optional[dict] = None


    # -------------------------------------------------
    # ELECTRONS namelist
    # -------------------------------------------------

    conv_thr: float = 1e-6
    mixing_beta: float = 0.3
    mixing_mode: str = "plain"
    electron_maxstep: int = 100

    diagonalization: str = "david"


    # -------------------------------------------------
    # IONS namelist (for relax)
    # -------------------------------------------------

    ion_dynamics: str = "bfgs"


    # -------------------------------------------------
    # CELL namelist (for vc-relax)
    # -------------------------------------------------

    cell_dynamics: str = "bfgs"
    press: float = 0.0
    cell_dofree: str = "all"


@dataclass
class QEOutputConfig:

    output_prefix: str = "qe"
    output_formats: str = "extxyz"
    write_metadata: bool = True


# ==========================================================
# Output schema
# ==========================================================

class QEOutput(NamedTuple):

    primary_input: FlyteFile
    run_script: FlyteFile

    host_formula: str
    calculation_mode: str

    encoded_xyz: Dict[str,str]
    metadata: dict



# ==========================================================
# Pseudopotential mapping (PSLibrary)
# ==========================================================

PSLIB_MAP = {
    "H": "H.pbe-kjpaw_psl.1.0.0.UPF",
    "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF",
    "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "N": "N.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Ti": "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Fe": "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Cu": "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF",
}


# ==========================================================
# Helpers
# ==========================================================

def _load_structure(path: str):

    try:
        return Structure.from_file(path)

    except Exception:
        atoms = ase_read(path)
        return AseAtomsAdaptor.get_structure(atoms)


def _pmg_to_ase(struct):

    return AseAtomsAdaptor.get_atoms(struct)


# ----------------------------------------------------------
# slab detection
# ----------------------------------------------------------

def _detect_slab(atoms):

    cell = atoms.cell.lengths()

    if cell[2] > 1.5 * max(cell[0], cell[1]):
        return True

    return False


# ----------------------------------------------------------
# auto kpoints
# ----------------------------------------------------------

def _auto_kpoints(atoms):

    slab = _detect_slab(atoms)

    if slab:
        return (4,4,1)

    return (6,6,6)


# ----------------------------------------------------------
# pseudo download
# ----------------------------------------------------------

def _download_pseudo(el, pseudo_dir, pseudo):

    pseudo_dir.mkdir(parents=True, exist_ok=True)

    if el not in PSLIB_MAP:
        raise RuntimeError(f"No pseudo defined for element {el}")

    fname = PSLIB_MAP[el]

    path = pseudo_dir / fname

    if path.exists():
        return path

    url = f"{pseudo.base_url}{fname}"

    print(f"[QE] downloading {el} → {url}")

    r = requests.get(url)

    if r.status_code != 200:
        raise RuntimeError(f"Failed downloading pseudo {el}")

    path.write_bytes(r.content)

    return path


# ==========================================================
# Main QE generator
# ==========================================================

@task(
    container_image=_IMAGE,
    requests=_RESOURCES,
)
def (

    structure_file: FlyteFile,

    calc: QECalculation,
    kpoints: QEKpoints,
    pseudo: QEPseudo,
    params: QEParameters,
    out_config: QEOutputConfig,

) -> QEOutput:

    structure_file.download()

    struct = _load_structure(structure_file.path)

    atoms = _pmg_to_ase(struct)

    formula = struct.composition.reduced_formula

    run_dir = Path("qe_run")
    run_dir.mkdir(exist_ok=True)

    pseudo_dir = run_dir / "pseudos"

    # -----------------------------------------------------
    # pseudopotentials
    # -----------------------------------------------------

    elements = sorted({a.symbol for a in atoms})

    pseudos = {}

    for el in elements:

        p = _download_pseudo(el, pseudo_dir, pseudo)

        pseudos[el] = p.name


    # -----------------------------------------------------
    # kpoints
    # -----------------------------------------------------

    if kpoints.auto_detect:

        kpts = _auto_kpoints(atoms)

    else:

        kpts = (kpoints.kx, kpoints.ky, kpoints.kz)


    # -----------------------------------------------------
    # QE input sections
    # -----------------------------------------------------

    control = {
        "calculation": calc.mode,
        "restart_mode": params.restart_mode,
        "wf_collect": params.wf_collect,
        "tstress": params.tstress,
        "tprnfor": params.tprnfor,
        "verbosity": params.verbosity,
        "etot_conv_thr": params.etot_conv_thr,
        "forc_conv_thr": params.forc_conv_thr,
        "prefix": out_config.output_prefix,
        "pseudo_dir": "./pseudos",
        "outdir": "./tmp",
    }

    system = {
        "ecutwfc": params.ecutwfc,
        "ecutrho": params.ecutrho,
        "occupations": params.occupations,
        "smearing": params.smearing,
        "degauss": params.degauss,
        "nspin": params.nspin,
    }

    # optional parameters
    if params.input_dft:
        system["input_dft"] = params.input_dft

    if params.vdw_corr:
        system["vdw_corr"] = params.vdw_corr

    if params.lda_plus_u:
        system["lda_plus_u"] = True

    electrons = {
        "conv_thr": params.conv_thr,
        "mixing_beta": params.mixing_beta,
        "mixing_mode": params.mixing_mode,
        "electron_maxstep": params.electron_maxstep,
        "diagonalization": params.diagonalization,
    }

    # -----------------------------------------------------
    # conditional namelists
    # -----------------------------------------------------

    input_data = {
        "control": control,
        "system": system,
        "electrons": electrons,
    }

    # geometry optimization
    if calc.mode == "relax":
        input_data["ions"] = {
            "ion_dynamics": params.ion_dynamics
        }

    # variable cell optimization
    if calc.mode == "vc-relax":
        input_data["ions"] = {
            "ion_dynamics": params.ion_dynamics
        }

        input_data["cell"] = {
            "cell_dynamics": params.cell_dynamics,
            "press": params.press,
            "cell_dofree": params.cell_dofree,
        }

    # -----------------------------------------------------
    # write QE input
    # -----------------------------------------------------

    infile = run_dir / f"{out_config.output_prefix}.in"

    write_espresso_in(
        infile,
        atoms,
        input_data=input_data,
        pseudopotentials=pseudos,
        kpts=kpts,
    )


    # -----------------------------------------------------
    # run script
    # -----------------------------------------------------

    script = run_dir / "run.sh"

    script.write_text(

f"""#!/bin/bash
mkdir -p tmp
mpirun -np 8 pw.x < {infile.name} > {out_config.output_prefix}.out
"""

    )

    script.chmod(0o755)


    # -----------------------------------------------------
    # encode structure
    # -----------------------------------------------------

    xyz = run_dir / "structure.extxyz"

    ase_write(xyz, atoms)

    with open(xyz,"rb") as f:

        encoded = base64.b64encode(
            gzip.compress(f.read())
        ).decode()

    encoded_xyz = {"extxyz": encoded}


    # -----------------------------------------------------
    # metadata
    # -----------------------------------------------------

    meta = {

        "formula": formula,
        "calculation": calc.mode,
        "kpoints": kpts,
        "n_atoms": len(struct),

    }

    if out_config.write_metadata:

        with open(run_dir/"metadata.json","w") as f:
            json.dump(meta,f,indent=2)


    return QEOutput(

        primary_input=FlyteFile(str(infile)),
        run_script=FlyteFile(str(script)),

        host_formula=formula,
        calculation_mode=calc.mode,

        encoded_xyz=encoded_xyz,
        metadata=meta,

    )


# ==========================================================
# Flyte workflow
# ==========================================================

@workflow
def qe_input_generation_wf(

    structure_file: FlyteFile,

    calc: QECalculation,
    kpoints: QEKpoints,
    pseudo: QEPseudo,
    params: QEParameters,
    out_config: QEOutputConfig,

) -> QEOutput:

    return qe_input_generator(

        structure_file=structure_file,

        calc=calc,
        kpoints=kpoints,
        pseudo=pseudo,
        params=params,
        out_config=out_config,

    )

# ==========================================================
# Local CLI testing with argparse
# ==========================================================

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Quantum ESPRESSO input files"
    )

    # ------------------------------------------------------
    # structure
    # ------------------------------------------------------

    parser.add_argument(
        "--structure",
        required=True,
        help="Input structure file (CIF / POSCAR / extxyz)",
    )

    # ------------------------------------------------------
    # calculation mode
    # ------------------------------------------------------

    parser.add_argument(
        "--mode",
        default="scf",
        choices=["scf","nscf","dos","bands","relax","vc-relax"],
        help="QE calculation mode",
    )

    # ------------------------------------------------------
    # kpoints
    # ------------------------------------------------------

    parser.add_argument("--kx", type=int, help="k-point grid X")
    parser.add_argument("--ky", type=int, help="k-point grid Y")
    parser.add_argument("--kz", type=int, help="k-point grid Z")

    parser.add_argument(
        "--auto-kpts",
        action="store_true",
        help="Auto detect slab/bulk kpoints",
    )

    # ------------------------------------------------------
    # cutoffs
    # ------------------------------------------------------

    parser.add_argument(
        "--ecutwfc",
        type=float,
        default=40,
        help="Wavefunction cutoff",
    )

    parser.add_argument(
        "--ecutrho",
        type=float,
        default=320,
        help="Charge density cutoff",
    )

    # ------------------------------------------------------
    # convergence
    # ------------------------------------------------------

    parser.add_argument(
        "--conv-thr",
        type=float,
        default=1e-6,
        help="SCF convergence threshold",
    )

    parser.add_argument(
        "--mixing-beta",
        type=float,
        default=0.3,
        help="Charge mixing beta",
    )

    # ------------------------------------------------------
    # spin
    # ------------------------------------------------------

    parser.add_argument(
        "--nspin",
        type=int,
        default=1,
        choices=[1,2],
        help="Spin polarization (1=nonmagnetic, 2=spin)",
    )

    # ------------------------------------------------------
    # vdW correction
    # ------------------------------------------------------

    parser.add_argument(
        "--vdw",
        default=None,
        choices=[None,"grimme-d2","grimme-d3"],
        help="van der Waals correction",
    )

    # ------------------------------------------------------
    # prefix
    # ------------------------------------------------------

    parser.add_argument(
        "--prefix",
        default="qe_test",
        help="QE output prefix",
    )

    args = parser.parse_args()

    # ------------------------------------------------------
    # Flyte-style dataclasses
    # ------------------------------------------------------

    calc = QECalculation(mode=args.mode)

    kpts = QEKpoints(
        kx=args.kx,
        ky=args.ky,
        kz=args.kz,
        auto_detect=args.auto_kpts,
    )

    pseudo = QEPseudo()

    params = QEParameters(
        calculation=args.mode,
        ecutwfc=args.ecutwfc,
        ecutrho=args.ecutrho,
        conv_thr=args.conv_thr,
        mixing_beta=args.mixing_beta,
        nspin=args.nspin,
        vdw_corr=args.vdw,
    )

    out_config = QEOutputConfig(
        output_prefix=args.prefix
    )

    # ------------------------------------------------------
    # FlyteFile wrapper
    # ------------------------------------------------------

    structure_file = FlyteFile(args.structure)

    # ------------------------------------------------------
    # run generator
    # ------------------------------------------------------

    result = qe_input_generator(
        structure_file=structure_file,
        calc=calc,
        kpoints=kpts,
        pseudo=pseudo,
        params=params,
        out_config=out_config,
    )

    print("\nQE input generated")
    print("Input file:", result.primary_input)
    print("Run script:", result.run_script)
    print("Formula:", result.host_formula)
    print("Calculation:", result.calculation_mode)


# ==========================================================
# CLI entry
# ==========================================================

if __name__ == "__main__":
    main()