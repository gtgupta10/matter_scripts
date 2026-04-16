"""
Work Function Post-Processing — Flyte Workflow
===============================================

Takes the completed pw.x SCF output (outdir + prefix) and generates
inputs for the remaining work-function pipeline:

    Work function  Φ = V_vacuum − E_Fermi

Required QE calculation chain (this module covers steps 2-4 only):
-------------------------------------------------------------------
  [SCF already done by separate code]
  2. pp.x       →  electrostatic potential (v_bare + v_H + v_xc)
  3. average.x  →  planar-average potential → vacuum level
  4. parse_wf.py→  extract Φ = V_vac − E_Fermi

Inputs expected from the upstream SCF task
------------------------------------------
  • scf_outdir   — path to the SCF outdir (the ./tmp folder written by pw.x)
  • scf_prefix   — QE prefix used in the SCF run (must match what pw.x wrote)
  • scf_pw_out   — the pw.x stdout file (contains 'the Fermi energy is ...')
  • cell_angstrom— c-axis length of the slab in Angstrom (for average.x)

Features
--------
• no pw.x call — pure post-processing (pp.x + average.x + parse_wf.py)
• macroscopic averaging window auto-set from cell_angstrom / n_layers
  or overridden by macroscopic_average_window
• full Flyte task / workflow integration
• local CLI testing via argparse (no Flyte cluster needed)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, NamedTuple, Optional

import numpy as np

from flytekit import task, workflow, Resources
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile


# ----------------------------------------------------------
# Flyte resources
# ----------------------------------------------------------

_RESOURCES = Resources(cpu="2", mem="8Gi", gpu="0")
_IMAGE = "guptag13/slab:v5"


# ==========================================================
# Input dataclasses
# ==========================================================

@dataclass
class PPParameters:
    """
    Parameters for the pp.x post-processing step.

    plot_num=11 selects the total electrostatic potential
    (v_bare + v_Hartree + v_xc), which is the correct quantity
    for identifying the vacuum level in a work-function calculation.
    """

    plot_num: int = 11  # 11 = total electrostatic potential (v_bare + v_H + v_xc)


@dataclass
class AverageParameters:
    """
    Parameters for the average.x planar/macroscopic averaging step.

    macroscopic_average_window
        Averaging window in Angstrom.  Should equal one interlayer spacing.
        If None, auto-computed as  cell_c_angstrom / n_layers_estimated.
        Provide an explicit value when auto-detection is unreliable
        (e.g. disordered slabs, very thin films).

    n_layers_estimated
        Number of atomic layers.  Used only when
        macroscopic_average_window is None.  If None, the code parses
        the pw.x output file to count distinct z-planes (0.5 Å tolerance).
    """

    macroscopic_average_window: Optional[float] = None  # Angstrom; None = auto
    n_layers_estimated: Optional[int] = None            # None = parse from pw.x out


@dataclass
class WFPostOutputConfig:
    """Output file and directory settings."""

    output_prefix: str = "wf"       # should match the SCF prefix
    output_dir:    str = "wf_pp"    # directory where pp/avg inputs are written
    write_metadata: bool = True


# ==========================================================
# Output schema
# ==========================================================

class WFPostOutput(NamedTuple):

    pp_input:      FlyteFile   # pp.x input file
    average_input: FlyteFile   # average.x input file
    run_script:    FlyteFile   # bash script: pp.x → average.x → parse_wf.py
    parser_script: FlyteFile   # embedded parse_wf.py

    metadata: dict             # echo of key settings for downstream tasks


# ==========================================================
# Helpers — parse cell geometry from pw.x output
# ==========================================================

def _parse_cell_c_from_pwout(pw_out_text: str) -> Optional[float]:
    """
    Extract the c-axis lattice parameter (Angstrom) from pw.x stdout.

    pw.x prints:
        lattice parameter (alat)  =   X.XXXX  a.u.
        ...
        a(3) = (  0.0000,  0.0000,  Z.ZZZZ )
    We read alat (Bohr) and a(3) z-component (in alat units) → c in Angstrom.
    """
    BOHR_TO_ANG = 0.529177

    alat_m = re.search(
        r"lattice parameter \(alat\)\s*=\s*([\d.]+)\s+a\.u\.", pw_out_text
    )
    a3_m = re.search(
        r"a\(3\)\s*=\s*\(\s*[\d.\-]+\s*,\s*[\d.\-]+\s*,\s*([\d.\-]+)\s*\)",
        pw_out_text,
    )

    if alat_m and a3_m:
        alat_bohr = float(alat_m.group(1))
        a3_z      = float(a3_m.group(1))          # in alat units
        return abs(a3_z) * alat_bohr * BOHR_TO_ANG

    return None


def _parse_n_layers_from_pwout(pw_out_text: str) -> Optional[int]:
    """
    Rough layer count from the ATOMIC_POSITIONS block in pw.x stdout.
    Counts distinct z-planes with 0.5 Å tolerance.
    Returns None if the block cannot be parsed.
    """
    # pw.x echoes atomic positions in Angstrom in the output
    pos_block = re.search(
        r"ATOMIC_POSITIONS\s*\(\w+\)(.*?)(?=\n\s*\n|\Z)",
        pw_out_text,
        re.DOTALL,
    )
    if not pos_block:
        return None

    z_vals = []
    for line in pos_block.group(1).splitlines():
        parts = line.split()
        if len(parts) >= 4:
            try:
                z_vals.append(float(parts[3]))
            except ValueError:
                pass

    if not z_vals:
        return None

    z_sorted = sorted(z_vals)
    layers   = [z_sorted[0]]
    for z in z_sorted[1:]:
        if z - layers[-1] > 0.5:
            layers.append(z)
    return len(layers)


# ==========================================================
# Input file writers
# ==========================================================

def _write_pp_input(run_dir: Path, prefix: str, scf_outdir: str, params: PPParameters) -> Path:
    """
    Write pp.x input to extract the total electrostatic potential.

    plot_num=11 -> v_bare + v_Hartree + v_xc
    This is the correct potential for identifying the vacuum level.

    Points pp.x at the SCF outdir so it reads the converged charge density
    produced by the upstream pw.x run.
    """
    text = f"""&INPUTPP
  prefix      = '{prefix}'
  outdir      = '{scf_outdir}'
  filplot     = '{prefix}.v_tot'
  plot_num    = {params.plot_num}
/

&PLOT
  nfile             = 1
  filepp(1)         = '{prefix}.v_tot'
  weight(1)         = 1.0
  iflag             = 3
  output_format     = 6
  fileout           = '{prefix}.v_tot.cube'
/
"""
    path = run_dir / "pp.in"
    path.write_text(text)
    return path


def _write_average_input(
    run_dir: Path,
    prefix: str,
    cell_c_ang: float,
    n_layers: int,
    params: AverageParameters,
) -> Path:
    """
    Write average.x input for planar and macroscopic averaging.

    average.x reads the filplot written by pp.x and produces
    the z-averaged potential in avg.dat.
    """
    BOHR_PER_ANG = 1.0 / 0.529177

    c_bohr = cell_c_ang * BOHR_PER_ANG

    if params.macroscopic_average_window is not None:
        window_bohr = params.macroscopic_average_window * BOHR_PER_ANG
    else:
        window_bohr = c_bohr / max(n_layers, 1)

    text = f"""1
{prefix}.v_tot
1.0
{int(round(c_bohr))}
3
{window_bohr:.4f}
"""
    path = run_dir / "average.in"
    path.write_text(text)
    return path


# ==========================================================
# Embedded post-processing parser script
# ==========================================================

_PARSER_SCRIPT = '''#!/usr/bin/env python3
"""
parse_wf.py
-----------
Parses avg.dat (from average.x) and the pw.x output to extract:
  - V_vacuum  : electrostatic potential plateau in vacuum region
  - E_Fermi   : from pw.x SCF output
  - Phi = V_vac - E_Fermi  (work function, eV)

Usage:  python parse_wf.py <prefix> [--pw-out <path>]

Arguments
---------
  prefix       QE prefix (used to locate <prefix>.out if --pw-out not given)
  --pw-out     explicit path to the pw.x stdout file
  --avg-out    explicit path to avg.dat (default: ./avg.dat)
"""

import sys, re, os, argparse
import numpy as np

RY_TO_EV = 13.6057039763


def parse_fermi(pw_out: str) -> float:
    pattern = re.compile(r"the Fermi energy is\\s+([-\\d.]+)\\s+ev", re.IGNORECASE)
    matches = pattern.findall(pw_out)
    if not matches:
        raise ValueError("Fermi energy not found in pw.x output.")
    return float(matches[-1])


def parse_average(avg_out: str):
    lines = avg_out.strip().splitlines()
    z_vals, v_vals = [], []
    for line in lines:
        parts = line.split()
        # Handle both 2-column (z, v) and 3-column (z, v1, v2) formats
        # Use first and second columns only
        if len(parts) >= 2:
            try:
                z = float(parts[0])
                v = float(parts[1])
                z_vals.append(z)
                v_vals.append(v)
            except ValueError:
                continue
    if not z_vals:
        raise ValueError("No data found in avg.dat.")
    return np.array(z_vals), np.array(v_vals)


def find_vacuum_level(z: np.ndarray, v: np.ndarray) -> float:
    """Max of macroscopic-averaged potential in top 20% of cell (Ry)."""
    z_min, z_max = z.min(), z.max()
    mask = z > z_min + 0.80 * (z_max - z_min)
    return v[mask].max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="QE prefix")
    ap.add_argument("--pw-out",  default=None, help="Path to pw.x stdout file")
    ap.add_argument("--avg-out", default="avg.dat", help="Path to avg.dat")
    args = ap.parse_args()

    pw_out_file  = args.pw_out or f"{args.prefix}.out"
    avg_out_file = args.avg_out

    for f in [pw_out_file, avg_out_file]:
        if not os.path.exists(f):
            print(f"[WF] ERROR: {f} not found.")
            sys.exit(1)

    e_fermi_ev = parse_fermi(open(pw_out_file).read())
    z, v_ry    = parse_average(open(avg_out_file).read())

    v_vac_ev = find_vacuum_level(z, v_ry) * RY_TO_EV
    phi      = v_vac_ev - e_fermi_ev

    print("=" * 50)
    print(f"  Fermi energy   : {e_fermi_ev:+.4f} eV")
    print(f"  Vacuum level   : {v_vac_ev:+.4f} eV")
    print(f"  Work function  : {phi:.4f} eV")
    print("=" * 50)

    with open("work_function_result.txt", "w") as f:
        f.write(f"Prefix          : {args.prefix}\\n")
        f.write(f"Fermi energy    : {e_fermi_ev:+.4f} eV\\n")
        f.write(f"Vacuum level    : {v_vac_ev:+.4f} eV\\n")
        f.write(f"Work function   : {phi:.4f} eV\\n")

    print("\\nResult saved to work_function_result.txt")


if __name__ == "__main__":
    main()
'''


def _write_parser(run_dir: Path) -> Path:
    path = run_dir / "parse_wf.py"
    path.write_text(_PARSER_SCRIPT)
    path.chmod(0o755)
    return path


# ==========================================================
# Run script (pp.x → average.x → parse_wf.py only)
# ==========================================================

def _write_run_script(
    run_dir: Path,
    prefix: str,
    scf_pw_out: str,
    pp_file: Path,
    avg_file: Path,
    n_cores: int = 4,
) -> Path:
    """
    Bash script that runs steps 2-4 of the work-function pipeline.
    pw.x (step 1) is intentionally absent — it was handled upstream.

    scf_pw_out is the path to the pw.x stdout file (needed by parse_wf.py
    to read the Fermi energy).
    """
    text = f"""#!/bin/bash
# =====================================================
# Work Function Post-Processing Pipeline
# =====================================================
#  [SCF / pw.x already done upstream]
#  1. pp.x      — extract electrostatic potential from SCF outdir
#  2. average.x — planar + macroscopic average of potential
#  3. parse_wf.py — Phi = V_vac - E_Fermi
# =====================================================

set -e

PREFIX="{prefix}"
NP={n_cores}
SCF_PW_OUT="{scf_pw_out}"

echo "============================================"
echo " Work Function Post-Processing: $PREFIX"
echo "============================================"

echo "[1/3] Running pp.x ..."
mpirun -np $NP pp.x < {pp_file.name} > pp.out
echo "      Done."

echo "[2/3] Running average.x ..."
average.x < {avg_file.name} > average.out
echo "      Done."

echo "[3/3] Parsing work function ..."
python3 parse_wf.py "$PREFIX" --pw-out "$SCF_PW_OUT" --avg-out avg.dat

echo ""
echo "Post-processing complete. Results in: work_function_result.txt"
"""
    script = run_dir / "run_pp.sh"
    script.write_text(text)
    script.chmod(0o755)
    return script


# ==========================================================
# Flyte task
# ==========================================================

@task(
    container_image=_IMAGE,
    requests=_RESOURCES,
)
def wf_postprocessing(

    scf_outdir:   FlyteDirectory,  # the ./tmp directory from pw.x
    scf_pw_out:   FlyteFile,       # pw.x stdout file (contains E_Fermi)
    scf_prefix:   str,             # QE prefix used in the SCF run
    cell_c_ang:   float,           # c-axis length of the slab in Angstrom

    pp_params:    PPParameters,
    avg_params:   AverageParameters,
    out_config:   WFPostOutputConfig,

) -> WFPostOutput:
    """
    Generate pp.x, average.x, and parse_wf.py inputs from a completed
    SCF run.  No pw.x is invoked here.

    Parameters
    ----------
    scf_outdir   : Flyte-managed directory pointing to the pw.x outdir
                   (typically called ./tmp in the SCF task).
    scf_pw_out   : Flyte-managed file pointing to the pw.x stdout.
    scf_prefix   : QE prefix string (e.g. "wf", "cu_slab") — must match
                   what pw.x used.
    cell_c_ang   : c-axis lattice parameter in Angstrom.  Pass this from
                   the SCF task's metadata dict or compute it yourself.
    pp_params    : pp.x settings (plot_num etc.).
    avg_params   : average.x settings (window, n_layers).
    out_config   : output directory and prefix for this task's files.
    """

    # ── download Flyte-managed inputs ────────────────────
    scf_outdir.download()
    scf_pw_out.download()

    scf_outdir_path = scf_outdir.path
    scf_pw_out_path = scf_pw_out.path
    pw_out_text     = Path(scf_pw_out_path).read_text()

    prefix  = scf_prefix
    run_dir = Path(out_config.output_dir)
    run_dir.mkdir(exist_ok=True)

    # ── resolve n_layers for average.x window ────────────
    n_layers = avg_params.n_layers_estimated

    if n_layers is None:
        n_layers = _parse_n_layers_from_pwout(pw_out_text)
        if n_layers is None:
            print(
                "[WF-PP] WARNING: could not parse atomic positions from pw.x "
                "output to estimate n_layers. Defaulting to 1 (window = c)."
            )
            n_layers = 1

    # ── optional: cross-check cell_c from pw.x output ────
    cell_c_parsed = _parse_cell_c_from_pwout(pw_out_text)
    if cell_c_parsed is not None:
        diff = abs(cell_c_parsed - cell_c_ang)
        if diff > 0.5:
            print(
                f"[WF-PP] WARNING: supplied cell_c_ang ({cell_c_ang:.3f} Å) "
                f"differs from pw.x output ({cell_c_parsed:.3f} Å) by "
                f"{diff:.3f} Å. Using supplied value."
            )

    # ── write input files ─────────────────────────────────
    pp_file  = _write_pp_input(run_dir, prefix, scf_outdir_path, pp_params)
    avg_file = _write_average_input(run_dir, prefix, cell_c_ang, n_layers, avg_params)
    parser   = _write_parser(run_dir)
    script   = _write_run_script(
        run_dir, prefix, scf_pw_out_path, pp_file, avg_file
    )

    # ── metadata ──────────────────────────────────────────
    meta = {
        "scf_prefix":              prefix,
        "scf_outdir":              scf_outdir_path,
        "scf_pw_out":              scf_pw_out_path,
        "cell_c_angstrom":         cell_c_ang,
        "n_layers_used":           n_layers,
        "plot_num":                pp_params.plot_num,
        "macroscopic_window_ang":  avg_params.macroscopic_average_window,
    }

    if out_config.write_metadata:
        with open(run_dir / "pp_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    return WFPostOutput(
        pp_input      = FlyteFile(str(pp_file)),
        average_input = FlyteFile(str(avg_file)),
        run_script    = FlyteFile(str(script)),
        parser_script = FlyteFile(str(parser)),
        metadata      = meta,
    )


# ==========================================================
# Flyte workflow
# ==========================================================

@workflow
def wf_postprocessing_wf(

    scf_outdir:   FlyteDirectory,
    scf_pw_out:   FlyteFile,
    scf_prefix:   str,
    cell_c_ang:   float,

    pp_params:    PPParameters,
    avg_params:   AverageParameters,
    out_config:   WFPostOutputConfig,

) -> WFPostOutput:

    return wf_postprocessing(

        scf_outdir  = scf_outdir,
        scf_pw_out  = scf_pw_out,
        scf_prefix  = scf_prefix,
        cell_c_ang  = cell_c_ang,

        pp_params   = pp_params,
        avg_params  = avg_params,
        out_config  = out_config,

    )


# ==========================================================
# Local CLI testing (no Flyte cluster needed)
# ==========================================================

def main() -> None:

    ap = argparse.ArgumentParser(
        description=(
            "Generate pp.x + average.x inputs for work-function post-processing.\n"
            "Requires a completed pw.x SCF run (outdir + stdout file).\n"
            "\n"
            "Pipeline produced:\n"
            "  pp.x (potential) -> average.x (planar avg) -> parse_wf.py (Phi)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── SCF inputs ─────────────────────────────────────────
    ap.add_argument(
        "--scf-outdir",
        required=True,
        help="Path to pw.x outdir (the ./tmp folder from the SCF run)",
    )
    ap.add_argument(
        "--scf-pw-out",
        required=True,
        help="Path to pw.x stdout file (contains 'the Fermi energy is ...')",
    )
    ap.add_argument(
        "--scf-prefix",
        required=True,
        help="QE prefix used in the SCF run",
    )
    ap.add_argument(
        "--cell-c",
        type=float,
        required=True,
        help="c-axis length of the slab in Angstrom",
    )

    # ── pp.x ───────────────────────────────────────────────
    ap.add_argument(
        "--plot-num",
        type=int,
        default=11,
        help="pp.x plot_num (default 11 = total electrostatic potential)",
    )

    # ── average.x ──────────────────────────────────────────
    ap.add_argument(
        "--macro-window",
        type=float,
        default=None,
        help="Macroscopic averaging window for average.x (Angstrom). Auto if not set.",
    )
    ap.add_argument(
        "--n-layers",
        type=int,
        default=None,
        help="Number of atomic layers (for auto window). Parsed from pw.x output if not set.",
    )

    # ── output ─────────────────────────────────────────────
    ap.add_argument("--prefix",    default="wf",    help="Output file prefix (default: wf)")
    ap.add_argument("--out-dir",   default="wf_pp", help="Output directory (default: wf_pp)")
    ap.add_argument("--ncores",    type=int, default=8, help="MPI cores for run script")

    args = ap.parse_args()

    # ── resolve absolute paths ─────────────────────────────
    # Flyte's type engine requires absolute paths and existing
    # filesystem objects. For local CLI testing we bypass the
    # @task wrapper entirely and call the core logic directly.

    scf_outdir_path = str(Path(args.scf_outdir).resolve())
    scf_pw_out_path = str(Path(args.scf_pw_out).resolve())

    if not Path(scf_outdir_path).is_dir():
        ap.error(f"--scf-outdir '{args.scf_outdir}' is not a directory")
    if not Path(scf_pw_out_path).is_file():
        ap.error(f"--scf-pw-out '{args.scf_pw_out}' is not a file")

    pw_out_text = Path(scf_pw_out_path).read_text()

    pp_params  = PPParameters(plot_num=args.plot_num)

    avg_params = AverageParameters(
        macroscopic_average_window=args.macro_window,
        n_layers_estimated=args.n_layers,
    )

    out_config = WFPostOutputConfig(
        output_prefix=args.prefix,
        output_dir=args.out_dir,
    )

    # ── run core logic directly (no Flyte type engine) ─────
    prefix  = args.scf_prefix
    run_dir = Path(out_config.output_dir)
    run_dir.mkdir(exist_ok=True)

    # resolve n_layers
    n_layers = avg_params.n_layers_estimated
    if n_layers is None:
        n_layers = _parse_n_layers_from_pwout(pw_out_text)
        if n_layers is None:
            print(
                "[WF-PP] WARNING: could not parse atomic positions from pw.x "
                "output to estimate n_layers. Defaulting to 1 (window = c)."
            )
            n_layers = 1

    # cross-check cell_c
    cell_c_parsed = _parse_cell_c_from_pwout(pw_out_text)
    if cell_c_parsed is not None:
        diff = abs(cell_c_parsed - args.cell_c)
        if diff > 0.5:
            print(
                f"[WF-PP] WARNING: supplied cell_c_ang ({args.cell_c:.3f} Å) "
                f"differs from pw.x output ({cell_c_parsed:.3f} Å) by "
                f"{diff:.3f} Å. Using supplied value."
            )

    pp_file  = _write_pp_input(run_dir, prefix, scf_outdir_path, pp_params)
    avg_file = _write_average_input(run_dir, prefix, args.cell_c, n_layers, avg_params)
    parser   = _write_parser(run_dir)
    script   = _write_run_script(
        run_dir, prefix, scf_pw_out_path, pp_file, avg_file, args.ncores
    )

    meta = {
        "scf_prefix":             prefix,
        "scf_outdir":             scf_outdir_path,
        "scf_pw_out":             scf_pw_out_path,
        "cell_c_angstrom":        args.cell_c,
        "n_layers_used":          n_layers,
        "plot_num":               pp_params.plot_num,
        "macroscopic_window_ang": avg_params.macroscopic_average_window,
    }

    if out_config.write_metadata:
        with open(run_dir / "pp_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    print("\nPost-processing inputs generated")
    print("pp.x input    :", pp_file)
    print("average.x in  :", avg_file)
    print("Run script    :", script)
    print("Metadata      :", meta)


# ==========================================================
# CLI entry
# ==========================================================

if __name__ == "__main__":
    main()