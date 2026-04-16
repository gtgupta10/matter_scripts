"""
CO/Pd(111) adsorption energy using MACE-MP-0b3 via ASE.

Install:
    pip install mace-torch

Usage:
    python 06a_eads_mace.py \
        --ads-manifest output/adsorbates/ads_manifest.json \
        --clean-slab output/slabs/L6_V15/POSCAR \
        --outdir output/eads_mace
"""

# ───────────────────────────────────────────────────────────────
# Torch compiler compatibility shim (fix for PyTorch ≥ 2.2)
# ───────────────────────────────────────────────────────────────
import torch
import types

if not hasattr(torch, "compiler"):
    torch.compiler = types.SimpleNamespace(is_compiling=lambda: False)
elif not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

# ───────────────────────────────────────────────────────────────

import argparse
import json
from pathlib import Path

from ase.io import read, write
from ase.optimize import BFGS
from ase.build import molecule
from mace.calculators import mace_mp


def load_calc():
    # 0b3 is trained on OMat24 — best for surfaces
    return mace_mp(model="medium-0b3", dispersion=False, default_dtype="float64")


def relax(atoms, calc, fmax=0.05, steps=300, logfile=None):
    atoms = atoms.copy()
    atoms.calc = calc
    opt = BFGS(atoms, logfile=logfile)
    opt.run(fmax=fmax, steps=steps)
    return atoms, atoms.get_potential_energy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ads-manifest", required=True)
    parser.add_argument("--clean-slab",   required=True)
    parser.add_argument("--outdir",        default="output/eads_mace")
    parser.add_argument("--fmax",          type=float, default=0.05)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    calc = load_calc()
    print("[MACE-MP-0b3] Calculator loaded")

    # ── Reference: clean slab ─────────────────────────────────────────────────
    slab = read(args.clean_slab)
    slab_relaxed, e_slab = relax(
        slab, calc, fmax=args.fmax,
        logfile=str(outdir / "slab_relax.log")
    )
    write(str(outdir / "slab_relaxed.POSCAR"), slab_relaxed, format="vasp")
    print(f"  E(slab) = {e_slab:.6f} eV")

    # ── Reference: CO gas (isolated molecule in box) ──────────────────────────
    co = molecule("CO")
    co.set_cell([15, 15, 15])
    co.center()
    co.pbc = True
    co_relaxed, e_co = relax(
        co, calc, fmax=args.fmax,
        logfile=str(outdir / "co_relax.log")
    )
    print(f"  E(CO)   = {e_co:.6f} eV")

    # ── Adsorbate structures ──────────────────────────────────────────────────
    manifest = json.loads(Path(args.ads_manifest).read_text())
    results = []

    for entry in manifest:
        tag = entry["tag"]
        atoms = read(entry["poscar"])
        run_dir = outdir / tag
        run_dir.mkdir(exist_ok=True)

        atoms_relaxed, e_tot = relax(
            atoms, calc, fmax=args.fmax,
            logfile=str(run_dir / "relax.log")
        )
        write(str(run_dir / "POSCAR_relaxed"), atoms_relaxed, format="vasp")

        e_ads = e_tot - e_slab - e_co
        result = {
            **entry,
            "e_total_ev":  e_tot,
            "e_slab_ev":   e_slab,
            "e_co_ev":     e_co,
            "e_ads_ev":    e_ads,
            "mlip":        "MACE-MP-0b3",
        }
        (run_dir / "result.json").write_text(json.dumps(result, indent=2))
        results.append(result)
        print(f"  {tag:45s} E_ads = {e_ads:+.4f} eV")

    results.sort(key=lambda x: x["e_ads_ev"])
    (outdir / "adsorption_energies.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'─'*65}")
    print(f"{'Tag':<42} {'Site':<10} {'E_ads (eV)'}")
    print(f"{'─'*65}")
    for r in results:
        print(f"{r['tag']:<42} {r['site_type']:<10} {r['e_ads_ev']:+.4f}")
    print(f"{'─'*65}")
    if results:
        b = results[0]
        print(f"\n★  Most stable [{b['site_type']}]  E_ads = {b['e_ads_ev']:.4f} eV")


if __name__ == "__main__":
    main()