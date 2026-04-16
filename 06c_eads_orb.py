"""
CO/Pd(111) adsorption energy using ORB-v2 via ASE.
ORB-v2 is the fastest of the three — good for screening many sites.

Install:
    pip install orb-models

Usage:
    python 06c_eads_orb.py \
        --ads-manifest output/adsorbates/ads_manifest.json \
        --clean-slab output/slabs/L6_V15/POSCAR \
        --outdir output/eads_orb
"""
import argparse
import json
from pathlib import Path

from ase.io import read, write
from ase.optimize import BFGS
from ase.build import molecule
from orb_models.forcefield import pretrained
from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs
from orb_models.forcefield.calculator import ORBCalculator


def load_calc():
    orbff = pretrained.orb_v2()
    return ORBCalculator(orbff, device="cpu")


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
    parser.add_argument("--outdir",        default="output/eads_orb")
    parser.add_argument("--fmax",          type=float, default=0.05)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    calc = load_calc()
    print("[ORB-v2] Calculator loaded")

    # ── Reference: clean slab ─────────────────────────────────────────────────
    slab = read(args.clean_slab)
    slab_relaxed, e_slab = relax(slab, calc, fmax=args.fmax,
                                  logfile=str(outdir / "slab_relax.log"))
    write(str(outdir / "slab_relaxed.POSCAR"), slab_relaxed, format="vasp")
    print(f"  E(slab) = {e_slab:.6f} eV")

    # ── Reference: CO gas ─────────────────────────────────────────────────────
    co = molecule("CO")
    co.set_cell([15, 15, 15])
    co.center()
    co.pbc = True
    co_relaxed, e_co = relax(co, calc, fmax=args.fmax,
                               logfile=str(outdir / "co_relax.log"))
    print(f"  E(CO)   = {e_co:.6f} eV")

    # ── Adsorbate structures ──────────────────────────────────────────────────
    manifest = json.loads(Path(args.ads_manifest).read_text())
    results = []

    for entry in manifest:
        tag = entry["tag"]
        atoms = read(entry["poscar"])
        run_dir = outdir / tag
        run_dir.mkdir(exist_ok=True)

        atoms_relaxed, e_tot = relax(atoms, calc, fmax=args.fmax,
                                      logfile=str(run_dir / "relax.log"))
        write(str(run_dir / "POSCAR_relaxed"), atoms_relaxed, format="vasp")

        e_ads = e_tot - e_slab - e_co
        result = {
            **entry,
            "e_total_ev": e_tot,
            "e_slab_ev":  e_slab,
            "e_co_ev":    e_co,
            "e_ads_ev":   e_ads,
            "mlip":       "ORB-v2",
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
