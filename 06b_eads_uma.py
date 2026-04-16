import argparse
import json
import torch
from pathlib import Path

from ase.io import read, write
from ase.optimize import BFGS
from ase.build import molecule
from fairchem.core import pretrained_mlip, FAIRChemCalculator

def load_calc():
    print("Loading UMA (Universal Machine-learning) Small model...")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # The device is actually set here when loading the model
    predictor = pretrained_mlip.get_predict_unit("uma-s-1", device=device)
    
    # Pass only the predictor and task_name to the calculator
    return FAIRChemCalculator(predictor, task_name="oc20")

def relax(atoms, calc, fmax=0.05, steps=300, logfile=None):
    """ASE relaxation loop using the MLIP calculator."""
    atoms = atoms.copy()
    atoms.calc = calc
    # Using BFGS for reliable convergence on surface systems
    opt = BFGS(atoms, logfile=logfile)
    opt.run(fmax=fmax, steps=steps)
    return atoms, atoms.get_potential_energy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ads-manifest", required=True, help="Path to ads_manifest.json")
    parser.add_argument("--clean-slab",   required=True, help="Path to the relaxed clean slab POSCAR")
    parser.add_argument("--outdir",        default="output/eads_equiformer", help="Output directory")
    parser.add_argument("--fmax",           type=float, default=0.05, help="Force convergence threshold")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the SOTA calculator
    calc = load_calc()
    print("[EquiformerV2/OC25] Calculator loaded successfully")

    # ── Reference 1: Clean Slab ──────────────────────────────────────────────
    print("Relaxing clean slab reference...")
    slab = read(args.clean_slab)
    slab_relaxed, e_slab = relax(slab, calc, fmax=args.fmax,
                                  logfile=str(outdir / "slab_relax.log"))
    write(str(outdir / "slab_relaxed.POSCAR"), slab_relaxed, format="vasp")
    print(f"  E(slab) = {e_slab:.6f} eV")

    # ── Reference 2: CO gas molecule ──────────────────────────────────────────
    print("Relaxing gas-phase CO reference...")
    co = molecule("CO")
    co.set_cell([15, 15, 15])
    co.center()
    co.pbc = True
    co_relaxed, e_co = relax(co, calc, fmax=args.fmax,
                               logfile=str(outdir / "co_relax.log"))
    print(f"  E(CO)   = {e_co:.6f} eV")

    # ── Adsorbate relaxations ────────────────────────────────────────────────
    manifest = json.loads(Path(args.ads_manifest).read_text())
    results = []

    for entry in manifest:
        tag = entry["tag"]
        print(f"\n---> Processing Site: {tag}")
        
        atoms = read(entry["poscar"])
        run_dir = outdir / tag
        run_dir.mkdir(exist_ok=True)

        # Run the relaxation
        atoms_relaxed, e_tot = relax(atoms, calc, fmax=args.fmax,
                                      logfile=str(run_dir / "relax.log"))
        
        # Save results
        write(str(run_dir / "POSCAR_relaxed"), atoms_relaxed, format="vasp")

        # Adsorption Energy Calculation: E_ads = E_total - (E_slab + E_molecule)
        e_ads = e_tot - e_slab - e_co
        
        result = {
            **entry,
            "e_total_ev": e_tot,
            "e_slab_ev":  e_slab,
            "e_co_ev":    e_co,
            "e_ads_ev":   e_ads,
            "mlip":       "EquiformerV2-OC25",
        }
        
        (run_dir / "result.json").write_text(json.dumps(result, indent=2))
        results.append(result)
        print(f"  Result: E_ads = {e_ads:+.4f} eV")

    # Final Summary
    results.sort(key=lambda x: x["e_ads_ev"])
    (outdir / "adsorption_energies.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'─'*65}")
    print(f"{'Tag':<42} {'Site':<10} {'E_ads (eV)'}")
    print(f"{'─'*65}")
    for r in results:
        print(f"{r['tag']:<42} {r['site_type']:<10} {r['e_ads_ev']:+.4f}")
    print(f"{'─'*65}")
    
    if results:
        best = results[0]
        print(f"\n★ Most stable site: {best['site_type']} ({best['tag']})")
        print(f"  Min E_ads = {best['e_ads_ev']:.4f} eV")

if __name__ == "__main__":
    main()