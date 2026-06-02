# `phase.py` — End-to-End Ab-Initio Phase Diagram Workflow

## Overview

`phase.py` is an automated, ab-initio pipeline for constructing surface phase diagrams of adsorbate–metal systems (e.g., O/Cu(111), H/Cu(111)). It orchestrates the full computational workflow from on-the-fly surface structure generation through DFT relaxation, phonon calculations, and multi-reservoir thermodynamic analysis — all scheduled via [Flyte](https://flyte.org/) for distributed, containerized execution.

The central scientific question answered is: **at a given temperature and gas-phase pressure, which surface coverage is thermodynamically most stable?**

---

## Physics Background

### 1. Surface Slab Model

The metal surface is represented as a periodic **FCC(111) slab**:
- A 2×2 supercell of the FCC(111) surface is built using ASE (`fcc111`)
- Bottom layers are fixed to mimic bulk constraints (`FixAtoms`)
- A vacuum layer (~12 Å default) separates periodic images along the surface normal

Adsorbate coverages from 0 (clean) to 1 ML (monolayer) are generated at FCC hollow sites:

| Label        | Coverage | Adsorbates (out of 4 FCC sites) |
|--------------|----------|--------------------------------|
| `clean`      | 0 ML     | 0                              |
| `0.25ML_fcc` | 0.25 ML  | 1                              |
| `0.50ML_fcc` | 0.50 ML  | 2                              |
| `0.75ML_fcc` | 0.75 ML  | 3                              |
| `1.00ML_fcc` | 1.00 ML  | 4                              |

---

### 2. DFT Relaxation (Quantum ESPRESSO)

Each structure is geometrically relaxed using **Quantum ESPRESSO** (`pw.x`):
- **Exchange-correlation**: GGA (PBE implied by pseudopotential choice) + **DFT-D3** van der Waals correction
- **Basis set**: Plane waves with energy cutoffs `ecutwfc` (default 40 Ry) and `ecutrho` (default 320 Ry)
- **k-point mesh**: 6×6×1 Monkhorst–Pack for slabs; Γ-only for gas-phase molecules
- **Smearing**: Methfessel–Paxton (`mv`) with `degauss = 0.02 Ry` (metallic systems)
- **Convergence thresholds**: Energy `1×10⁻⁸ Ry`, Force `1×10⁻⁴ Ry/Bohr`

The total DFT energy E_DFT is extracted from the `!  total energy` line in the `pw.x` output (in Ry, converted to eV).

---

### 3. Phonon Calculations (Phonopy + finite differences)

Vibrational (phonon) contributions are computed via the **finite displacement method** using [Phonopy](https://phonopy.github.io/phonopy/):

1. The relaxed structure is displaced atom-by-atom by a small distance (default 0.03 Å)
2. Single-point SCF calculations are run for each displaced supercell
3. Forces are parsed from `pw.x` outputs (converted from Ry/Bohr to eV/Å × 25.711)
4. Force constants are assembled; phonon frequencies are obtained at the Γ-point (gas) or a full k-mesh (slabs)

The vibrational **Helmholtz free energy** at temperature T is:

```
F_vib(T) = ZPE + kT Σ_ν ln(1 - exp(-hν/kT))
```

where ν are the phonon frequencies. Frequencies below 50 cm⁻¹ are excluded (acoustic/spurious modes).

For gas molecules, the **ideal-gas chemical potential** is computed from statistical mechanics:

```
μ(T, P) = E_DFT + ZPE + F_vib,therm - kT ln(q_trans · q_rot · g_spin)
```

where:
- `q_trans = (2πmkT/h²)^(3/2) · V` — translational partition function (V = kT/P)
- `q_rot` — rotational partition function (linear or non-linear molecule)
- `g_spin = 2S+1` — spin degeneracy (e.g., 3 for O₂)
- `σ` — molecular symmetry number (e.g., 2 for H₂, O₂, N₂; 1 for CO)

---

### 4. Multi-Reservoir Thermodynamics (Phase Diagram Assembly)

The surface Gibbs free energy of adsorption relative to the clean slab is:

```
ΔG(T, P) = [E_slab(θ) - E_clean + F_vib,slab(T) - F_vib,clean(T)] - N_ads · μ_ads(T, P)
```

where:
- `E_slab(θ)` — DFT energy of slab at coverage θ
- `N_ads` — number of adsorbate atoms in the supercell
- `μ_ads(T, P)` — chemical potential of the adsorbate element, derived from the gas-phase molecule (e.g., μ_O = μ(O₂)/2)

`ΔG` is normalized per unit cell area (Å²) when `normalization = "area"` (default). The phase with the **most negative ΔG** at each (T, P) point is the thermodynamically stable phase.

---

## Code Architecture

```
phase.py
├── Constants & Lattice Parameters
├── Dataclasses (config, parameters, I/O structs)
├── Pure Helpers
│   ├── _make_clean_slab()            — ASE slab builder
│   ├── _add_adsorbate_coverage()     — FCC site adsorbate placement
│   ├── _ideal_gas_mu()               — statistical-mechanics chemical potential
│   ├── _parse_pw_energy()            — parse QE total energy
│   └── _parse_pw_forces()            — parse QE atomic forces
├── Flyte Tasks (6 stages)
│   ├── Task 0: generate_structures_task
│   ├── Task 1: structure_prep_task
│   ├── Task 2: relax_task
│   ├── Task 3: phonopy_displacements_task
│   ├── Task 4: run_displacement_task
│   ├── Task 5: phonopy_free_energy_task
│   └── Task 6: phase_diagram_task
├── Dynamic Workflow: run_phase_diagram_dynamic
└── Main Workflow: integrated_phase_diagram_workflow
```

### Key Dataclasses

| Dataclass             | Purpose                                                        |
|-----------------------|----------------------------------------------------------------|
| `QEParams`            | All Quantum ESPRESSO input parameters                         |
| `PhononParams`        | Phonopy supercell, displacement, and mesh settings            |
| `PhaseDiagramParams`  | Temperature grid, pressure range, normalization, output prefix|
| `ConfigEntry`         | Per-structure metadata (label, role, stoichiometry, gas data) |
| `RelaxOutput`         | DFT energy, relaxed geometry, mass, moments of inertia        |
| `PhononOutput`        | Force sets, vibrational free energies, ZPE, frequencies       |
| `PhaseDiagramOutput`  | CSV/JSON/TXT results, optional PNG plot                       |

---

## Inputs

### CLI Arguments (direct script execution)

| Argument        | Type    | Default                    | Description                                      |
|-----------------|---------|----------------------------|--------------------------------------------------|
| `--metal`       | str     | `Cu`                       | FCC metal symbol (Cu, Ni, Pd, Ag, Au, Al)        |
| `--layers`      | int     | `3`                        | Number of atomic layers in the slab              |
| `--vacuum`      | float   | `12.0`                     | Vacuum thickness above slab (Å)                  |
| `--adsorbate`   | str     | `H`                        | Atomic symbol for the adsorbate (O, H, N, C, …)  |
| `--gas`         | str     | `H2`                       | Gas molecule acting as adsorbate reservoir        |
| `--temperatures`| str     | `"200,300,400,500,600"`    | Comma-separated list of temperatures (K)         |
| `--pseudo-dir`  | str     | `"./pseudos"`              | Directory containing pseudopotential files       |
| `--mpi-procs`   | int     | `8`                        | Number of MPI processes for `pw.x`               |

### Workflow Parameters (Flyte / Python API)

Additional parameters are passed via dataclasses when calling `integrated_phase_diagram_workflow()`:

**`QEParams`**
| Field            | Default     | Description                              |
|------------------|-------------|------------------------------------------|
| `pseudo_dir`     | `./pseudos` | Pseudopotential directory                |
| `ecutwfc`        | `40.0`      | Wavefunction cutoff (Ry)                 |
| `ecutrho`        | `320.0`     | Charge density cutoff (Ry)               |
| `kpoints`        | `6 6 1 0 0 0` | Monkhorst–Pack grid for slabs          |
| `smearing`       | `mv`        | Smearing type (Methfessel–Paxton)        |
| `degauss`        | `0.02`      | Smearing width (Ry)                      |
| `conv_thr`       | `1e-8`      | SCF energy convergence (Ry)              |
| `forc_conv_thr`  | `1e-4`      | Force convergence threshold (Ry/Bohr)    |
| `mixing_beta`    | `0.3`       | Charge mixing parameter                  |
| `vdw_corr`       | `dft-d3`    | Van der Waals correction scheme          |
| `nstep`          | `200`       | Maximum ionic steps                      |
| `mpi_procs`      | `4`         | MPI processes per task                   |
| `pw_exe`         | `pw.x`      | Path/name of the QE executable           |

**`PhononParams`**
| Field               | Default    | Description                                      |
|---------------------|------------|--------------------------------------------------|
| `supercell_matrix`  | `2 2 1`    | Phonopy supercell expansion                      |
| `displacement_dist` | `0.03`     | Atomic displacement distance (Å)                 |
| `mesh`              | `20 20 1`  | q-point mesh for slab thermal properties         |
| `is_symmetry`       | `True`     | Use symmetry to reduce displacements             |

**`PhaseDiagramParams`**
| Field            | Default                         | Description                                          |
|------------------|---------------------------------|------------------------------------------------------|
| `temperatures_K` | `200,300,...,900`               | Temperature grid (K), comma-separated                |
| `logP_species`   | `""`                            | Species for pressure axis (reserved for future use)  |
| `logP_min/max`   | `-15.0` / `5.0`                 | log₁₀ pressure range (Pa)                           |
| `logP_step`      | `0.25`                          | log₁₀ pressure step                                 |
| `normalization`  | `area`                          | Normalize ΔG by: `area` (Å²) or `adsorbate`        |
| `output_prefix`  | `phase_diagram`                 | Prefix for output file names                         |

---

## Outputs

| File / Field             | Format   | Description                                                |
|--------------------------|----------|------------------------------------------------------------|
| `results_csv`            | CSV      | ΔG(T) for all phases; columns: `T_K`, one per phase label |
| `results_json`           | JSON     | Same data as CSV (currently same file path)               |
| `summary_txt`            | TXT      | One-line summary of phases and references found            |
| `plot_png`               | PNG      | Phase diagram plot (currently `None`, placeholder)        |
| `summary`                | str      | Printed to stdout when run as `__main__`                  |

### CSV Output Example

```
T_K,clean,0.25ML_fcc,0.50ML_fcc,0.75ML_fcc,1.00ML_fcc
200,-0.0000,-0.0312,-0.0418,-0.0395,-0.0271
300,-0.0000,-0.0289,-0.0376,-0.0341,-0.0198
...
```

---

## Example CLI Invocations

### Oxygen on Copper(111) at multiple temperatures

```bash
python phase.py \
  --metal Cu \
  --adsorbate O \
  --gas O2 \
  --layers 4 \
  --vacuum 15.0 \
  --temperatures "300,400,500,600,700,800" \
  --pseudo-dir ./pseudos \
  --mpi-procs 8
```

### Hydrogen on Nickel(111), quick test (3 temperatures)

```bash
python phase.py \
  --metal Ni \
  --adsorbate H \
  --gas H2 \
  --layers 3 \
  --temperatures "300,500,700" \
  --pseudo-dir ./pseudos \
  --mpi-procs 4
```

### Nitrogen on Palladium(111)

```bash
python phase.py \
  --metal Pd \
  --adsorbate N \
  --gas N2 \
  --layers 4 \
  --temperatures "400,600,800,1000" \
  --pseudo-dir ./pseudos \
  --mpi-procs 16
```

### Default run (H/Cu, all defaults)

```bash
python phase.py
```

---

## Example JSON Inputs (Flyte / Python API)

When calling the workflow programmatically or via a Flyte launch form, parameters are structured as JSON:

### Minimal (O/Cu, defaults elsewhere)

```json
{
  "metal": "Cu",
  "adsorbate": "O",
  "gas": "O2",
  "layers": 3,
  "vacuum": 12.0,
  "qe_params": {
    "pseudo_dir": "./pseudos",
    "ecutwfc": 40.0,
    "ecutrho": 320.0,
    "mpi_procs": 8
  },
  "pd_params": {
    "temperatures_K": "300,400,500,600,700"
  }
}
```

### High-accuracy run (CO/Pd, tighter convergence, more layers)

```json
{
  "metal": "Pd",
  "adsorbate": "C",
  "gas": "CO",
  "layers": 5,
  "vacuum": 15.0,
  "qe_params": {
    "pseudo_dir": "/shared/pseudos",
    "ecutwfc": 60.0,
    "ecutrho": 480.0,
    "kpoints": "8 8 1 0 0 0",
    "conv_thr": 1e-10,
    "forc_conv_thr": 1e-5,
    "mpi_procs": 32
  },
  "ph_params": {
    "supercell_matrix": "3 3 1",
    "displacement_dist": 0.02,
    "mesh": "30 30 1"
  },
  "pd_params": {
    "temperatures_K": "200,300,400,500,600,700,800,900,1000",
    "normalization": "area",
    "output_prefix": "CO_Pd111"
  }
}
```

### Hydrogen/Nickel quick test (minimal phonon cost)

```json
{
  "metal": "Ni",
  "adsorbate": "H",
  "gas": "H2",
  "layers": 3,
  "vacuum": 12.0,
  "qe_params": {
    "pseudo_dir": "./pseudos",
    "mpi_procs": 4
  },
  "ph_params": {
    "supercell_matrix": "2 2 1",
    "mesh": "10 10 1"
  },
  "pd_params": {
    "temperatures_K": "300,600,900"
  }
}
```

---

## Dependencies

| Package            | Role                                              |
|--------------------|---------------------------------------------------|
| `numpy`            | Numerical operations, array math                  |
| `ase`              | Atomic Simulation Environment — structure building, I/O |
| `phonopy`          | Finite-displacement phonon calculations           |
| `flytekit`         | Workflow orchestration (tasks, dynamic, map_task) |
| `Quantum ESPRESSO` | External DFT engine (`pw.x`)                     |
| Docker image       | `guptag13/slab:v5` — contains all the above       |

---

## Physical Constants Used

| Symbol         | Value                   | Meaning                          |
|----------------|-------------------------|----------------------------------|
| `_RY_TO_EV`    | 13.6057 eV/Ry           | Rydberg → eV conversion          |
| `_EV_TO_J`     | 1.6022×10⁻¹⁹ J/eV      | eV → Joules                      |
| `_KB_EV`       | 8.6173×10⁻⁵ eV/K        | Boltzmann constant (eV)          |
| `_KB_J`        | 1.3806×10⁻²³ J/K        | Boltzmann constant (J)           |
| `_H_J`         | 6.6261×10⁻³⁴ J·s        | Planck constant                  |
| `_AMU_TO_KG`   | 1.6605×10⁻²⁷ kg/amu     | Atomic mass unit → kg            |
| `_P0_PA`       | 101325 Pa               | Standard pressure (1 atm)        |
| `_CM1_TO_EV`   | 1.2398×10⁻⁴ eV/cm⁻¹    | Wavenumber → eV                  |
| `_C_CM_S`      | 2.9979×10¹⁰ cm/s        | Speed of light                   |

---

## Notes & Limitations

- **Supported metals**: Cu, Ni, Pd, Ag, Au, Al (FCC only; lattice parameters hardcoded).
- **Supported gases**: H₂, O₂, N₂, CO (symmetry numbers and spin multiplicities hardcoded). Other molecules fall back to `σ=1, 2S+1=1`.
- **Coverage range**: Always generates 0.25 ML steps at FCC hollow sites. Bridge/top/HCP sites are not currently explored.
- **Pressure axis**: The `logP` fields in `PhaseDiagramParams` are reserved but not yet wired into the phase diagram assembly task.
- **Plot output**: `plot_png` is always `None` in the current implementation; plotting is not yet implemented.
- **Execution environment**: Requires a running Flyte cluster or `pyflyte` local sandbox. Cannot be run standalone without Flyte.
