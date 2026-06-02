# Phase Diagram Workflow — Documentation

**Ab-initio surface phase diagram: QE + Phonopy + Flyte**  
Generalized for arbitrary gas reservoirs, surface systems, and bulk references.

---

## Table of Contents

1. [What the Code Does](#1-what-the-code-does)  
2. [Physics Background](#2-physics-background)  
3. [Workflow Stages](#3-workflow-stages)  
4. [Data Classes Reference](#4-data-classes-reference)  
   - 4.1 ConfigEntry  
   - 4.2 GasData  
   - 4.3 QEParams  
   - 4.4 PhononParams  
   - 4.5 PhaseDiagramParams  
5. [configs.json Format](#5-configsjson-format)  
6. [Outputs](#6-outputs)  
7. [CLI Reference](#7-cli-reference)  
8. [Example Systems](#8-example-systems)  
   - 8.1 H/Pt(111) — hydrogen adsorption (JANAF, single reservoir)  
   - 8.2 O/Cu(111) — oxygen adsorption (ideal-gas O₂)  
   - 8.3 CO+O co-adsorption — two independent reservoirs  
   - 8.4 N/Fe(110) — nitrogen adsorption + bulk nitride reference  
   - 8.5 H₂O/Ru(0001) — water adsorption with two dependent reservoirs  
9. [Chemical Potential Models](#9-chemical-potential-models)  
10. [Supported Gas Species (JANAF)](#10-supported-gas-species-janaf)  
11. [Normalisation Modes](#11-normalisation-modes)  
12. [Flyte Remote Execution](#12-flyte-remote-execution)  
13. [Known Limitations](#13-known-limitations)  

---

## 1. What the Code Does

`phase_diagram_workflow.py` is a fully automated, end-to-end pipeline that takes
crystal structure files (CIF, VASP POSCAR, or QE output) for a surface system and
produces a thermodynamic surface phase diagram as a function of temperature and
gas-phase pressure.

**Key generalisations over the original code:**

| Feature | Original | Generalised |
|---|---|---|
| Gas reservoirs | H₂ only (hardcoded) | Any number; any gas species |
| Chemical potential model | JANAF (H₂) only | `janaf` / `ideal_gas` / `phonopy` per species |
| Stoichiometry tracking | Single `n_H` integer | Flexible `stoich` dict per config |
| Pressure axes | One shared P axis | Per-reservoir P ranges |
| Normalisation | Area only | Area / formula unit / none / auto |
| Bulk references | Not supported | `bulk_ref` role supported |
| Gas metadata | None | `GasData` with σ, 2S+1, linearity |

The pipeline runs on [Flyte](https://flyte.org/) for distributed execution but
also works locally via the CLI entry point.

---

## 2. Physics Background

For each surface configuration θ, the surface Gibbs free energy relative to the
clean slab is:

```
ΔG(θ, T, {P_s}) = [ E_slab(θ) − E_clean − Σ_s n_s · μ_s(T, P_s)
                     + ΔF_vib(θ, T) ]  /  A_norm
```

| Symbol | Meaning |
|---|---|
| `E_slab(θ)` | DFT total energy of slab with adsorbates (eV) |
| `E_clean` | DFT total energy of the bare reference slab (eV) |
| `n_s` | Net number of formula units of gas species `s` consumed |
| `μ_s(T, P_s)` | Chemical potential of gas reservoir `s` (eV/molecule) |
| `ΔF_vib(θ, T)` | `F_vib(slab+ads, T) − F_vib(clean, T)` from Phonopy (eV) |
| `A_norm` | Normalisation: cell area (Å²), formula units, or 1 |

The **stable phase** at each (T, P) point is the configuration with the lowest ΔG.

### Chemical potential models

Three models are available per gas reservoir (set via `gas_data.mu_model`):

#### `janaf` (default)
```
μ(T, P) = E_DFT + ZPE + Δμ°(T) + k_B T · ln(P / P°)
```
`Δμ°(T)` is read from built-in NIST-JANAF tables.  
ZPE is taken from the literature (see [Section 10](#10-supported-gas-species-janaf)).

#### `ideal_gas`
Full statistical-mechanical partition function computed from the QE-relaxed
geometry and Phonopy vibrational frequencies:

```
μ(T, P) = E_DFT + ZPE + F_vib_thermal(T)
         − k_B T [ ln(q_trans/V) + ln(q_rot) + ln(g_e) ]
         + k_B T · ln(P / P°)
```

| Partition function | Formula | Inputs needed |
|---|---|---|
| Translational `q_trans/V` | `(2πmkT/h²)^(3/2)` | Molecular mass from ASE |
| Rotational (linear) `q_rot` | `T / (σ · Θ_rot)` | Moment of inertia from geometry |
| Rotational (nonlinear) `q_rot` | `√π/σ · (T³/Θ_AΘ_BΘ_C)^(1/2)` | Three principal moments |
| Vibrational `q_vib` | `∏ exp(−ħω_i/2kT) / (1−exp(−ħω_i/kT))` | Phonopy frequencies |
| Electronic `g_e` | `2S+1` | `spin_multiplicity` in GasData |

#### `phonopy`
```
μ(T, P) = E_DFT + F_vib(T) + k_B T · ln(P / P°)
```
Uses the harmonic vibrational free energy from Phonopy, no tabulated data.

---

## 3. Workflow Stages

```
configs.json
    │
    ├─ For each slab/clean config ──────────────────────────────────┐
    │   structure_prep_task  →  relax_task  →  phonopy_displacements_task
    │                                          │
    │                                    map_task: run_displacement_task (×N, parallel)
    │                                          │
    │                                    phonopy_free_energy_task  → F_vib(T)
    │                                                                           │
    ├─ For each gas_ref (JANAF) ───────────────────────────────────┐           │
    │   structure_prep_task  →  relax_task  →  gas_reference_task  ─────────────┤
    │                                                                           │
    ├─ For each gas_ref (ideal_gas/phonopy) ─── same as slab pipeline ──────────┤
    │                                                                           │
    └──────────────────────────────────────────────────────────────────────────▼
                                                      phase_diagram_task
                                                      ΔG(T,P) → CSV/JSON/PNG/TXT
```

| Task | Description | Resources |
|---|---|---|
| `structure_prep_task` | Load CIF/VASP, detect cell area and n_fu, write pw.x input | Light |
| `relax_task` | Run `pw.x` relax or SCF, parse energy, save POSCAR | Heavy |
| `phonopy_displacements_task` | Generate displaced supercells, write SCF inputs | Light |
| `run_displacement_task` | Run single-point SCF for one displaced supercell | Heavy |
| `phonopy_free_energy_task` | Assemble FORCE_SETS, run phonopy DOS, produce F_vib(T) | Medium |
| `gas_reference_task` | Build μ(T,P) grid from JANAF/ideal_gas/phonopy model | Light |
| `phase_diagram_task` | Compute ΔG, write all outputs | Light |

---

## 4. Data Classes Reference

### 4.1 ConfigEntry

One entry in `configs.json`.

| Field | Type | Required | Description |
|---|---|---|---|
| `label` | str | ✓ | Unique identifier (used in all output filenames and labels) |
| `role` | str | ✓ | `"clean"` / `"slab"` / `"gas_ref"` / `"bulk_ref"` |
| `geometry` | str | ✓ | `"slab"` / `"bulk"` / `"gas"` |
| `structure` | str | ✓ | Path to CIF, VASP POSCAR, or QE output |
| `stoich` | dict | — | Gas consumed to form this config from the clean surface. Key = gas_ref label, value = float. Positive = consumed from gas phase. Default `{}`. |
| `gas_data` | GasData | — | Only for `role="gas_ref"`. Controls chemical potential model. |
| `n_fu` | int | — | Number of formula units in bulk cell (only for `bulk_ref`). 0 = auto-detect. |

**Role semantics:**

| Role | Enters Phase Diagram | Used As |
|---|---|---|
| `clean` | Yes (as reference, ΔG = 0 by definition) | `E_clean` in ΔG formula |
| `slab` | Yes | Surface configuration at some coverage |
| `gas_ref` | No | Chemical potential reservoir |
| `bulk_ref` | No (currently) | Energy reference for oxide/nitride stability |

### 4.2 GasData

Gas-molecule metadata (nested inside ConfigEntry for `role="gas_ref"`).

| Field | Type | Default | Description |
|---|---|---|---|
| `molecule` | str | `""` | Species name, e.g. `"H2"`, `"O2"`, `"H2O"`. Used to look up JANAF table. Falls back to `label` if empty. |
| `symmetry_number` | int | 1 | Rotational symmetry number σ. H₂=2, N₂=2, O₂=2, H₂O=2, NH₃=3, CO=1. |
| `spin_multiplicity` | int | 1 | Electronic degeneracy 2S+1. O₂=3, NO=2, all others=1. |
| `linear` | bool | True | True for linear molecules (H₂, N₂, O₂, CO, NO, CO₂). |
| `mu_model` | str | `"janaf"` | `"janaf"` / `"ideal_gas"` / `"phonopy"` |
| `stoich_factor` | float | 1.0 | Scaling factor applied to μ. Use 0.5 to get per-atom chemical potential from a diatomic. |

### 4.3 QEParams

Quantum ESPRESSO pw.x settings shared across all calculations.

| Field | Type | Default | Description |
|---|---|---|---|
| `pseudo_dir` | str | `"/pseudos"` | Directory containing `.UPF` pseudopotential files |
| `ecutwfc` | float | 60.0 | Plane-wave cutoff (Ry) |
| `ecutrho` | float | 480.0 | Density cutoff (Ry). Typically 4× ecutwfc for NC, 8× for USPP |
| `kpoints` | str | `"6 6 1 0 0 0"` | k-point mesh for slab/bulk as `"nx ny nz sx sy sz"` |
| `kpoints_gas` | str | `"1 1 1 0 0 0"` | k-point mesh for gas/molecule calculations |
| `smearing` | str | `"mv"` | Smearing type: `mv` (Marzari-Vanderbilt), `mp`, `fd`, `gs` |
| `degauss` | float | 0.02 | Smearing width (Ry) |
| `conv_thr` | float | 1e-8 | SCF convergence threshold (Ry) |
| `forc_conv_thr` | float | 1e-4 | Force convergence for relax (Ry/Bohr) |
| `mixing_beta` | float | 0.3 | Charge density mixing |
| `vdw_corr` | str | `"dft-d3"` | Van der Waals correction. `""` to disable. |
| `nstep` | int | 200 | Max ionic steps in relax |
| `mpi_procs` | int | `os.cpu_count()` | MPI processes per pw.x run |
| `pw_exe` | str | `"pw.x"` | pw.x executable |
| `fix_bottom` | bool | True | Fix bottom half of slab atoms during relaxation |

### 4.4 PhononParams

Phonopy displacement and k-mesh settings.

| Field | Type | Default | Description |
|---|---|---|---|
| `supercell_matrix` | str | `"2 2 1"` | Diagonal expansion as `"nx ny nz"`, or full 3×3 as 9 ints |
| `displacement_dist` | float | 0.03 | Atomic displacement distance (Å) |
| `mesh` | str | `"20 20 1"` | q-point sampling mesh for phonon DOS |
| `is_symmetry` | bool | True | Exploit crystal symmetry to reduce number of displacements |
| `sc_kpoints` | str | `"2 2 1 0 0 0"` | k-mesh for displaced supercell SCF runs. Usually coarser than the slab mesh. |

### 4.5 PhaseDiagramParams

Temperature/pressure grid and output options.

| Field | Type | Default | Description |
|---|---|---|---|
| `temperatures_K` | str | `"100,...,900"` | Comma-separated temperature list in Kelvin |
| `logP_min` | float | -15.0 | Global minimum log₁₀(P/Pa) |
| `logP_max` | float | 5.0 | Global maximum log₁₀(P/Pa) |
| `logP_step` | float | 0.25 | Step size for the pressure grid |
| `logP_species` | str | `""` | Per-reservoir overrides: `"O2:-20:5,H2:-15:0"` |
| `normalization` | str | `"auto"` | `"area"` / `"formula_unit"` / `"none"` / `"auto"` |
| `reference_label` | str | `""` | Label of clean config (auto-detected as the unique `role="clean"` entry) |
| `subtract_clean_vib` | bool | True | Subtract F_vib of clean slab from ΔF_vib |
| `output_prefix` | str | `"phase_diagram"` | Prefix for all output files |

---

## 5. configs.json Format

The JSON array must contain exactly **one** entry with `"role": "clean"`, at least
**one** `"role": "slab"`, and at least **one** `"role": "gas_ref"`.

Every `stoich` key in a slab config must match the `label` of a `gas_ref` entry.

**Minimal example (one species, two coverages):**

```json
[
  {
    "label": "clean",
    "role": "clean",
    "geometry": "slab",
    "structure": "clean.cif",
    "stoich": {}
  },
  {
    "label": "0.25ML",
    "role": "slab",
    "geometry": "slab",
    "structure": "0.25ML.cif",
    "stoich": {"H2": 1}
  },
  {
    "label": "H2",
    "role": "gas_ref",
    "geometry": "gas",
    "structure": "h2.cif",
    "stoich": {},
    "gas_data": {
      "molecule": "H2",
      "symmetry_number": 2,
      "spin_multiplicity": 1,
      "linear": true,
      "mu_model": "janaf",
      "stoich_factor": 0.5
    }
  }
]
```

> **`stoich_factor: 0.5` on H₂** means the stored μ represents per-H atom chemical
> potential, so `"stoich": {"H2": 1}` means one adsorbed H atom (not one H₂).
> Alternatively keep `stoich_factor: 1.0` and use `"stoich": {"H2": 0.5}`.
> Both conventions give identical results. The first is more natural when thinking
> in terms of adsorbate counts.

**Multi-reservoir example (CO + O₂):**

```json
[
  {
    "label": "clean",
    "role": "clean",
    "geometry": "slab",
    "structure": "clean.cif",
    "stoich": {}
  },
  {
    "label": "0.25ML_CO",
    "role": "slab",
    "geometry": "slab",
    "structure": "0.25ML_CO.cif",
    "stoich": {"CO": 1}
  },
  {
    "label": "0.25ML_O",
    "role": "slab",
    "geometry": "slab",
    "structure": "0.25ML_O.cif",
    "stoich": {"O2": 0.5}
  },
  {
    "label": "0.25ML_CO+O",
    "role": "slab",
    "geometry": "slab",
    "structure": "0.25ML_CO+O.cif",
    "stoich": {"CO": 1, "O2": 0.5}
  },
  {
    "label": "CO",
    "role": "gas_ref",
    "geometry": "gas",
    "structure": "co.cif",
    "stoich": {},
    "gas_data": {"molecule": "CO", "mu_model": "janaf"}
  },
  {
    "label": "O2",
    "role": "gas_ref",
    "geometry": "gas",
    "structure": "o2.cif",
    "stoich": {},
    "gas_data": {
      "molecule": "O2",
      "symmetry_number": 2,
      "spin_multiplicity": 3,
      "linear": true,
      "mu_model": "ideal_gas"
    }
  }
]
```

---

## 6. Outputs

All output files are written to the directory where the workflow runs (local) or
to Flyte's object store (remote). Each file is named `{output_prefix}_*`.

| File | Format | Description |
|---|---|---|
| `{prefix}_free_energies.csv` | CSV | ΔG for all configs at every (T, P) grid point, plus `stable_phase` column |
| `{prefix}_phase_diagram.json` | JSON | Full grid data including metadata, T array, P array per reservoir, ΔG arrays, stable-phase map |
| `{prefix}_phase_diagram.png` | PNG | 2D phase diagram plot (T vs first P axis), inset ΔG-vs-T curves |
| `{prefix}_summary.txt` | TXT | Human-readable summary: config energies, ZPEs, stable phases at key (T,P) points |

### CSV columns

| Column | Description |
|---|---|
| `T_K` | Temperature (K) |
| `log10_P_{species}_Pa` | log₁₀(P/Pa) for each reservoir |
| `G_{label}_eV_norm` | ΔG in the normalised units (eV/Å², eV/fu, or eV/cell) |
| `stable_phase` | Label of the most stable config at this (T, P) |

### JSON structure

```json
{
  "meta": {
    "cell_area_ang2": 12.345,
    "normalization": "area",
    "stoich": {"clean": {}, "0.25ML": {"H2": 1}},
    "gas_refs": ["H2"]
  },
  "T_K": [200, 300, ...],
  "logP_Pa": {"H2": [-15.0, -14.75, ...]},
  "configs": {
    "clean":  [[ΔG at T0P0, ΔG at T0P1, ...], ...],
    "0.25ML": [[...], ...]
  },
  "stable_phase": [["clean", "0.25ML", ...], ...]
}
```

---

## 7. CLI Reference

```
python phase_diagram_workflow.py --configs <file.json> [options]
```

### Required

| Argument | Description |
|---|---|
| `--configs` | Path to configs.json |

### QE parameters

| Argument | Default | Description |
|---|---|---|
| `--pseudo-dir` | `/pseudos` | Pseudopotential directory |
| `--ecutwfc` | 60.0 | Plane-wave cutoff (Ry) |
| `--ecutrho` | 480.0 | Density cutoff (Ry) |
| `--kpoints` | `"6 6 1 0 0 0"` | k-mesh for slab calculations |
| `--kpoints-gas` | `"1 1 1 0 0 0"` | k-mesh for gas molecule SCF |
| `--vdw-corr` | `dft-d3` | VdW correction (`""` to disable) |
| `--mpi-procs` | 16 | MPI processes per pw.x run |
| `--pw-exe` | `pw.x` | pw.x executable |
| `--no-fix-bottom` | — | Disable fixing bottom-half slab atoms |

### Phonopy parameters

| Argument | Default | Description |
|---|---|---|
| `--supercell` | `"2 2 1"` | Supercell expansion |
| `--displacement` | 0.03 | Displacement distance (Å) |
| `--mesh` | `"20 20 1"` | Phonon DOS q-mesh |
| `--sc-kpoints` | `"2 2 1 0 0 0"` | k-mesh for supercell SCF runs |

### Phase diagram parameters

| Argument | Default | Description |
|---|---|---|
| `--temperatures` | `"100,...,900"` | Comma-separated temperatures (K) |
| `--pmin` | -15.0 | Global log₁₀(P/Pa) minimum |
| `--pmax` | 5.0 | Global log₁₀(P/Pa) maximum |
| `--pstep` | 0.25 | log₁₀(P) step size |
| `--logp-species` | `""` | Per-species P range: `"O2:-20:5,H2:-15:0"` |
| `--normalization` | `auto` | `area` / `formula_unit` / `none` / `auto` |
| `--no-clean-vib` | — | Do not subtract F_vib(clean) from ΔG |
| `--prefix` | `phase_diagram` | Output file prefix |

---

## 8. Example Systems

### 8.1 H/Pt(111) — hydrogen adsorption (JANAF, single reservoir)

**configs_H_Pt111.json:**
```json
[
  {"label": "clean",      "role": "clean",   "geometry": "slab",
   "structure": "Pt111_clean.cif",    "stoich": {}},
  {"label": "0.11ML_fcc", "role": "slab",    "geometry": "slab",
   "structure": "Pt111_0.11ML.cif",   "stoich": {"H2": 0.5}},
  {"label": "0.25ML_fcc", "role": "slab",    "geometry": "slab",
   "structure": "Pt111_0.25ML.cif",   "stoich": {"H2": 1}},
  {"label": "0.50ML",     "role": "slab",    "geometry": "slab",
   "structure": "Pt111_0.50ML.cif",   "stoich": {"H2": 2}},
  {"label": "1ML",        "role": "slab",    "geometry": "slab",
   "structure": "Pt111_1ML.cif",      "stoich": {"H2": 4}},
  {"label": "H2",         "role": "gas_ref", "geometry": "gas",
   "structure": "h2.cif",             "stoich": {},
   "gas_data": {"molecule": "H2", "mu_model": "janaf"}}
]
```

**CLI:**
```bash
python phase_diagram_workflow.py \
    --configs configs_H_Pt111.json \
    --pseudo-dir /pseudos \
    --temperatures "200,300,400,500,600,700,800,900" \
    --pmin -15 --pmax 5 \
    --prefix H_Pt111
```

**Expected output**: phase diagram showing clean → 0.25ML → higher coverages
as T decreases or P(H₂) increases.

---

### 8.2 O/Cu(111) — oxygen adsorption (ideal-gas O₂)

**configs_O_Cu111.json:**
```json
[
  {"label": "clean",    "role": "clean",   "geometry": "slab",
   "structure": "Cu111_clean.cif", "stoich": {}},
  {"label": "0.25ML_O","role": "slab",    "geometry": "slab",
   "structure": "Cu111_0.25O.cif", "stoich": {"O2": 0.5}},
  {"label": "0.50ML_O","role": "slab",    "geometry": "slab",
   "structure": "Cu111_0.50O.cif", "stoich": {"O2": 1.0}},
  {"label": "1ML_O",   "role": "slab",    "geometry": "slab",
   "structure": "Cu111_1ML_O.cif", "stoich": {"O2": 2.0}},
  {"label": "O2",      "role": "gas_ref", "geometry": "gas",
   "structure": "o2.cif",          "stoich": {},
   "gas_data": {
     "molecule": "O2",
     "symmetry_number": 2,
     "spin_multiplicity": 3,
     "linear": true,
     "mu_model": "ideal_gas"
   }}
]
```

**CLI:**
```bash
python phase_diagram_workflow.py \
    --configs configs_O_Cu111.json \
    --pseudo-dir /pseudos \
    --temperatures "300,500,700,900,1100" \
    --pmin -20 --pmax 5 \
    --prefix O_Cu111
```

> **Note:** `ideal_gas` requires a phonopy run for O₂. This adds one set of
> displacement calculations for the O₂ molecule. For a diatomic, phonopy will
> produce only one non-trivial frequency (the O-O stretch), which feeds into
> q_vib and ZPE.

---

### 8.3 CO+O co-adsorption — two independent reservoirs

Each reservoir (CO and O₂) has its own independent pressure axis. The output
CSV will have two P columns; the JSON and PNG use the first reservoir's P axis
with the second at its midpoint for 2D visualisation.

**configs_CO_O_Pd111.json:**
```json
[
  {"label": "clean",      "role": "clean",   "geometry": "slab",
   "structure": "Pd111_clean.cif",    "stoich": {}},
  {"label": "0.25ML_CO",  "role": "slab",    "geometry": "slab",
   "structure": "Pd111_0.25CO.cif",   "stoich": {"CO": 1}},
  {"label": "0.25ML_O",   "role": "slab",    "geometry": "slab",
   "structure": "Pd111_0.25O.cif",    "stoich": {"O2": 0.5}},
  {"label": "0.25CO+0.25O","role": "slab",   "geometry": "slab",
   "structure": "Pd111_CO_O.cif",     "stoich": {"CO": 1, "O2": 0.5}},
  {"label": "CO",         "role": "gas_ref", "geometry": "gas",
   "structure": "co.cif",             "stoich": {},
   "gas_data": {"molecule": "CO",  "mu_model": "janaf"}},
  {"label": "O2",         "role": "gas_ref", "geometry": "gas",
   "structure": "o2.cif",             "stoich": {},
   "gas_data": {"molecule": "O2",  "mu_model": "janaf",
                "spin_multiplicity": 3}}
]
```

**CLI:**
```bash
python phase_diagram_workflow.py \
    --configs configs_CO_O_Pd111.json \
    --pseudo-dir /pseudos \
    --temperatures "300,400,500,600,700,800" \
    --logp-species "CO:-20:5,O2:-20:5" \
    --prefix CO_O_Pd111
```

---

### 8.4 N/Fe(110) — nitrogen adsorption + bulk nitride reference

A bulk iron nitride (`Fe₄N`) is included as a `bulk_ref`. Its energy (per
formula unit, auto-detected from the supercell) can be referenced in stoich to
model bulk phase competition. Note: `bulk_ref` entries do not enter the phase
diagram directly in this version but are stored in outputs for post-processing.

**configs_N_Fe110.json:**
```json
[
  {"label": "clean",      "role": "clean",    "geometry": "slab",
   "structure": "Fe110_clean.cif",   "stoich": {}},
  {"label": "0.25ML_N",   "role": "slab",     "geometry": "slab",
   "structure": "Fe110_0.25N.cif",   "stoich": {"N2": 0.5}},
  {"label": "0.50ML_N",   "role": "slab",     "geometry": "slab",
   "structure": "Fe110_0.50N.cif",   "stoich": {"N2": 1.0}},
  {"label": "N2",         "role": "gas_ref",  "geometry": "gas",
   "structure": "n2.cif",            "stoich": {},
   "gas_data": {"molecule": "N2", "mu_model": "janaf"}},
  {"label": "Fe4N",       "role": "bulk_ref", "geometry": "bulk",
   "structure": "fe4n.cif",          "stoich": {}, "n_fu": 1}
]
```

**CLI:**
```bash
python phase_diagram_workflow.py \
    --configs configs_N_Fe110.json \
    --pseudo-dir /pseudos \
    --temperatures "400,500,600,700,800,900,1000" \
    --pmin -20 --pmax 5 \
    --prefix N_Fe110
```

---

### 8.5 H₂O/Ru(0001) — water adsorption with two dependent reservoirs

H₂O, H₂, and O₂ are linked by the reaction H₂O ⇌ H₂ + ½O₂. By including both
H₂ and H₂O as separate gas_ref entries, you can express adsorbate stoichiometry
in terms of either (or both) reservoirs.

**configs_H2O_Ru0001.json:**
```json
[
  {"label": "clean",      "role": "clean",   "geometry": "slab",
   "structure": "Ru0001_clean.cif",  "stoich": {}},
  {"label": "H2O_ads",    "role": "slab",    "geometry": "slab",
   "structure": "Ru0001_H2O.cif",    "stoich": {"H2O": 1}},
  {"label": "OH_ads",     "role": "slab",    "geometry": "slab",
   "structure": "Ru0001_OH.cif",     "stoich": {"H2O": 1, "H2": -0.5}},
  {"label": "O_ads",      "role": "slab",    "geometry": "slab",
   "structure": "Ru0001_O.cif",      "stoich": {"H2O": 1, "H2": -1}},
  {"label": "H2O",        "role": "gas_ref", "geometry": "gas",
   "structure": "h2o.cif",           "stoich": {},
   "gas_data": {
     "molecule": "H2O",
     "symmetry_number": 2,
     "spin_multiplicity": 1,
     "linear": false,
     "mu_model": "janaf"
   }},
  {"label": "H2",         "role": "gas_ref", "geometry": "gas",
   "structure": "h2.cif",            "stoich": {},
   "gas_data": {"molecule": "H2", "mu_model": "janaf"}}
]
```

**CLI:**
```bash
python phase_diagram_workflow.py \
    --configs configs_H2O_Ru0001.json \
    --pseudo-dir /pseudos \
    --temperatures "200,300,400,500,600" \
    --logp-species "H2O:-5:5,H2:-15:0" \
    --prefix H2O_Ru0001
```

> **Negative stoich convention:** `"stoich": {"H2O": 1, "H2": -0.5}` means
> this config is formed by adsorbing one H₂O molecule **and releasing** 0.5 H₂
> back to the gas phase (i.e., dissociative adsorption leaving an OH).

---

## 9. Chemical Potential Models

### When to use each model

| System | Recommended model | Reason |
|---|---|---|
| H/metal at T < 600 K | `janaf` | NIST-JANAF highly accurate; fast |
| O/metal at any T | `janaf` or `ideal_gas` | Both reliable; `ideal_gas` avoids tabulated data |
| CO/metal | `janaf` | Excellent CO data in JANAF |
| H₂O/metal | `janaf` | Accurate experimental data available |
| Novel/rare gas | `ideal_gas` or `phonopy` | No JANAF entry available |
| High accuracy check | `ideal_gas` | Cross-validate against JANAF |
| Tight-binding/cheap DFT | `phonopy` | Consistent level of theory throughout |

### `stoich_factor` usage

The `stoich_factor` scales the entire μ array after construction.
Use it to define whether the stoich values in slab configs are
counted in molecules or atoms:

```
# Per-atom convention (stoich_factor = 0.5 for H2):
# stoich: {"H2": 2}  →  2 × (0.5 × μ_H2) = μ_H2  → adds 2 H atoms

# Per-molecule convention (stoich_factor = 1.0 for H2):
# stoich: {"H2": 1}  →  1 × μ_H2  → equivalent
```

Both give the same ΔG. Choose the convention that matches how you
count adsorbates in your stoich dicts.

---

## 10. Supported Gas Species (JANAF)

The following species have built-in JANAF tables and ZPE corrections:

| Molecule | σ | 2S+1 | Linear | ZPE (eV) | T range (K) |
|---|---|---|---|---|---|
| H₂ | 2 | 1 | Yes | 0.2567 | 0–1200 |
| O₂ | 2 | 3 | Yes | 0.0974 | 0–1200 |
| N₂ | 2 | 1 | Yes | 0.1449 | 0–1200 |
| CO | 1 | 1 | Yes | 0.1323 | 0–1200 |
| H₂O | 2 | 1 | No  | 0.5580 | 0–1200 |
| NO | 1 | 2 | Yes | 0.1180 | 0–1200 |
| NH₃ | 3 | 1 | No  | 0.9020 | 0–1000 |
| CO₂ | 2 | 1 | Yes | 0.3050 | 0–1200 |

For species not in this table, use `mu_model: "ideal_gas"` (requires phonopy run
for frequencies) or `mu_model: "phonopy"` (uses harmonic F_vib directly).

---

## 11. Normalisation Modes

| Mode | Formula | Units | Typical use |
|---|---|---|---|
| `area` | `ΔG / A_cell` | eV/Å² | Surface slabs — compare coverage stability |
| `formula_unit` | `ΔG / n_fu` | eV/f.u. | Bulk phases, oxides, nitrides |
| `none` | `ΔG` (raw) | eV/cell | Post-processing; custom normalisation |
| `auto` | `area` if slab, `formula_unit` if bulk | — | Default; safe choice |

For `auto`, the geometry of the **clean** reference config determines the mode.

---

## 12. Flyte Remote Execution

```bash
# Register the workflow
pyflyte register phase_diagram_workflow.py

# Run remotely
pyflyte run --remote phase_diagram_workflow.py \
    phase_diagram_workflow \
    --configs_file s3://my-bucket/configs.json \
    --qe_params.pseudo_dir /pseudos \
    --qe_params.ecutwfc 80 \
    --ph_params.supercell_matrix "3 3 1" \
    --pd_params.temperatures_K "300,500,700,900" \
    --pd_params.logP_min -20 \
    --pd_params.logP_max 5
```

**Resource configuration** (edit at top of file):

```python
_RES_LIGHT  = Resources(cpu="1",  mem="4Gi",  gpu="0")   # prep, analysis tasks
_RES_MEDIUM = Resources(cpu="4",  mem="8Gi",  gpu="0")   # phonopy assembly
_RES_HEAVY  = Resources(cpu="16", mem="32Gi", gpu="0")   # pw.x relax, SCF
```

All displacement SCF calculations (`run_displacement_task`) run as a `map_task`
and are fully parallelised across the Flyte cluster — one pod per displaced
supercell.

---

## 13. Known Limitations

1. **Multi-reservoir 2D plotting**: the PNG phase diagram is always a 2D slice
   (T vs first reservoir P). For two-reservoir systems, the second P axis is
   evaluated at its midpoint. Full 3D or faceted plots require post-processing
   the JSON output.

2. **`bulk_ref` in ΔG**: bulk references are computed but not yet subtracted
   from ΔG automatically. To model bulk phase competition you must manually
   add a slab config whose stoich accounts for the bulk formation energy, or
   post-process the JSON.

3. **Imaginary modes**: the code drops any phonopy frequencies below 1 cm⁻¹
   (numerical noise / rigid translation). If a slab has genuine soft modes
   (instabilities), the resulting F_vib will be unphysical — check the
   phonon DOS before interpreting results.

4. **Ideal-gas for solids**: `mu_model: "ideal_gas"` is only physically
   meaningful for gas-phase molecules. Do not set it for slab or bulk entries.

5. **JANAF range**: the built-in JANAF tables extend to 1200 K for most species
   and 1000 K for NH₃. Temperatures above these values will use the last
   tabulated point (flat extrapolation), which may introduce errors.
