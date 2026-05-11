# sneddon-flat-frustum-fem

Finite-element (FEniCSx / DOLFINx) simulations and Sneddon analytical
solutions for **axisymmetric indentation of an elastic half-space** by

- a rigid **flat cylindrical punch**, and
- a **flat-tipped cone (frustum)** indenter sharing the same flat-tip radius.

The repository accompanies a chapter on linear-elastic contact mechanics
in a finite-element course project (UH Mānoa, Nanosystems Lab). It
reproduces all of the chapter's quantitative results — load–depth
response, contact stress, free-surface displacement, von Mises fields,
and the FEM-vs-Sneddon validation figures — and includes a 10× Young's
modulus sensitivity-study variant.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

> The DOI badge above is a placeholder. After the first Zenodo release,
> replace `XXXXXXX` in this README, `CITATION.cff`, and `.zenodo.json`
> with the assigned record ID.

## Repository layout

```
sneddon-flat-frustum-fem/
├── sneddon_analytical.py          # Pure-NumPy Sneddon 1965 reference solution (library)
├── flat_punch.py                  # FEM: rigid flat cylindrical punch  (writes results_flat_punch/)
├── frustum.py                     # FEM: flat-tipped conical punch     (writes results_frustum/)
├── compare_punches.py             # Overlay FEM results + Sneddon analytics → results_comparison/
├── sneddon_comparison.py          # Standalone analytical FP-vs-FEC sanity check (no FEM needed)
├── cone_contact_analysis.py       # Cone-side contact study: when does the frustum cone touch?
│
├── flat_punch_10xE.py             # Sensitivity study: same as flat_punch.py with E × 10
├── frustum_10xE.py                # Sensitivity study: same as frustum.py with E × 10
├── compare_punches_10xE.py        # Sensitivity-study comparison driver
│
├── scripts/figures/               # Post-processing helpers (regenerate chapter figures)
│   ├── figure1_mesh.py            #   PyVista render of the gmsh meshes (Fig. 1)
│   ├── plot_displacement_crosssection.py  # y=0 slice, displacement magnitude
│   ├── plot_von_mises_crosssection.py     # y=0 slice, von Mises stress
│   └── make_scalebars.py          #   Standalone colorbar PNGs for ParaView panels
│
├── requirements.txt               # Pure-Python deps (numpy / scipy / matplotlib / gmsh)
├── CITATION.cff                   # Citation metadata (consumed by GitHub + Zenodo)
├── .zenodo.json                   # Zenodo deposition metadata
├── LICENSE                        # MIT
└── .gitignore                     # Excludes results_* and FEM output artifacts
```

## Architecture

```
                       sneddon_analytical.py
                       (NumPy Sneddon 1965)
                          ▲          ▲
                          │          │
        ┌─────────────────┘          └────────────────┐
        │                                             │
  sneddon_comparison.py                       compare_punches.py
  cone_contact_analysis.py                    compare_punches_10xE.py
        │                                             ▲
        │                                             │
        │                              ┌──────────────┴──────────────┐
        │                              │                             │
        │                       flat_punch.py                  frustum.py
        │                       flat_punch_10xE.py             frustum_10xE.py
        │                       (DOLFINx + gmsh)               (DOLFINx + gmsh)
        │                              │                             │
        │                              ▼                             ▼
        │                       results_flat_punch/          results_frustum/
        │                       results_flat_punch_10xE/     results_frustum_10xE/
        │                              │                             │
        │                              ▼                             ▼
        │                  ┌──────────────────────────────────────────────┐
        │                  │  simulation_parameters.json                  │
        │                  │  interface_profile.npz                       │
        │                  │  *.xdmf / *.h5 / *.vtu  (fields)             │
        │                  └──────────────────────────────────────────────┘
        │                                       │
        ▼                                       ▼
   Console + PNG figures             scripts/figures/*.py
                                     (PyVista slices, colorbars)
                                            │
                                            ▼
                                 results_comparison/*.png/.pdf
```

**Data-flow contract.** The FEM drivers (`flat_punch.py`, `frustum.py`,
and their `_10xE` siblings) own the meshing, weak-form assembly,
solve, and I/O. They each write a self-describing `results_*/`
directory containing:

| Artifact | Purpose |
| --- | --- |
| `simulation_parameters.json` | All inputs (geometry, material, load, mesh sizing) — single source of truth read by the comparison scripts |
| `interface_profile.npz` | Indenter–substrate interface `(r, u_z, σ_zz)` arrays used for the FEM-vs-Sneddon overlays |
| `*.xdmf` / `*.h5` / `*.vtu` | Displacement and von Mises fields, viewable in ParaView / PyVista |

`compare_punches.py` reads only the JSON + NPZ from each results
directory (never re-runs the FEM); the analytical curves come from
`sneddon_analytical.py`. This keeps the comparison stage fast and
deterministic.

## Physics summary

Both simulations solve **2-D axisymmetric linear elasticity** in the
`r`–`z` plane on a steel substrate (E = 200 GPa, ν = 0.28), with
cylindrical-coordinate strain (`ε_θθ = u_r / r`) and r-weighted
integration. The flat punch is modelled as a uniform pressure on a
circular contact patch; the frustum as the same flat tip plus a conical
body. Both runs use the **same total contact force** so the analytic
Sneddon prediction is identical at low loads (cone disengaged), which
serves as a baseline validation. The 10× E variants repeat the entire
pipeline with a stiffer substrate (E = 2 TPa) to check that the
FEM-vs-Sneddon agreement is preserved under load scaling.

Analytical solutions implemented here:

- **Flat cylindrical punch** — Sneddon (1965), Section 6a, eqs. 6.1–6.3
  (load–depth, contact stress, surface displacement).
- **Flat-tipped cone (frustum)** — derived from the Sneddon (1965)
  general framework; the derivation is documented inline in
  `compare_punches.py`.

> Sneddon, I. N. (1965). *The relation between load and penetration in
> the axisymmetric Boussinesq problem for a punch of arbitrary
> profile.* International Journal of Engineering Science, 3, 47–57.

## Dependencies

| Package | Used by | Notes |
| --- | --- | --- |
| `dolfinx` ≥ 0.7 | `flat_punch.py`, `frustum.py`, `*_10xE.py` | DOLFINx FEM library (mesh, fem, io) |
| `ufl` | same | Variational form language |
| `basix` | same | Finite-element definitions |
| `petsc4py` | same | PETSc linear-algebra backend |
| `mpi4py` | same | MPI bindings (single-rank is fine) |
| `gmsh` ≥ 4.11 | FEM drivers + `scripts/figures/figure1_mesh.py` | Programmatic axisymmetric mesh generation |
| `pyvista` | `scripts/figures/*.py` | 3-D slicing / VTU reading for cross-section plots |
| `numpy` ≥ 1.24 | all | Array math |
| `scipy` ≥ 1.10 | analytical + comparison scripts | `ellipk`, `ellipkinc`, `integrate`, `brentq` |
| `matplotlib` ≥ 3.7 | all figure-producing scripts | TeX rendering optional |

Pure-Python deps (covers the analytical / comparison / figure scripts):

```bash
pip install -r requirements.txt
```

For the FEM drivers, the recommended path is `conda-forge`:

```bash
conda create -n fenicsx -c conda-forge fenics-dolfinx mpich pyvista gmsh python-gmsh
conda activate fenicsx
pip install -r requirements.txt
```

### Optional figure styling

Five scripts try to import a personal styling module
(`figure_style.style`) for fonts, the Paul-Tol palette, and a shared
`save_figure` helper. Each import is wrapped in `try/except`: if the
module is not on `PYTHONPATH`, the scripts fall back to Matplotlib
defaults plus a small inline Tol palette and run unchanged.

## Reproducing the figures

Run from the repository root, in order:

```bash
# 1. Baseline FEM simulations  (each writes its own results directory)
python flat_punch.py
python frustum.py

# 2. Baseline FEM + analytical comparison
python compare_punches.py

# 3. (Optional) 10× Young's-modulus sensitivity study
python flat_punch_10xE.py
python frustum_10xE.py
python compare_punches_10xE.py

# 4. (Optional) standalone analytical sanity checks
python sneddon_comparison.py
python cone_contact_analysis.py

# 5. (Optional) regenerate chapter cross-section figures
python scripts/figures/figure1_mesh.py
python scripts/figures/plot_displacement_crosssection.py
python scripts/figures/plot_von_mises_crosssection.py
python scripts/figures/make_scalebars.py
```

`compare_punches.py` (and its `_10xE` counterpart) and
`cone_contact_analysis.py` read JSON + NPZ written by the FEM drivers,
so the simulations must run first. The cross-section helpers in
`scripts/figures/` read `*.vtu` from the same `results_*/` directories.

Output directories (`results_flat_punch/`, `results_frustum/`,
`results_comparison/`, and the `_10xE` variants) are git-ignored —
re-running the scripts is sufficient to regenerate everything.

## License

MIT — see [LICENSE](LICENSE).

## Citing this code

If you use these scripts in academic work, please cite the archived
Zenodo release. A `CITATION.cff` file is included; once the repository
is archived on Zenodo, replace the DOI placeholder in `CITATION.cff`,
`.zenodo.json`, and the badge at the top of this README with the
assigned DOI.

```text
Rocheville, E. J. (2026). sneddon-flat-frustum-fem: FEniCSx and
analytical solutions for flat-punch and frustum nanoindentation
(v0.2.0) [Computer software]. Zenodo.
https://doi.org/<DOI assigned at release>
```

## Author

Ethan Jon Rocheville — University of Hawaiʻi at Mānoa, Nanosystems Lab
ORCID: [0009-0004-6667-737X](https://orcid.org/0009-0004-6667-737X)
