# sneddon-flat-frustum-fem

Finite-element (FEniCSx) simulations and Sneddon analytical solutions for
axisymmetric indentation of an elastic half-space by a **flat cylindrical
punch** and a **flat-tipped cone (frustum)** indenter.

The repository accompanies a chapter on linear-elastic contact mechanics
in a finite-element course project. It reproduces all of the chapter's
quantitative results — load–depth response, contact stress, free-surface
displacement, von Mises fields, and the FEM-vs-Sneddon validation figures.

## Contents

| File | Role |
| --- | --- |
| `sneddon_analytical.py` | Pure-NumPy module implementing Sneddon (1965) eqs. 6.1–6.3 for the flat cylindrical punch. Used as a library by the comparison scripts and standalone for quick checks. |
| `flat_punch.py` | DOLFINx simulation of a rigid flat cylindrical punch on a steel substrate. Writes `results_flat_punch/` (mesh, fields, JSON parameters, interface profile). |
| `frustum.py` | DOLFINx simulation of a flat-tipped conical (frustum) punch with the same flat-tip radius as `flat_punch.py`. Writes `results_frustum/`. |
| `compare_punches.py` | Loads `results_flat_punch/` and `results_frustum/`, overlays the Sneddon analytical solutions, and writes the chapter's comparison figures to `results_comparison/`. |
| `sneddon_comparison.py` | Standalone analytical FEC-vs-flat-punch comparison; no FEM results required. Useful as a sanity check that the analytical models reduce to one another in the small-load limit. |
| `cone_contact_analysis.py` | Cone-side contact study: where does the frustum cone first touch the deformed free surface? Parametric sweep over indentation depth. |

## Physics summary

Both simulations solve **2-D axisymmetric linear elasticity** in the
`r`–`z` plane on a steel substrate (E = 200 GPa, ν = 0.28), with
cylindrical-coordinate strain (`ε_θθ = u_r / r`) and r-weighted
integration. The flat punch is modelled as an applied uniform pressure
on a circular contact patch, the frustum as the same flat tip plus a
conical body. Both runs use the same total contact force so the analytic
Sneddon prediction is identical at low loads (cone disengaged), which
serves as a baseline validation.

The analytical solutions implemented here are:

- **Flat cylindrical punch** — Sneddon (1965), Section 6a, eqs. 6.1–6.3
  (load–depth, stress, surface displacement).
- **Flat-tipped cone (frustum)** — derived from the Sneddon (1965)
  general framework; see comments inside `compare_punches.py` for the
  derivation.

> Sneddon, I. N. (1965). *The relation between load and penetration in
> the axisymmetric Boussinesq problem for a punch of arbitrary
> profile.* International Journal of Engineering Science, 3, 47–57.

## Requirements

The FEM scripts (`flat_punch.py`, `frustum.py`) require a working
**FEniCSx** environment (DOLFINx ≥ 0.7) plus **Gmsh** for mesh
generation. The analytical / comparison scripts only need NumPy, SciPy,
and Matplotlib.

| Package | Where used |
| --- | --- |
| `dolfinx`, `ufl`, `basix`, `petsc4py`, `mpi4py` | `flat_punch.py`, `frustum.py` |
| `gmsh` | `flat_punch.py`, `frustum.py` |
| `numpy`, `scipy` | all scripts |
| `matplotlib` | all scripts that generate figures |

A minimal `pip` install for the non-FEM scripts:

```bash
pip install -r requirements.txt
```

For the FEM scripts, the recommended path is conda + the official
FEniCSx package:

```bash
conda create -n fenicsx -c conda-forge fenics-dolfinx mpich pyvista gmsh python-gmsh
conda activate fenicsx
pip install -r requirements.txt
```

## Reproducing the figures

Run from the repository root, in order:

```bash
# 1. FEM simulations (each writes its own results directory)
python flat_punch.py
python frustum.py

# 2. Combined FEM + analytical comparison
python compare_punches.py

# 3. (Optional) standalone analytical sanity checks
python sneddon_comparison.py
python cone_contact_analysis.py
```

`compare_punches.py` and `cone_contact_analysis.py` read JSON metadata
and `interface_profile.npz` written by `flat_punch.py`/`frustum.py`, so
the FEM scripts must be run first.

Output directories (`results_flat_punch/`, `results_frustum/`,
`results_comparison/`) are git-ignored — re-running the scripts is
sufficient to regenerate everything.

## Optional figure styling

Five of the scripts try to import a personal styling module
(`figure_style.style`) for fonts, the Paul-Tol palette, and a shared
`save_figure` helper. The import is wrapped in `try/except`: if the
module is not on `PYTHONPATH`, the scripts fall back to Matplotlib
defaults and a small inline copy of the Tol palette, and run unchanged.

## License

MIT — see `LICENSE`.

## Citing this code

If you use these scripts in academic work, please cite the archived
Zenodo release. A `CITATION.cff` file is included; once the repository
is archived on Zenodo, replace the `doi:` placeholder in
`CITATION.cff` and `.zenodo.json` with the assigned DOI.

```text
Rocheville, E. J. (2026). sneddon-flat-frustum-fem: FEniCSx and
analytical solutions for flat-punch and frustum nanoindentation
[Computer software]. Zenodo. https://doi.org/<DOI assigned at release>
```
