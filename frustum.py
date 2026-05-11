"""
Frustum (Flat-Ended Conical) Punch — copied directly from Rocheville_Final_indent.ipynb
Only geometry changed: addCone instead of addCylinder for punch.
Same contact radius (b_flat = R_punch), same total force.
Pressure adjusted for larger top face area.
"""

# === Cell 1: Imports and Setup ===
import numpy as np
import gmsh
from mpi4py import MPI
from datetime import datetime
import json
import os

import dolfinx
from dolfinx import fem, mesh, io, default_scalar_type
from dolfinx.mesh import create_mesh, meshtags_from_entities
from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem

import ufl
from ufl import (ds, dx, grad, inner, div, Identity, tr, sqrt, sym,
                 TrialFunction, TestFunction, dot, nabla_grad)

from petsc4py import PETSc

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

print(f"DOLFINx version: {dolfinx.__version__}")
print(f"Running on {MPI.COMM_WORLD.size} MPI process(es)")

# === Cell 2: Geometry Parameters ===
# All dimensions in mm, pressures in MPa

# Substrate geometry
R_substrate = 3.0      # mm radius
H_substrate = 3.0      # mm height

# Punch geometry — FRUSTUM
R_punch = 0.5          # mm flat tip radius (solve at convenient scale, rescale later)
H_punch = 1.0          # mm height

# Frustum-specific parameters
# Paper convention: cone half-angle alpha = 60 deg from the surface, equivalently
# 30 deg from the indenter axis, i.e. a 60 deg full cone (apex) angle.
half_angle_deg = 30.0   # semi-included angle from indenter axis (degrees) -- 60 deg full cone
half_angle_rad = np.radians(half_angle_deg)
R_frustum_top = R_punch + H_punch * np.tan(half_angle_rad)

# Applied pressure — adjusted so total force matches flat punch
# Flat punch force: F = P_flat * pi * R_punch^2
P_flat = 100000.0
F_total = P_flat * np.pi * R_punch**2
A_frustum_top = np.pi * R_frustum_top**2
P_applied = F_total / A_frustum_top   # <-- adjusted for frustum top face

# Mesh sizing - DISTANCE FIELD PARAMETERS
mesh_size_min = 0.008           # mm finest mesh at interface (stress singularity)
mesh_size_max = 0.5            # mm coarsest mesh far field
refinement_distance = 1.5      # mm distance over which mesh transitions
dist_min = 0.1                 # mm keep finest mesh within this distance

# Material tags for physical groups
SUBSTRATE_TAG = 1
PUNCH_TAG = 2

# Boundary tags
BOTTOM_TAG = 10
TOP_PUNCH_TAG = 11

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results_frustum"

print("Geometry parameters:")
print(f"  Substrate: R={R_substrate} mm, H={H_substrate} mm")
print(f"  Frustum flat tip radius: R={R_punch} mm (same as flat punch)")
print(f"  Frustum top radius: R={R_frustum_top:.4f} mm")
print(f"  Frustum half-angle (from axis): {half_angle_deg}°")
print(f"  Frustum half-angle (from surface): {90 - half_angle_deg:.1f}°")
print(f"  Punch height: H={H_punch} mm")
print(f"  Total force (same as flat punch): {F_total:.6f} N")
print(f"  Applied pressure (on top face): {P_applied:.2f} MPa")
print(f"\nMesh refinement (distance field):")
print(f"  Finest size (at interface): {mesh_size_min} mm")
print(f"  Coarsest size (far field): {mesh_size_max} mm")
print(f"  Transition distance: {refinement_distance} mm")
print(f"  Fine region: within {dist_min} mm of interface")
print(f"\nOutput directory: {output_dir}/")

# === Cell 3: Mesh Generation with Distance Field Refinement ===
# ONLY CHANGE: addCone instead of addCylinder for the punch
gmsh.initialize()
gmsh.model.add("frustum_indentation")

# Create substrate cylinder
substrate = gmsh.model.occ.addCylinder(
    0, 0, 0,           # center at origin
    0, 0, H_substrate,  # extends in +z direction
    R_substrate         # radius
)

# Create frustum (truncated cone) on top of substrate
# Base at z=H_substrate with radius R_punch (flat tip),
# top at z=H_substrate+H_punch with radius R_frustum_top
punch = gmsh.model.occ.addCone(
    0, 0, H_substrate,      # base center (on substrate top)
    0, 0, H_punch,          # axis direction (upward)
    R_punch,                # base radius = flat tip
    R_frustum_top            # top radius (wider end)
)

# Boolean fragment creates conforming interface so nodes are shared
all_volumes, volume_map = gmsh.model.occ.fragment([(3, substrate)], [(3, punch)])

gmsh.model.occ.synchronize()

# Identify which volumes are substrate vs punch based on volume
# Substrate will be larger volume
volumes = gmsh.model.getEntities(dim=3)
volume_sizes = []
for dim, tag in volumes:
    mass = gmsh.model.occ.getMass(dim, tag)
    volume_sizes.append((tag, mass))

# Sort by volume
volume_sizes.sort(key=lambda x: x[1], reverse=True)

# Largest volume is substrate, smallest is punch
substrate_vol = volume_sizes[0][0]
punch_vol = volume_sizes[1][0]

print(f"Substrate volume tag: {substrate_vol}")
print(f"Punch volume tag: {punch_vol}")

# Create physical groups to become cell tags
gmsh.model.addPhysicalGroup(3, [substrate_vol], SUBSTRATE_TAG)
gmsh.model.setPhysicalName(3, SUBSTRATE_TAG, "Steel_Substrate")

gmsh.model.addPhysicalGroup(3, [punch_vol], PUNCH_TAG)
gmsh.model.setPhysicalName(3, PUNCH_TAG, "Diamond_Frustum")

# Mark boundary surfaces
all_surfaces = gmsh.model.getEntities(dim=2)

bottom_surfaces = []
top_punch_surfaces = []

for dim, tag in all_surfaces:
    com = gmsh.model.occ.getCenterOfMass(dim, tag)
    z = com[2]
    r = np.sqrt(com[0]**2 + com[1]**2)

    # Bottom surface (z ≈ 0)
    if abs(z) < 1e-6:
        bottom_surfaces.append(tag)

    # Top of punch (z ≈ H_substrate + H_punch, r < R_frustum_top)
    if abs(z - (H_substrate + H_punch)) < 1e-3 and r < R_frustum_top + 1e-3:
        top_punch_surfaces.append(tag)

gmsh.model.addPhysicalGroup(2, bottom_surfaces, BOTTOM_TAG)
gmsh.model.setPhysicalName(2, BOTTOM_TAG, "Bottom_Fixed")

gmsh.model.addPhysicalGroup(2, top_punch_surfaces, TOP_PUNCH_TAG)
gmsh.model.setPhysicalName(2, TOP_PUNCH_TAG, "Top_Loaded")

print(f"\nBoundary surfaces:")
print(f"  Bottom surfaces: {len(bottom_surfaces)}")
print(f"  Top punch surfaces: {len(top_punch_surfaces)}")

# ===================================================================
# DISTANCE-BASED MESH REFINEMENT
# ===================================================================
# Fine mesh near interface,
# gradually get coarser

print(f"\nConfiguring distance field mesh refinement...")

# Identify interface surfaces by the nodes
punch_surfaces = gmsh.model.getBoundary([(3, punch_vol)], oriented=False)
interface_surfaces = []

for dim, tag in punch_surfaces:
    if dim == 2:
        # Get center of mass of surface
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        z = com[2]
        # Interface is at z ≈ H_substrate, where punch meets substrate
        if abs(z - H_substrate) < 0.1:
            interface_surfaces.append(tag)

print(f"  Interface surfaces identified: {interface_surfaces}")

# Create distance field from interface surfaces
field_distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(field_distance, "SurfacesList", interface_surfaces)
gmsh.model.mesh.field.setNumber(field_distance, "Sampling", 100)

# Create threshold field, assign size to distance
field_threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(field_threshold, "InField", field_distance)
gmsh.model.mesh.field.setNumber(field_threshold, "SizeMin", mesh_size_min)
gmsh.model.mesh.field.setNumber(field_threshold, "SizeMax", mesh_size_max)
gmsh.model.mesh.field.setNumber(field_threshold, "DistMin", dist_min)  # Fine mesh within dist_min
gmsh.model.mesh.field.setNumber(field_threshold, "DistMax", refinement_distance)  # Transition to coarse

# Set as background mesh field
gmsh.model.mesh.field.setAsBackgroundMesh(field_threshold)

# Turn off default mesh size from geometry
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

print(f"\nMesh field configured:")
print(f"  Min size (at interface): {mesh_size_min} mm")
print(f"  Max size (far field): {mesh_size_max} mm")
print(f"  Fine region: < {dist_min} mm from interface")
print(f"  Transition distance: {refinement_distance} mm")
# ===================================================================

# Generate mesh with second-order elements
gmsh.option.setNumber("Mesh.ElementOrder", 2)
gmsh.model.mesh.generate(3)

print("\nMesh generated successfully")

# Import into DOLFINx
domain, cell_tags, facet_tags = gmshio.model_to_mesh(
    gmsh.model,
    MPI.COMM_WORLD,
    0,
    gdim=3
)

gmsh.finalize()

print(f"\nMesh imported to DOLFINx:")
print(f"  Total cells: {domain.topology.index_map(3).size_global}")
print(f"  Total vertices: {domain.topology.index_map(0).size_global}")

# === Cell 4: Verify Mesh Tags ===
print("Cell tags verification:")
print(f"  Meshtags dimension: {cell_tags.dim}")
print(f"  Number of tagged cells: {len(cell_tags.values)}")
print(f"  Unique tags: {np.unique(cell_tags.values)}")

# Count cells in each region
num_substrate = np.sum(cell_tags.values == SUBSTRATE_TAG)
num_punch = np.sum(cell_tags.values == PUNCH_TAG)

print(f"\nCell distribution:")
print(f"  Substrate cells (tag={SUBSTRATE_TAG}): {num_substrate}")
print(f"  Punch cells (tag={PUNCH_TAG}): {num_punch}")
print(f"  Total: {num_substrate + num_punch}")

# Verify facet tags
print(f"\nFacet tags verification:")
print(f"  Number of tagged facets: {len(facet_tags.values)}")
print(f"  Unique tags: {np.unique(facet_tags.values)}")

num_bottom = np.sum(facet_tags.values == BOTTOM_TAG)
num_top = np.sum(facet_tags.values == TOP_PUNCH_TAG)

print(f"\nFacet distribution:")
print(f"  Bottom facets (tag={BOTTOM_TAG}): {num_bottom}")
print(f"  Top punch facets (tag={TOP_PUNCH_TAG}): {num_top}")

# Sanity checks
assert num_substrate > 0, "ERROR: No substrate cells found!"
assert num_punch > 0, "ERROR: No punch cells found!"
assert num_bottom > 0, "ERROR: No bottom facets found!"
assert num_top > 0, "ERROR: No top facets found!"

print("\n✓ All mesh tags verified successfully!")

# === Cell 5: Material Properties Definition ===
# Material properties, MPa units for consistency

# Steel substrate
E_steel = 200_000.0      # MPa 200 GPa
nu_steel = 0.28          # Poisson's ratio

# Diamond punch
E_diamond = 1_200_000.0  # MPa 1200 GPa
nu_diamond = 0.20        # Poisson's ratio

# Compute Lame parameters for verification
lambda_steel = E_steel * nu_steel / ((1 + nu_steel) * (1 - 2*nu_steel))
mu_steel = E_steel / (2 * (1 + nu_steel))

lambda_diamond = E_diamond * nu_diamond / ((1 + nu_diamond) * (1 - 2*nu_diamond))
mu_diamond = E_diamond / (2 * (1 + nu_diamond))

print("Material Properties:")
print("\nSteel Substrate:")
print(f"  E = {E_steel:,.0f} MPa = {E_steel/1000:.0f} GPa")
print(f"  ν = {nu_steel}")
print(f"  λ = {lambda_steel:,.0f} MPa")
print(f"  μ = {mu_steel:,.0f} MPa")

print("\nDiamond Punch:")
print(f"  E = {E_diamond:,.0f} MPa = {E_diamond/1000:.0f} GPa")
print(f"  ν = {nu_diamond}")
print(f"  λ = {lambda_diamond:,.0f} MPa")
print(f"  μ = {mu_diamond:,.0f} MPa")

# === Cell 6: Assign Material Properties ===
# Create DG-0 function space, one value per cell
DG0 = fem.functionspace(domain, ("DG", 0))

# Create functions for material properties
E = fem.Function(DG0, name="Youngs_Modulus")
nu = fem.Function(DG0, name="Poissons_Ratio")

# Get cell indices for each material using meshtags
# Use meshtags, NOT geometric locators
cells_substrate = cell_tags.indices[cell_tags.values == SUBSTRATE_TAG]
cells_punch = cell_tags.indices[cell_tags.values == PUNCH_TAG]

print(f"Extracting cells from meshtags:")
print(f"  Substrate cells: {len(cells_substrate)}")
print(f"  Punch cells: {len(cells_punch)}")

# Only assign to local cells
tdim = domain.topology.dim
num_cells_local = domain.topology.index_map(tdim).size_local

# Filter to local cells only
cells_substrate_local = cells_substrate[cells_substrate < num_cells_local]
cells_punch_local = cells_punch[cells_punch < num_cells_local]

print(f"\nLocal cells on this MPI rank:")
print(f"  Substrate: {len(cells_substrate_local)}")
print(f"  Punch: {len(cells_punch_local)}")

# Assign material properties
E.x.array[cells_substrate_local] = E_steel
E.x.array[cells_punch_local] = E_diamond

nu.x.array[cells_substrate_local] = nu_steel
nu.x.array[cells_punch_local] = nu_diamond

# Verify assignment worked
print(f"\nMaterial property assignment verification:")
print(f"  E range: [{E.x.array.min():,.0f}, {E.x.array.max():,.0f}] MPa")
print(f"  ν range: [{nu.x.array.min():.3f}, {nu.x.array.max():.3f}]")

# Check that both materials are present
unique_E = np.unique(E.x.array)
print(f"  Unique E values: {[f'{v:,.0f}' for v in unique_E]}")

assert len(unique_E) == 2, f"ERROR: Expected 2 materials, found {len(unique_E)}"
assert np.isclose(unique_E[0], E_steel) or np.isclose(unique_E[1], E_steel), "Steel E not found"
assert np.isclose(unique_E[0], E_diamond) or np.isclose(unique_E[1], E_diamond), "Diamond E not found"

print("\n✓ Material properties assigned successfully")
print("  Both steel and diamond are present in the domain.")

# === Cell 7: Function Space and Variational Formulation ===
# Create vector function space for displacement (P2 elements)
V = fem.functionspace(domain, ("Lagrange", 2, (3,)))

print(f"Function space:")
print(f"  Element: Lagrange P2 (quadratic)")
print(f"  Vector dimension: 3 (u_x, u_y, u_z)")
print(f"  Total DOFs: {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}")

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Compute spatially-varying Lame parameters from E and nu
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))

print(f"\nLame parameters computed from E and ν fields")
print(f"  λ will vary based on material")
print(f"  μ will vary based on material")

# Define strain tensor (symmetric gradient)
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor (isotropic linear elasticity)
def sigma(u):
    return lmbda * tr(epsilon(u)) * Identity(3) + 2.0 * mu * epsilon(u)

# Variational forms
a = inner(sigma(u), epsilon(v)) * dx

# Right hand side
f = fem.Constant(domain, default_scalar_type((0.0, 0.0, 0.0)))
b = dot(f, v) * dx

# === Cell 8: Boundary Conditions ===
# Locate DOFs for boundary conditions
fdim = domain.topology.dim - 1

# Bottom surface: fully fixed (u = 0)
bottom_facets = facet_tags.indices[facet_tags.values == BOTTOM_TAG]
bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)

u_fixed = np.array([0.0, 0.0, 0.0], dtype=default_scalar_type)
bc_bottom = fem.dirichletbc(u_fixed, bottom_dofs, V)

print(f"Dirichlet BC (bottom fixed):")
print(f"  Facets: {len(bottom_facets)}")
print(f"  DOFs constrained: {len(bottom_dofs)}")

# Top of punch: applied traction (Neumann BC)
top_facets = facet_tags.indices[facet_tags.values == TOP_PUNCH_TAG]

# Define traction vector (downward pressure)
t = fem.Constant(domain, default_scalar_type((0.0, 0.0, -P_applied)))

# Create measure for boundary integrals
ds_custom = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

# Add traction term to linear form
b += dot(t, v) * ds_custom(TOP_PUNCH_TAG)

print(f"\nNeumann BC (top loaded):")
print(f"  Facets: {len(top_facets)}")
print(f"  Traction: (0, 0, {-P_applied:.2f}) MPa")
print(f"  Applied to tag: {TOP_PUNCH_TAG}")

# Collect all Dirichlet BCs
bcs = [bc_bottom]

print(f"\n✓ Boundary conditions defined")
print(f"  Dirichlet BCs: {len(bcs)}")
print(f"  Neumann BCs: 1 (traction on top)")

# === Cell 9: Assemble and Solve ===
# Create linear problem using the imported LinearProblem class
problem = LinearProblem(
    a, b, bcs=bcs,
    petsc_options={
        "ksp_type": "cg",
        "ksp_rtol": 1e-8,
        "ksp_atol": 1e-10,
        "ksp_max_it": 1000,
        "pc_type": "gamg",
        "pc_gamg_type": "agg",
        "pc_gamg_agg_nsmooths": 1,
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "jacobi",
        "ksp_monitor": None,  # Print convergence info
    }
)

print("Solver configuration:")
print("  Type: CG (Conjugate Gradient)")
print("  Preconditioner: GAMG (Algebraic Multigrid)")
print("  Relative tolerance: 1e-8")
print("  Max iterations: 1000")
print("\nSolving...\n")

# Solve
uh = problem.solve()

print("\n✓ Solution computed successfully")

# Quick check of solution
u_values = uh.x.array.reshape(-1, 3)
u_mag = np.linalg.norm(u_values, axis=1)

print(f"\nDisplacement field:")
print(f"  Max |u|: {u_mag.max():.6e} mm")
print(f"  Max u_x: {np.abs(u_values[:, 0]).max():.6e} mm")
print(f"  Max u_y: {np.abs(u_values[:, 1]).max():.6e} mm")
print(f"  Max u_z: {np.abs(u_values[:, 2]).max():.6e} mm")

# Sanity check: displacement should be non-zero
assert u_mag.max() > 1e-10, "ERROR: Zero displacement! Solution failed."
print("\n✓ Displacement is non-zero (solution is physical)")

# === Cell 10: Compute Von Mises Stress ===
# Create DG-0 space for stress (discontinuous at interfaces)
W = fem.functionspace(domain, ("DG", 0))

# Compute stress tensor
sigma_expr = sigma(uh)

# Extract stress components
s11 = sigma_expr[0, 0]
s22 = sigma_expr[1, 1]
s33 = sigma_expr[2, 2]
s12 = sigma_expr[0, 1]
s23 = sigma_expr[1, 2]
s13 = sigma_expr[0, 2]

# Von Mises stress formula
# σ_vm = sqrt(0.5*[(σ11-σ22)² + (σ22-σ33)² + (σ33-σ11)²] + 3*(σ12² + σ23² + σ13²))
von_mises_expr = sqrt(
    0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2) +
    3.0 * (s12**2 + s23**2 + s13**2)
)

# Project to DG-0 space
von_mises = fem.Function(W, name="von_Mises_stress")
von_mises_expression = fem.Expression(von_mises_expr, W.element.interpolation_points())
von_mises.interpolate(von_mises_expression)

print("Von Mises stress computed")
print(f"  Min: {von_mises.x.array.min():.2f} MPa")
print(f"  Max: {von_mises.x.array.max():.2f} MPa")
print(f"  Mean: {von_mises.x.array.mean():.2f} MPa")

# Check stress in each material
stress_substrate = von_mises.x.array[cells_substrate_local]
stress_punch = von_mises.x.array[cells_punch_local]

print(f"\nStress by material:")
print(f"  Steel substrate:")
print(f"    Max: {stress_substrate.max():.2f} MPa")
print(f"    Mean: {stress_substrate.mean():.2f} MPa")
print(f"  Diamond punch:")
print(f"    Max: {stress_punch.max():.2f} MPa")
print(f"    Mean: {stress_punch.mean():.2f} MPa")

# Sanity check: stress should be non-zero
assert von_mises.x.array.max() > 1.0, "ERROR: Near-zero stress! Check material properties."
print("\n✓ Stress field is physical")

# === Cell 11: Force Balance Validation ===
# Applied force (on top of punch)
F_applied = F_total  # same total force as flat punch

print(f"Applied load:")
print(f"  Pressure: {P_applied:.2f} MPa (adjusted for frustum top face)")
print(f"  Top face area: {A_frustum_top:.6f} mm²")
print(f"  Force: {F_applied:.6f} N")

# Compute reaction force at bottom
n = ufl.FacetNormal(domain)
T = dot(sigma(uh), n)

# Integrate traction over bottom surface (tag = BOTTOM_TAG)
F_reaction = fem.assemble_scalar(fem.form(T[2] * ds_custom(BOTTOM_TAG)))

print(f"\nReaction force (at bottom):")
print(f"  F_z: {F_reaction:.6f} N")

# Force balance
error = abs(F_applied - abs(F_reaction)) / F_applied * 100

print(f"\nForce balance:")
print(f"  Applied: {F_applied:.6f} N")
print(f"  Reaction: {abs(F_reaction):.6f} N")
print(f"  Error: {error:.2f}%")

if error < 5.0:
    print("\n✓ Force balance validated (error < 5%)")
else:
    print(f"\n⚠ Force balance error is high ({error:.2f}%)")
    print("  This may indicate mesh refinement needed or BC issues")

# === Cell 12: Export Results for ParaView Visualization ===
# Create unique timestamped output directory
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}/")
print("="*70)

# Export displacement field (vector field on vertices)
print("\n1. Exporting displacement field...")
displacement_file = os.path.join(output_dir, "displacement.pvd")
with io.VTKFile(domain.comm, displacement_file, "w") as vtk:
    uh.name = "Displacement"
    vtk.write_function(uh)
print(f"   ✓ Saved: {displacement_file}")
print(f"     Type: Vector field (3D)")
print(f"     Data: Point data (on vertices)")

# Export von Mises stress (scalar field on cells)
print("\n2. Exporting von Mises stress...")
stress_file = os.path.join(output_dir, "von_mises_stress.pvd")
with io.VTKFile(domain.comm, stress_file, "w") as vtk:
    von_mises.name = "von_Mises_Stress"
    vtk.write_function(von_mises)
print(f"   ✓ Saved: {stress_file}")
print(f"     Type: Scalar field (MPa)")
print(f"     Data: Cell data (piecewise constant)")

# Export material IDs (for filtering by material in ParaView)
print("\n3. Exporting material IDs...")
material_field = fem.Function(DG0, name="Material_ID")
material_field.x.array[:] = cell_tags.values
material_file = os.path.join(output_dir, "materials.pvd")
with io.VTKFile(domain.comm, material_file, "w") as vtk:
    vtk.write_function(material_field)
print(f"   ✓ Saved: {material_file}")
print(f"     Type: Scalar field (integer tags)")
print(f"     Values: 1=Steel, 2=Diamond")
print(f"     Data: Cell data")

# Export Young's modulus field (to verify material assignment)
print("\n4. Exporting Young's modulus field...")
E_file = os.path.join(output_dir, "youngs_modulus.pvd")
with io.VTKFile(domain.comm, E_file, "w") as vtk:
    E.name = "Youngs_Modulus"
    vtk.write_function(E)
print(f"   ✓ Saved: {E_file}")
print(f"     Type: Scalar field (MPa)")
print(f"     Values: {E_steel/1000:.0f} GPa (steel), {E_diamond/1000:.0f} GPa (diamond)")
print(f"     Data: Cell data")

print("\n" + "="*70)
print(f"✓ All FEM files exported to: {os.path.abspath(output_dir)}/")
print("="*70)

# === Cell 13: Simulation Summary ===
print("="*70)
print("SIMULATION SUMMARY - FRUSTUM PUNCH")
print("="*70)

print("\n1. GEOMETRY")
print(f"   Substrate: R={R_substrate} mm, H={H_substrate} mm")
print(f"   Frustum: flat tip R={R_punch} mm, top R={R_frustum_top:.4f} mm, H={H_punch} mm")
print(f"   Half-angle: {half_angle_deg}° (from axis)")

print("\n2. MESH (Distance Field Refinement)")
print(f"   Total cells: {domain.topology.index_map(3).size_global}")
print(f"   Substrate cells: {num_substrate}")
print(f"   Punch cells: {num_punch}")
print(f"   Element order: P2 (quadratic)")
print(f"   Refinement strategy: Distance field from interface")
print(f"   Min element size: {mesh_size_min} mm")
print(f"   Max element size: {mesh_size_max} mm")

print("\n3. MATERIALS")
print(f"   Steel: E={E_steel/1000:.0f} GPa, ν={nu_steel}")
print(f"   Diamond: E={E_diamond/1000:.0f} GPa, ν={nu_diamond}")
print(f"   Stiffness ratio: {E_diamond/E_steel:.1f}:1")

print("\n4. LOADING")
print(f"   Applied pressure: {P_applied:.2f} MPa (on frustum top face)")
print(f"   Applied force: {F_applied:.6f} N (same as flat punch)")

print("\n5. SOLUTION")
print(f"   DOFs: {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}")
print(f"   Max displacement: {u_mag.max():.6e} mm")
print(f"   Max stress (total): {von_mises.x.array.max():.2f} MPa")
print(f"   Max stress (substrate): {stress_substrate.max():.2f} MPa")
print(f"   Max stress (punch): {stress_punch.max():.2f} MPa")

print("\n6. VALIDATION")
print(f"   Force balance error: {error:.2f}%")
if error < 5.0:
    print("   ✓ Force balance validated")
else:
    print("   ⚠ Force balance needs improvement")

print("\n7. VISUALIZATION")
print(f"   Method: ParaView (VTK export)")
print(f"   Output directory: {output_dir}/")
print(f"   Files exported: 4 (.pvd files + .vtu data)")

print("\n" + "="*70)
print("STATUS: ✓ FRUSTUM MESH SIMULATION COMPLETE")
print("="*70)

# === Cell 14: Extract FEM Data for Sneddon Comparison ===
# Extract FEM data for Sneddon comparison using MESH NODES (not point evaluation)
print("Extracting FEM data for Sneddon comparison...")
print("="*70)

# ============================================================================
# Compute indentation depth candidates
# ============================================================================
# NOTE:
# Sneddon assumes a rigid punch, the relevant depth is the
# substrate surface displacement under the punch.

u_values = uh.x.array.reshape(-1, 3)
u_z_values = u_values[:, 2]  # z-component

D_global = float(np.abs(u_z_values.min()))
print("\nIndentation depth candidates:")
print(f"  D_global = {D_global:.6e} mm (abs(min u_z) over ALL DOFs; includes punch compression)")

# ============================================================================
# Project σ_zz to P1 nodal space
# ============================================================================
print(f"\nProjecting σ_zz to P1 nodal space...")

V_scalar = fem.functionspace(domain, ("Lagrange", 1))

sigma_fem = sigma(uh)
sigma_zz_expr = sigma_fem[2, 2]

u_trial = ufl.TrialFunction(V_scalar)
v_test = ufl.TestFunction(V_scalar)

a_proj = inner(u_trial, v_test) * dx
L_proj = inner(sigma_zz_expr, v_test) * dx

proj_problem = LinearProblem(a_proj, L_proj, petsc_options={"ksp_type": "cg", "pc_type": "jacobi"})
sigma_zz_nodal = proj_problem.solve()

print("  ✓ Projected σ_zz to nodal values (P1 space)")

# ============================================================================
# Extract interface nodes from P1 space
# ============================================================================
print(f"\nExtracting interface nodes from P1 function space...")

V_scalar_coords = V_scalar.tabulate_dof_coordinates()
print(f"  Total P1 DOFs (local): {V_scalar_coords.shape[0]}")

# Tighten to get thinner slice of the interface
z_tolerance = 0.03  # mm
z_coords_p1 = V_scalar_coords[:, 2]
mask_interface_p1 = np.abs(z_coords_p1 - H_substrate) < z_tolerance

nodes_interface = V_scalar_coords[mask_interface_p1]
print(f"  Nodes near interface (z = {H_substrate} ± {z_tolerance} mm): {nodes_interface.shape[0]}")

r_nodes = np.sqrt(nodes_interface[:, 0]**2 + nodes_interface[:, 1]**2)

# ============================================================================
# Extract displacement and stress values at interface nodes
# ============================================================================
print(f"\nExtracting field values at interface nodes...")

# Project displacement from P2 vector space to P1 vector space for nodal extraction
V_p1_vector = fem.functionspace(domain, ("Lagrange", 1, (3,)))

u_trial_vec = ufl.TrialFunction(V_p1_vector)
v_test_vec = ufl.TestFunction(V_p1_vector)

a_proj_vec = inner(u_trial_vec, v_test_vec) * dx
L_proj_vec = inner(uh, v_test_vec) * dx

proj_problem_vec = LinearProblem(a_proj_vec, L_proj_vec, petsc_options={"ksp_type": "cg", "pc_type": "jacobi"})
uh_p1 = proj_problem_vec.solve()

print("  ✓ Projected displacement to P1 space")

u_p1_values = uh_p1.x.array.reshape(-1, 3)
u_interface = u_p1_values[mask_interface_p1, :]
u_z_interface = u_interface[:, 2]

sigma_zz_interface = sigma_zz_nodal.x.array[mask_interface_p1]

print(f"  Extracted displacement and stress at {len(r_nodes)} interface nodes")

# ============================================================================
# Compute interface-based indentation depth (used for Sneddon)
# ============================================================================
print("\nComputing interface-based indentation depth for Sneddon...")

# Exclude a small ring near r=a where fields can be noisy/singular
edge_buffer = float(max(2.0 * mesh_size_min, 0.0))
r_contact_max = float(R_punch - edge_buffer)
if r_contact_max <= 0.0:
    r_contact_max = float(R_punch)

mask_contact = r_nodes < r_contact_max
if np.count_nonzero(mask_contact) < 10:
    mask_contact = r_nodes < R_punch
    r_contact_max = float(R_punch)

# Optional far-field offset correction (Sneddon uses u_z → 0 as r → ∞)
mask_far = r_nodes > 0.9 * R_substrate
u_z_far = float(np.median(u_z_interface[mask_far])) if np.any(mask_far) else 0.0

u_z_interface_ref = u_z_interface - u_z_far

D_contact = -u_z_interface_ref[mask_contact]

D_interface_median = float(np.median(D_contact))
D_interface_mean = float(np.mean(D_contact))
D_interface_min = float(np.min(D_contact))
D_interface_max = float(np.max(D_contact))
D_interface_std = float(np.std(D_contact))

D_fem = D_interface_median

print(f"  Contact selection: r < {r_contact_max:.4f} mm (edge buffer {edge_buffer:.4f} mm)")
print(f"  Far-field offset u_z_far = {u_z_far:.6e} mm (median at r > 0.9*R_substrate)")
print(f"  D_interface (median) = {D_interface_median:.6e} mm  <-- used for Sneddon")
print(f"  D_interface (mean)   = {D_interface_mean:.6e} mm")
print(f"  D_interface range    = [{D_interface_min:.6e}, {D_interface_max:.6e}] mm")
print(f"  D_interface std      = {D_interface_std:.6e} mm")

# ============================================================================
# Sort by radial distance for plotting
# ============================================================================
sort_idx = np.argsort(r_nodes)
r_profile_fem = r_nodes[sort_idx]
u_z_fem = u_z_interface_ref[sort_idx]
sigma_zz_fem = sigma_zz_interface[sort_idx]

print(f"\nRadial profile statistics (interface slice):")
print(f"  Radial range: {r_profile_fem.min():.4f} to {r_profile_fem.max():.4f} mm")
print(f"  Displacement range: {u_z_fem.min():.6e} to {u_z_fem.max():.6e} mm")
print(f"  Stress range: {sigma_zz_fem.min():.2f} to {sigma_zz_fem.max():.2f} MPa")
print(f"  Punch radius: {R_punch} mm")

n_under = np.sum(r_profile_fem < R_punch)
n_outside = np.sum(r_profile_fem > R_punch)
print(f"  Points under punch (r < {R_punch}): {n_under}")
print(f"  Points outside punch (r > {R_punch}): {n_outside}")

print("\n✓ FEM data extraction complete")
print(f"  D used for Sneddon (interface median): {D_fem:.6e} mm")
print(f"  D_global (for reference): {D_global:.6e} mm")
print(f"  Using {len(r_profile_fem)} P1 nodes from interface")

# === Cell 15+16+17: Save Results ===
os.makedirs(output_dir, exist_ok=True)

# Save simulation parameters as JSON
params_dict = {
    "timestamp": timestamp,
    "punch_type": "frustum",
    "geometry": {
        "R_substrate_mm": R_substrate,
        "H_substrate_mm": H_substrate,
        "R_punch_mm": R_punch,
        "R_frustum_top_mm": R_frustum_top,
        "H_punch_mm": H_punch,
        "half_angle_deg": half_angle_deg,
    },
    "loading": {
        "P_applied_MPa": P_applied,
        "F_applied_N": float(F_applied)
    },
    "materials": {
        "steel": {
            "E_MPa": E_steel,
            "nu": nu_steel,
            "mu_MPa": float(mu_steel)
        },
        "diamond": {
            "E_MPa": E_diamond,
            "nu": nu_diamond
        }
    },
    "mesh": {
        "total_cells": int(domain.topology.index_map(3).size_global),
        "substrate_cells": int(num_substrate),
        "punch_cells": int(num_punch),
        "element_order": 2,
        "mesh_size_min_mm": mesh_size_min,
        "mesh_size_max_mm": mesh_size_max
    },
    "fem_results": {
        "max_displacement_mm": float(u_mag.max()),
        "max_von_mises_stress_MPa": float(von_mises.x.array.max()),
        "max_von_mises_substrate_MPa": float(stress_substrate.max()),
        "max_von_mises_punch_MPa": float(stress_punch.max()),
        "F_reaction_N": float(F_reaction),
        "force_balance_error_percent": float(error),
        "indentation_depth_global_mm": float(D_global),
        "indentation_depth_interface_median_mm": float(D_interface_median),
        "indentation_depth_interface_mean_mm": float(D_interface_mean),
        "indentation_depth_interface_min_mm": float(D_interface_min),
        "indentation_depth_interface_max_mm": float(D_interface_max),
        "indentation_depth_interface_std_mm": float(D_interface_std),
        "indentation_depth_farfield_offset_mm": float(u_z_far),
        "indentation_depth_contact_r_max_mm": float(r_contact_max),
        "indentation_depth_edge_buffer_mm": float(edge_buffer),
        "indentation_depth_contact_points": int(np.count_nonzero(mask_contact)),
        "interface_nodes_extracted": int(len(r_profile_fem))
    },
}

params_file = os.path.join(output_dir, "simulation_parameters.json")
with open(params_file, 'w') as f:
    json.dump(params_dict, f, indent=2)
print(f"✓ Parameters saved: {params_file}")

# Save comparison data as CSV
comparison_data = np.column_stack([r_profile_fem, u_z_fem, sigma_zz_fem])
comparison_file = os.path.join(output_dir, "fem_data_profile.csv")
np.savetxt(comparison_file, comparison_data, delimiter=',',
           header='r_mm,u_z_mm,sigma_zz_MPa', comments='')
print(f"✓ FEM profile data saved: {comparison_file}")

# Save interface profile arrays for comparison script
np.savez(
    os.path.join(output_dir, "interface_profile.npz"),
    r_profile=r_profile_fem,
    u_z_profile=u_z_fem,
    sigma_zz_profile=sigma_zz_fem,
)
print(f"✓ Interface profile saved: {output_dir}/interface_profile.npz")

print("\n" + "="*70)
print("ALL RESULTS SAVED")
print("="*70)
print(f"\nOutput directory: {os.path.abspath(output_dir)}/")
print("\n" + "="*70)
print("✓ FRUSTUM SIMULATION COMPLETE")
print("="*70)
