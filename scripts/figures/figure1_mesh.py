"""
Figure 1 — Mesh visualization for flat punch and frustum (FEC) geometries.

Builds the same gmsh meshes used in flat_punch.py and frustum.py, then renders
each with PyVista showing element edges and nodes. Saves a per-geometry PNG
plus a side-by-side comparison.

Run:
    python figure1_mesh.py
"""

import numpy as np
import gmsh
import pyvista as pv
from mpi4py import MPI

from dolfinx.io import gmshio
from dolfinx.plot import vtk_mesh

# -------------------------------------------------------------------
# Shared geometry / mesh parameters (matches flat_punch.py & frustum.py)
# -------------------------------------------------------------------
R_substrate = 3.0
H_substrate = 3.0
R_punch     = 0.5
H_punch     = 1.0

# Frustum-only
half_angle_deg = 30.0
R_frustum_top  = R_punch + H_punch * np.tan(np.radians(half_angle_deg))

# Distance-field mesh sizing
mesh_size_min       = 0.008
mesh_size_max       = 0.5
refinement_distance = 1.5
dist_min            = 0.1

SUBSTRATE_TAG = 1
PUNCH_TAG     = 2


def build_mesh(geometry: str):
    """Return (domain, cell_tags) for 'flat' or 'frustum'."""
    gmsh.initialize()
    gmsh.model.add(f"fig1_{geometry}")

    substrate = gmsh.model.occ.addCylinder(
        0, 0, 0, 0, 0, H_substrate, R_substrate
    )
    if geometry == "flat":
        punch = gmsh.model.occ.addCylinder(
            0, 0, H_substrate, 0, 0, H_punch, R_punch
        )
    elif geometry == "frustum":
        punch = gmsh.model.occ.addCone(
            0, 0, H_substrate, 0, 0, H_punch, R_punch, R_frustum_top
        )
    else:
        raise ValueError(geometry)

    gmsh.model.occ.fragment([(3, substrate)], [(3, punch)])
    gmsh.model.occ.synchronize()

    # Tag substrate (largest volume) vs punch
    vols = gmsh.model.getEntities(dim=3)
    sizes = sorted(
        ((tag, gmsh.model.occ.getMass(3, tag)) for _, tag in vols),
        key=lambda x: x[1], reverse=True,
    )
    substrate_vol, punch_vol = sizes[0][0], sizes[1][0]
    gmsh.model.addPhysicalGroup(3, [substrate_vol], SUBSTRATE_TAG)
    gmsh.model.addPhysicalGroup(3, [punch_vol],     PUNCH_TAG)

    # Distance-field refinement at the substrate-punch interface
    punch_bnd = gmsh.model.getBoundary([(3, punch_vol)], oriented=False)
    interface_surfaces = [
        tag for dim, tag in punch_bnd
        if dim == 2 and abs(gmsh.model.occ.getCenterOfMass(dim, tag)[2] - H_substrate) < 0.1
    ]

    fd = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(fd, "SurfacesList", interface_surfaces)
    gmsh.model.mesh.field.setNumber(fd, "Sampling", 100)

    ft = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(ft, "InField",  fd)
    gmsh.model.mesh.field.setNumber(ft, "SizeMin",  mesh_size_min)
    gmsh.model.mesh.field.setNumber(ft, "SizeMax",  mesh_size_max)
    gmsh.model.mesh.field.setNumber(ft, "DistMin",  dist_min)
    gmsh.model.mesh.field.setNumber(ft, "DistMax",  refinement_distance)
    gmsh.model.mesh.field.setAsBackgroundMesh(ft)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints",         0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",      0)

    # Linear elements for visualization (P1 cells render cleaner than P2)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.model.mesh.generate(3)

    domain, cell_tags, _ = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=3
    )
    gmsh.finalize()
    return domain, cell_tags


def render_mesh(domain, cell_tags, title: str, out_png: str):
    """Render a half-clipped mesh with element edges + nodes, color by material."""
    topology, cell_types, geom = vtk_mesh(domain)
    grid = pv.UnstructuredGrid(topology, cell_types, geom)
    # Map cell tags onto the grid (cell_tags ordering matches grid cells)
    grid.cell_data["material"] = cell_tags.values.astype(np.int32)

    p = pv.Plotter(off_screen=True, window_size=(1800, 1500))
    p.set_background("white")

    # Material-colored faces with element edges
    p.add_mesh(
        grid,
        scalars="material",
        cmap=["#cfd8dc", "#ffd180"],   # substrate (steel grey), punch (warm)
        show_scalar_bar=False,
        show_edges=True,
        edge_color="black",
        line_width=0.4,
        opacity=1.0,
    )

    # Camera: oblique view of the half-section
    p.camera_position = [
        (10.0, -10.0, 8.0),   # camera location
        (0.0,   0.0,  2.0),   # focal point near interface
        (0.0,   0.0,  1.0),   # up
    ]
    p.add_text(title, position="upper_edge", font_size=14, color="black")
    p.add_axes(line_width=2, color="black")
    p.screenshot(out_png, transparent_background=False)
    p.close()
    print(f"  wrote {out_png}  ({domain.topology.index_map(3).size_global} cells, "
          f"{domain.topology.index_map(0).size_global} nodes)")
    return out_png


def side_by_side(pngs, out_png):
    """Combine two PNGs into a single figure for the paper."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, path, label in zip(axes, pngs, ["(a) Flat punch", "(b) Frustum (FEC)"]):
        ax.imshow(mpimg.imread(path))
        ax.set_title(label, fontsize=14)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


if __name__ == "__main__":
    print("Building flat-punch mesh ...")
    d_flat, t_flat = build_mesh("flat")
    flat_png = render_mesh(d_flat, t_flat,
                           "Flat cylindrical punch",
                           "figure1_mesh_flat.png")

    print("Building frustum mesh ...")
    d_fec, t_fec = build_mesh("frustum")
    fec_png = render_mesh(d_fec, t_fec,
                          "Frustum (FEC) punch",
                          "figure1_mesh_frustum.png")

    print("Composing side-by-side figure ...")
    side_by_side([flat_png, fec_png], "figure1_mesh.png")
    print("Done.")
