"""
Cross-section of displacement magnitude through center (y=0 plane)
for both FEC (frustum) and FP (flat punch) simulations.
"""

import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

pv.OFF_SCREEN = True

# ---- Actual-probe rescaling ----
# Sim uses R_punch = 0.5 mm; the real probe has b = 10.728 um flat-tip radius.
# Length scale s = b / R_sim. Displacements scale as s (linear elasticity),
# stresses are invariant. Convert directly to micrometers: mm_sim -> um_actual.
S = (0.010728 / 0.5) * 1000.0   # = 21.456;  sim mm  ->  actual um

# ---- Use TeX fonts ----
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'text.latex.preamble': r'\usepackage{bm}\renewcommand{\mathdefault}[1]{\textbf{#1}}',
})

# ---- Load and slice both meshes ----
def slice_and_extract(vtu_path):
    """Read VTU, interpolate cell->point data, slice at y=0, return triangulation+disp magnitude.

    Both spatial coords and displacement magnitudes are returned in actual-probe
    micrometers (sim mm multiplied by S = 21.456).
    """
    mesh = pv.read(vtu_path)
    mesh = mesh.cell_data_to_point_data()
    sliced = mesh.slice(normal='y', origin=(0, 0, 0))
    sliced = sliced.triangulate()
    pts = sliced.points
    disp = sliced.point_data['Displacement']
    mag = np.linalg.norm(disp, axis=1) * S   # mm_sim -> um_actual
    x = pts[:, 0] * S
    z = pts[:, 2] * S

    # Use actual mesh connectivity (not Delaunay) so empty space stays empty
    triangles = sliced.faces.reshape(-1, 4)[:, 1:]
    triang = tri.Triangulation(x, z, triangles=triangles)

    return triang, mag

print('Slicing frustum...')
tri1, d1 = slice_and_extract('results_frustum/displacement_p0_000000.vtu')
print('Slicing flat punch...')
tri2, d2 = slice_and_extract('results_flat_punch/displacement_p0_000000.vtu')

# ---- Shared color range ----
vmin = 0
vmax = max(d1.max(), d2.max())

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True,
                         gridspec_kw={'wspace': 0})

for ax, triang, d, title in [
    (axes[0], tri1, d1, r'\textbf{FEC (Frustum)}'),
    (axes[1], tri2, d2, r'\textbf{FP (Flat Punch)}'),
]:
    tc = ax.tripcolor(triang, d, shading='gouraud', cmap='coolwarm',
                      vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(r'\textbf{x (\textmu{}m)}', fontsize=16)
    ax.set_aspect('equal')
    ax.set_facecolor('none')
    ax.set_xlim(-3 * S, 3 * S)
    ax.margins(x=0)
    ax.tick_params(labelsize=14)

# Fix doubled "3" on FEC x-axis (ticks at 0, +/- 20, +/- 40, +/- 60 um)
axes[0].set_xticks([-60, -40, -20, 0, 20, 40])

# Hide inner spines so substrate looks continuous
axes[0].spines['right'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[0].tick_params(right=False)
axes[1].tick_params(left=False, labelleft=False)
axes[0].set_ylabel(r'\textbf{z (\textmu{}m)}', fontsize=16)

# Shared colorbar
cbar = fig.colorbar(tc, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label(r'\textbf{Displacement (\textmu{}m)}', fontsize=16)
cbar.ax.tick_params(labelsize=14)
cbar_ticks = np.linspace(0, vmax, 6)
cbar_ticks[-1] = vmax
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([r'$\mathbf{%.2f}$' % t for t in cbar_ticks])

fig.patch.set_alpha(0.0)
fig.savefig('displacement_crosssection.png', dpi=600, bbox_inches='tight', transparent=True)
fig.savefig('displacement_crosssection.pdf', bbox_inches='tight')
plt.close(fig)
print('Saved: displacement_crosssection.png and .pdf')

# ---- fig1 copy: 1.6x larger fonts and simplified ticks ----
F = 1.6
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True,
                         gridspec_kw={'wspace': 0})

for ax, triang, d, title in [
    (axes[0], tri1, d1, r'\textbf{FEC (Frustum)}'),
    (axes[1], tri2, d2, r'\textbf{FP (Flat Punch)}'),
]:
    tc = ax.tripcolor(triang, d, shading='gouraud', cmap='coolwarm',
                      vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=18 * F)
    ax.set_xlabel(r'\textbf{x (\textmu{}m)}', fontsize=16 * F)
    ax.set_aspect('equal')
    ax.set_facecolor('none')
    ax.set_xlim(-3 * S, 3 * S)
    ax.margins(x=0)
    ax.tick_params(labelsize=14 * F)

# Simplified ticks: every 30 um on x, every 40 um on y (shared)
axes[0].set_xticks([-60, -30, 0, 30])   # skip +60 to avoid clash with FP's -60
axes[1].set_xticks([-30, 0, 30, 60])    # skip -60 to avoid clash with FEC's +60
axes[0].set_yticks([0, 40, 80])
axes[0].spines['right'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[0].tick_params(right=False)
axes[1].tick_params(left=False, labelleft=False)
axes[0].set_ylabel(r'\textbf{z (\textmu{}m)}', fontsize=16 * F)

cbar = fig.colorbar(tc, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label(r'\textbf{Displacement (\textmu{}m)}', fontsize=16 * F)
cbar.ax.tick_params(labelsize=14 * F)
cbar_ticks = [0, vmax / 2.0, vmax]
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([r'$\mathbf{%.2f}$' % t for t in cbar_ticks])

fig.patch.set_alpha(0.0)
fig.savefig('displacement_crosssection_fig1.png', dpi=600, bbox_inches='tight', transparent=True)
fig.savefig('displacement_crosssection_fig1.pdf', bbox_inches='tight')
plt.close(fig)
print('Saved: displacement_crosssection_fig1.png and .pdf')
