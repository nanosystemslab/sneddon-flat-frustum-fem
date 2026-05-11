"""
Cross-section of Von Mises stress through center (y=0 plane)
for both FEC (frustum) and FP (flat punch) simulations.
"""

import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

pv.OFF_SCREEN = True

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
    """Read VTU, interpolate cell->point data, slice at y=0, return triangulation+stress."""
    mesh = pv.read(vtu_path)
    mesh = mesh.cell_data_to_point_data()
    sliced = mesh.slice(normal='y', origin=(0, 0, 0))
    sliced = sliced.triangulate()
    pts = sliced.points
    stress = sliced.point_data['von_Mises_Stress']
    x, z = pts[:, 0], pts[:, 2]

    # Use actual mesh connectivity (not Delaunay) so empty space stays empty
    triangles = sliced.faces.reshape(-1, 4)[:, 1:]
    triang = tri.Triangulation(x, z, triangles=triangles)

    return triang, stress

print('Slicing frustum...')
tri1, s1 = slice_and_extract('results_frustum/von_mises_stress_p0_000000.vtu')
print('Slicing flat punch...')
tri2, s2 = slice_and_extract('results_flat_punch/von_mises_stress_p0_000000.vtu')

# ---- Shared color range ----
vmin = 0
vmax = 5e5

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True,
                         gridspec_kw={'wspace': 0})

for ax, triang, s, title in [
    (axes[0], tri1, s1, r'\textbf{FEC (Frustum)}'),
    (axes[1], tri2, s2, r'\textbf{FP (Flat Punch)}'),
]:
    tc = ax.tripcolor(triang, s, shading='gouraud', cmap='coolwarm',
                      vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(r'\textbf{x (mm)}', fontsize=16)
    ax.set_aspect('equal')
    ax.set_facecolor('none')
    ax.set_xlim(-3, 3)
    ax.margins(x=0)
    ax.tick_params(labelsize=14)

# Fix doubled "3" on FEC x-axis: hide the rightmost tick
axes[0].set_xticks([-3, -2, -1, 0, 1, 2])

# Hide inner spines so substrate looks continuous
axes[0].spines['right'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[0].tick_params(right=False)
axes[1].tick_params(left=False, labelleft=False)
axes[0].set_ylabel(r'\textbf{z (mm)}', fontsize=16)

# Shared colorbar with scientific notation
cbar = fig.colorbar(tc, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label(r'\textbf{Von Mises Stress (MPa)}', fontsize=16)
cbar.ax.tick_params(labelsize=14)
# Set ticks: round values plus exact vmax at top
cbar_ticks = [0, 1e5, 2e5, 3e5, 4e5, 5e5]
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([r'$\mathbf{%.2f}$' % (t / 1e5) for t in cbar_ticks])
# Place exponent above colorbar with extra padding to avoid overlap
cbar.ax.text(0.5, 1.06, r'$\mathbf{\times 10^5}$', fontsize=14,
             ha='center', va='bottom', transform=cbar.ax.transAxes)

fig.patch.set_alpha(0.0)
fig.savefig('von_mises_crosssection.png', dpi=600, bbox_inches='tight', transparent=True)
fig.savefig('von_mises_crosssection.pdf', bbox_inches='tight')
plt.close(fig)
print('Saved: von_mises_crosssection.png and .pdf')

# ---- fig1 copy: same plot with 1.6x larger fonts and simplified ticks ----
F = 1.6
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True,
                         gridspec_kw={'wspace': 0})

for ax, triang, s, title in [
    (axes[0], tri1, s1, r'\textbf{FEC (Frustum)}'),
    (axes[1], tri2, s2, r'\textbf{FP (Flat Punch)}'),
]:
    tc = ax.tripcolor(triang, s, shading='gouraud', cmap='coolwarm',
                      vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=18 * F)
    ax.set_xlabel(r'\textbf{x (mm)}', fontsize=16 * F)
    ax.set_aspect('equal')
    ax.set_facecolor('none')
    ax.set_xlim(-3, 3)
    ax.margins(x=0)
    ax.tick_params(labelsize=14 * F)

# Simplified ticks: 0, 1.5, 3 (absolute) on x and 0, 2, 4 on y (shared)
axes[0].set_xticks([-3, -1.5, 0, 1.5])  # skip 3 to avoid clash with FP's -3
axes[1].set_xticks([-1.5, 0, 1.5, 3])   # skip -3 to avoid clash with FEC's 3
axes[0].set_yticks([0, 2, 4])
axes[0].spines['right'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[0].tick_params(right=False)
axes[1].tick_params(left=False, labelleft=False)
axes[0].set_ylabel(r'\textbf{z (mm)}', fontsize=16 * F)

cbar = fig.colorbar(tc, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label(r'\textbf{Von Mises Stress (MPa)}', fontsize=16 * F)
cbar.ax.tick_params(labelsize=14 * F)
cbar_ticks = [0, 2.5e5, 5e5]
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([r'$\mathbf{%.1f}$' % (t / 1e5) for t in cbar_ticks])
cbar.ax.text(0.5, 1.06, r'$\mathbf{\times 10^5}$', fontsize=14 * F,
             ha='center', va='bottom', transform=cbar.ax.transAxes)

fig.patch.set_alpha(0.0)
fig.savefig('von_mises_crosssection_fig1.png', dpi=600, bbox_inches='tight', transparent=True)
fig.savefig('von_mises_crosssection_fig1.pdf', bbox_inches='tight')
plt.close(fig)
print('Saved: von_mises_crosssection_fig1.png and .pdf')
