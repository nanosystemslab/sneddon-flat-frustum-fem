"""
Generate standalone high-quality scale bar PNGs matching ParaView colormaps.

Data ranges pulled from simulation_parameters.json files.

Displacement (frustum): 0 -- 0.687 mm
Young's Modulus (flat punch): 200,000 -- 1,200,000 MPa  (200 -- 1200 GPa)
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ---------- ParaView "Cool to Warm" (Moreland diverging) ----------
# This is very close to matplotlib's 'coolwarm'.
# If you use a different colormap in ParaView, swap this out.
# Other common options: 'viridis', 'plasma', 'RdBu_r', 'jet'
CMAP = 'coolwarm'


def make_scalebar(vmin, vmax, label, filename, cmap=CMAP,
                  orientation='vertical', figsize=None, dpi=600,
                  num_ticks=None):
    """Save a single standalone scale bar as a transparent PNG."""
    cmap_obj = mpl.colormaps[cmap]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    if orientation == 'vertical':
        figsize = figsize or (1.2, 4)
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=0.05, right=0.45, top=0.95, bottom=0.05)
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj),
                          cax=ax, orientation='vertical')
    else:
        figsize = figsize or (5, 1.0)
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=0.05, right=0.95, top=0.55, bottom=0.45)
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj),
                          cax=ax, orientation='horizontal')

    cb.set_label(label, fontsize=14, fontweight='bold')
    cb.ax.tick_params(labelsize=11)

    if num_ticks is not None:
        cb.set_ticks(np.linspace(vmin, vmax, num_ticks))

    fig.savefig(filename, dpi=dpi, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f'Saved: {filename}')


# ---- Displacement (frustum) ----
make_scalebar(
    vmin=0.0,
    vmax=0.687,
    label='Displacement (mm)',
    filename='scalebar_displacement.png',
    num_ticks=6,
)

# ---- Young's Modulus (flat punch) ----
# Shown in GPa for readability
make_scalebar(
    vmin=200,
    vmax=1200,
    label="Young's Modulus (GPa)",
    filename='scalebar_youngs_modulus.png',
    num_ticks=6,
)

# ---- Von Mises Stress (shared range for both geometries) ----
make_scalebar(
    vmin=0.0,
    vmax=772189,
    label='Von Mises Stress (MPa)',
    filename='scalebar_von_mises.png',
    num_ticks=6,
)

print('\nDone. Adjust CMAP variable at top if your ParaView colormap differs.')
