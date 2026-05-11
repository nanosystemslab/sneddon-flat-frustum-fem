"""
Cone-side contact analysis: Does the frustum cone contact the material surface?

Compares the Sneddon free-surface displacement (flat punch, no cone) against
the frustum cone profile. Where the cone is below the free surface, contact
should occur (the cone would push the material down further).

Also shows parametric study: how the contact zone grows with indentation depth.
"""
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

# Optional: personal figure-styling module. Falls back to matplotlib
# defaults if `figure_style` is not on PYTHONPATH (e.g., on a fresh clone).
try:
    sys.path.insert(0, '/Users/ethan/Desktop')
    from figure_style.style import apply_style, save_figure  # noqa: E402
except ImportError:
    def apply_style(use_tex=False):
        pass

    def save_figure(fig, filename, dpi=300):
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

apply_style(use_tex=True)
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{bm}"
    r"\renewcommand{\seriesdefault}{b}"
    r"\boldmath"
)

output_dir = "results_comparison"
os.makedirs(output_dir, exist_ok=True)

# Load frustum simulation parameters
with open("results_frustum/simulation_parameters.json") as f:
    fr_params = json.load(f)
with open("results_flat_punch/simulation_parameters.json") as f:
    fp_params = json.load(f)

# Geometry (at simulation scale)
b_sim = fr_params["geometry"]["R_punch_mm"]
half_angle_from_axis = fr_params["geometry"]["half_angle_deg"]  # 59.7 deg
alpha_surface = 90.0 - half_angle_from_axis          # 30.3 deg from surface
tan_alpha = np.tan(np.radians(alpha_surface))         # ~0.584

# Scale factor: simulation ran at convenient scale, rescale to actual punch size
R_punch_actual = 0.010728  # mm (flat-tip RADIUS; 21.456 um is the face diameter)
alpha_scale = R_punch_actual / b_sim
b = R_punch_actual  # scaled flat tip radius

# Material
E = fp_params["materials"]["steel"]["E_MPa"]
nu = fp_params["materials"]["steel"]["nu"]
mu = E / (2.0 * (1.0 + nu))

# Current FEM indentation depth (scaled)
D_fem = fr_params["fem_results"]["indentation_depth_interface_median_mm"] * alpha_scale
F_fem = fr_params["loading"]["F_applied_N"] * alpha_scale**2

print(f"Scale factor: alpha = {alpha_scale:.6f} (sim R={b_sim} mm -> actual R={R_punch_actual} mm)")
print(f"\nFrustum geometry (scaled to actual):")
print(f"  Flat tip radius b = {b*1e3:.3f} um ({b} mm)")
print(f"  Half-angle from axis = {half_angle_from_axis} deg")
print(f"  Alpha from surface = {alpha_surface:.1f} deg")
print(f"  tan(alpha) = {tan_alpha:.4f}")
print(f"  Current D = {D_fem*1e6:.1f} nm")
print(f"  Current F = {F_fem:.4f} N")

# ============================================================================
# Sneddon flat punch: free surface displacement outside punch
# ============================================================================
def sneddon_uz(r, D, a):
    """Surface displacement outside flat punch (r >= a). Sneddon eq 6.3."""
    r = np.asarray(r, dtype=float)
    result = np.zeros_like(r)
    mask = r >= a
    result[mask] = -(2.0 * D / np.pi) * np.arcsin(a / r[mask])
    result[~mask] = -D  # under punch
    return result

def cone_profile(r, D, b_flat, tan_a):
    """Lower surface of frustum cone, displaced by depth D."""
    r = np.asarray(r, dtype=float)
    result = np.full_like(r, -D)  # flat tip for r <= b
    mask = r > b_flat
    result[mask] = -D + (r[mask] - b_flat) * tan_a
    return result

# ============================================================================
# FIGURE 1: Zoomed-in view at current load
# ============================================================================
r = np.linspace(0, 0.1, 2000)  # mm, covers ~5x punch radius at actual scale

uz_free = sneddon_uz(r, D_fem, b)
uz_cone = cone_profile(r, D_fem, b, tan_alpha)

# Find contact zone boundary (where cone crosses below free surface)
# Contact where uz_cone < uz_free (cone is deeper = more negative)
gap = uz_free - uz_cone  # positive = gap (no contact), negative = interpenetration

# Find crossover point
r_fine = np.linspace(b + 1e-7, b + 0.05, 100000)
gap_fine = sneddon_uz(r_fine, D_fem, b) - cone_profile(r_fine, D_fem, b, tan_alpha)
# gap > 0 means free surface ABOVE cone → cone below material → CONTACT
# gap < 0 means free surface BELOW cone → cone above material → no contact
# gap starts positive (contact near b), crosses zero, goes negative
crossings = np.where(np.diff(np.sign(gap_fine)))[0]
if len(crossings) > 0:
    r_contact_edge = r_fine[crossings[0]]
    print(f"\nAt current load (D = {D_fem*1e3:.1f} um):")
    print(f"  Cone contacts material from r = {b} to r = {r_contact_edge:.4f} mm")
    print(f"  Contact zone width: {(r_contact_edge - b)*1e3:.1f} um")
    print(f"  Contact extends {(r_contact_edge - b)/b*100:.1f}% beyond flat tip edge")
else:
    r_contact_edge = b
    print(f"\nNo cone contact detected at current load")

# Load FEM data for overlay (scale to actual dimensions)
fr_data = np.load("results_frustum/interface_profile.npz")
fr_r = fr_data["r_profile"] * alpha_scale
fr_uz = fr_data["u_z_profile"] * alpha_scale

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: zoomed view near punch edge
ax = axes[0]
ax.plot(r * 1e3, uz_free * 1e3, '-', linewidth=2, color='C0',
        label='Sneddon free surface (no cone)')
ax.plot(r * 1e3, uz_cone * 1e3, '-', linewidth=2, color='red',
        label='Frustum cone lower surface')

# FEM data
mask = fr_r < 0.1
ax.plot(fr_r[mask] * 1e3, fr_uz[mask] * 1e3, '.', markersize=1, alpha=0.3,
        color='C1', label='Frustum FEM')

# Shade interpenetration zone
r_zone = r[(r >= b) & (r <= r_contact_edge + 0.01)]
uz_free_zone = sneddon_uz(r_zone, D_fem, b)
uz_cone_zone = cone_profile(r_zone, D_fem, b, tan_alpha)
contact_mask = uz_cone_zone < uz_free_zone
if np.any(contact_mask):
    ax.fill_between(r_zone[contact_mask] * 1e3,
                    uz_free_zone[contact_mask] * 1e3,
                    uz_cone_zone[contact_mask] * 1e3,
                    alpha=0.3, color='red', label='Interpenetration zone')

ax.axvline(b * 1e3, color='gray', linestyle='--', alpha=0.7, label=f'b = {b*1e3:.0f} um (flat tip edge)')
if r_contact_edge > b:
    ax.axvline(r_contact_edge * 1e3, color='darkred', linestyle=':', alpha=0.7,
               label=f'Contact edge = {r_contact_edge*1e3:.0f} um')

ax.set_xlim(0, 80)  # um, ~4x punch radius
ax.set_xlabel('Radial distance r (um)')
ax.set_ylabel('$u_z$ (um)')
ax.set_title(f'Cone vs Surface — Current Load (D = {D_fem*1e3:.1f} um)')
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)

# Right: gap plot
ax = axes[1]
r_gap = np.linspace(b + 1e-7, 0.06, 5000)
gap_plot = sneddon_uz(r_gap, D_fem, b) - cone_profile(r_gap, D_fem, b, tan_alpha)
ax.plot(r_gap * 1e3, gap_plot * 1e3, '-', linewidth=2, color='C0')
ax.axhline(0, color='k', linewidth=0.5)
ax.fill_between(r_gap * 1e3, gap_plot * 1e3, 0,
                where=(gap_plot >= 0), alpha=0.3, color='red',
                label='Contact zone (cone below surface)')
ax.fill_between(r_gap * 1e3, gap_plot * 1e3, 0,
                where=(gap_plot < 0), alpha=0.2, color='green',
                label='Gap (cone above surface)')
ax.axvline(b * 1e3, color='gray', linestyle='--', alpha=0.7, label=f'b = {b*1e3:.0f} um')
ax.set_xlabel('Radial distance r (um)')
ax.set_ylabel('Gap: $u_z^{free} - u_z^{cone}$ (um)')
ax.set_title('Gap Between Material Surface and Cone')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle(f'Frustum Cone Contact Analysis — D = {D_fem*1e3:.1f} um, F = {F_fem:.0f} N',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(output_dir, "cone_contact_zoomed.pdf"), dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {output_dir}/cone_contact_zoomed.pdf")

# ============================================================================
# FIGURE 2: Contact zone vs indentation depth (parametric)
# ============================================================================
# Sneddon flat punch: F = 4 mu a D / (1 - nu)
# D = F (1-nu) / (4 mu a)

# --- Helper: compute contact zone for array of depths ---
def compute_contact_zones(D_arr, b_val, tan_a):
    r_search = np.linspace(b_val + 1e-8, b_val + 0.1, 500000)
    widths = []
    radii = []
    for D in D_arr:
        uz_f = sneddon_uz(r_search, D, b_val)
        uz_c = cone_profile(r_search, D, b_val, tan_a)
        gap_s = uz_f - uz_c  # positive = contact (cone below surface)
        sign_changes = np.where(np.diff(np.sign(gap_s)))[0]
        if len(sign_changes) > 0:
            rc = r_search[sign_changes[0]]
        else:
            rc = r_search[-1] if gap_s[0] > 0 else b_val
        radii.append(rc)
        widths.append(rc - b_val)
    return np.array(widths), np.array(radii)

# Full range (scaled to actual punch size)
D_values = np.linspace(0.0001, 0.015, 500)  # mm (~0.1 to 15 um)
F_values = 4.0 * mu * b * D_values / (1.0 - nu)  # Sneddon force
contact_widths, contact_radii = compute_contact_zones(D_values, b, tan_alpha)
contact_widths_um = contact_widths * 1e3

# Nano range (0 to 1000 nm = 0 to 0.001 mm)
D_nano = np.linspace(1e-7, 0.001, 500)  # mm (0.1 nm to 1000 nm)
F_nano = 4.0 * mu * b * D_nano / (1.0 - nu)
cw_nano, cr_nano = compute_contact_zones(D_nano, b, tan_alpha)
cw_nano_nm = cw_nano * 1e6  # mm to nm
cr_nano_nm = cr_nano * 1e6  # mm to nm

# ---- FIGURE 2a: Nano-scale contact (0-1000 nm) ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
ax.plot(D_nano * 1e6, cw_nano_nm, '-', linewidth=2, color='C0')
ax.set_xlabel('Indentation depth D (nm)')
ax.set_ylabel('Contact zone width (nm)')
ax.set_title('Cone Contact Width vs Depth (0–1000 nm)')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(D_nano * 1e6, cr_nano_nm, '-', linewidth=2, color='C0',
        label='Contact radius')
ax.axhline(b * 1e6, color='gray', linestyle='--', alpha=0.5,
           label=f'Flat tip edge b = {b*1e6:.0f} nm')
ax.set_xlabel('Indentation depth D (nm)')
ax.set_ylabel('Contact radius (nm)')
ax.set_title('Effective Contact Radius (0–1000 nm)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(F_nano, cw_nano_nm, '-', linewidth=2, color='C0')
ax.set_xlabel('Applied force F (N)')
ax.set_ylabel('Contact zone width (nm)')
ax.set_title('Cone Contact Width vs Force (nano range)')
ax.grid(True, alpha=0.3)

fig.suptitle(f'Frustum Contact at Small Depths — b = {b*1e6:.0f} nm, alpha = {alpha_surface:.1f} deg',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(output_dir, "cone_contact_nano.pdf"), dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {output_dir}/cone_contact_nano.pdf")

# ---- Zoomed displacement plot at a few nano-scale depths ----
D_nano_show = [10e-6, 50e-6, 100e-6, 500e-6, 1000e-6]  # mm (10, 50, 100, 500, 1000 nm)
colors_nano = plt.cm.plasma(np.linspace(0.15, 0.85, len(D_nano_show)))
r_plot_nano = np.linspace(b * 0.9, b * 1.5, 5000)  # mm, tight zoom around b

fig, ax = plt.subplots(figsize=(10, 7))
for i, D in enumerate(D_nano_show):
    uz_f = sneddon_uz(r_plot_nano, D, b)
    uz_c = cone_profile(r_plot_nano, D, b, tan_alpha)
    F = 4.0 * mu * b * D / (1.0 - nu)
    lbl = f'D = {D*1e6:.0f} nm (F = {F:.2f} N)'
    ax.plot(r_plot_nano * 1e6, uz_f * 1e6, '-', linewidth=2, color=colors_nano[i], label=lbl)
    ax.plot(r_plot_nano * 1e6, uz_c * 1e6, '--', linewidth=1.5, color=colors_nano[i], alpha=0.7)

ax.axvline(b * 1e6, color='gray', linestyle='--', alpha=0.5, label='Flat tip edge b')
ax.set_xlabel('Radial distance r (nm)')
ax.set_ylabel('$u_z$ (nm)')
ax.set_title('Sneddon Free Surface (solid) vs Cone Profile (dashed) — Nano-scale Depths')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "cone_contact_nano_profiles.pdf"), dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {output_dir}/cone_contact_nano_profiles.pdf")

# Print nano-scale summary
print(f"\nNano-scale contact summary:")
for D in D_nano_show:
    _, cr = compute_contact_zones(np.array([D]), b, tan_alpha)
    width = (cr[0] - b) * 1e6
    F = 4.0 * mu * b * D / (1.0 - nu)
    print(f"  D = {D*1e6:6.0f} nm  F = {F:8.3f} N  contact width = {width:.1f} nm  "
          f"contact radius = {cr[0]*1e6:.0f} nm")

# ---- FIGURE 2b: Full range parametric (original, kept for reference) ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
ax.plot(D_values * 1e3, contact_widths_um, '-', linewidth=2, color='C0')
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(D_fem * 1e3, color='red', linestyle='--', alpha=0.7,
           label=f'Current FEM (D={D_fem*1e3:.1f} um)')
ax.set_xlabel('Indentation depth D (um)')
ax.set_ylabel('Contact zone width (um)')
ax.set_title('Cone Contact Zone Width vs Depth')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(D_values * 1e3, contact_radii * 1e3, '-', linewidth=2, color='C0',
        label='Contact radius a')
ax.axhline(b * 1e3, color='gray', linestyle='--', alpha=0.5,
           label=f'Flat tip edge b = {b*1e3:.0f} um')
ax.axvline(D_fem * 1e3, color='red', linestyle='--', alpha=0.7,
           label=f'Current FEM')
ax.set_xlabel('Indentation depth D (um)')
ax.set_ylabel('Contact radius (um)')
ax.set_title('Effective Contact Radius vs Depth')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(F_values, contact_widths_um, '-', linewidth=2, color='C0')
ax.axvline(F_fem, color='red', linestyle='--', alpha=0.7,
           label=f'Current FEM (F={F_fem:.0f} N)')
ax.set_xlabel('Applied force F (N)')
ax.set_ylabel('Contact zone width (um)')
ax.set_title('Cone Contact Zone Width vs Force')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle(f'Frustum Contact Zone Growth — b = {b*1e3:.0f} um, alpha = {alpha_surface:.1f} deg',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(output_dir, "cone_contact_parametric.pdf"), dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {output_dir}/cone_contact_parametric.pdf")

# ============================================================================
# FIGURE 3: Multiple depths overlaid
# ============================================================================
D_show = [0.0005, 0.001, 0.002, D_fem, 0.005, 0.008, 0.01, 0.013]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(D_show)))

fig, ax = plt.subplots(figsize=(10, 7))
r_plot = np.linspace(0, 0.08, 3000)  # mm, ~4x punch radius

for i, D in enumerate(D_show):
    uz_f = sneddon_uz(r_plot, D, b)
    uz_c = cone_profile(r_plot, D, b, tan_alpha)

    F = 4.0 * mu * b * D / (1.0 - nu)
    lbl_f = f'D={D*1e3:.0f} um (F={F:.0f} N)' if D != D_fem else f'D={D*1e3:.1f} um (FEM, F={F:.0f} N)'
    style = '-' if D != D_fem else '-'
    lw = 1.5 if D != D_fem else 2.5

    ax.plot(r_plot * 1e3, uz_f * 1e3, style, linewidth=lw, color=colors[i],
            label=lbl_f)
    ax.plot(r_plot * 1e3, uz_c * 1e3, '--', linewidth=1, color=colors[i], alpha=0.7)

ax.axvline(b * 1e3, color='gray', linestyle='--', alpha=0.5, label='Flat tip edge')
ax.set_xlabel('Radial distance r (um)')
ax.set_ylabel('$u_z$ (um)')
ax.set_title('Sneddon Free Surface (solid) vs Cone Profile (dashed) at Multiple Depths')
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(output_dir, "cone_contact_multi_depth.pdf"), dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {output_dir}/cone_contact_multi_depth.pdf")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("CONE CONTACT ANALYSIS SUMMARY")
print("=" * 70)
print(f"\nKey finding: The cone sides contact the material at ANY non-zero load.")
print(f"Near r = b, the Sneddon free surface rises with infinite slope,")
print(f"but the displacement magnitude is still less than the cone depth.")
print(f"The cone lower surface is below the free surface in a small annular zone.")
print(f"\nAt current FEM load (D = {D_fem*1e3:.1f} um, F = {F_fem:.0f} N):")
print(f"  Contact zone: b = {b*1e3:.0f} um to r = {r_contact_edge*1e3:.0f} um")
print(f"  Width: {(r_contact_edge-b)*1e3:.1f} um")
print(f"\nImplication: The flat-tip-only FEM model slightly underestimates")
print(f"the true contact area. A proper contact formulation would be needed")
print(f"to capture the cone-side contact accurately.")
print("=" * 70)


# ============================================================================
# Paper Table 1: D_{x%} depth threshold where the FEC contact radius exceeds
# the flat-tip radius by x percent.  From eq. (Dxpct) in the paper:
#
#     D_{x%}(alpha, b) = (1 + x/100) * b * tan(alpha) * arccos(1/(1 + x/100))
#                     = k_x * b * tan(alpha)
#
# The corresponding average-pressure deviation depends only on a/b via
# eq. (dsig-x):
#
#     sbar_FEC / sbar_FP = (b/a) * [ 1/2 + beta*sqrt(1-beta^2) / (2 * arccos(beta)) ]
#
# Both quantities are independent of mu, nu, and E -- they are pure-geometry
# bounds on the FP-FEC equivalence.
# ============================================================================

def D_xpct(x_pct, b_radius, alpha_surface_rad):
    """Depth at which a/b = 1 + x_pct/100.  Paper eq. (Dxpct)."""
    f = 1.0 + x_pct / 100.0
    return f * b_radius * np.tan(alpha_surface_rad) * np.arccos(1.0 / f)


def sigma_bar_ratio(x_pct):
    """sbar_FEC / sbar_FP at a/b = 1 + x_pct/100.  Paper eq. (dsig-x)."""
    f = 1.0 + x_pct / 100.0
    beta_loc = 1.0 / f
    return beta_loc * (0.5 + beta_loc * np.sqrt(1.0 - beta_loc**2)
                       / (2.0 * np.arccos(beta_loc)))


x_thresholds = [0.1, 0.2, 0.3, 0.5]
alpha_degs = [30, 45, 60, 70, 75, 80, 85, 89]
b_micron_list = [5.0, 10.728, 15.0, 20.0, 25.0]

print("\n" + "=" * 78)
print("PAPER TABLE 1: D_{x%} CONTACT-RADIUS THRESHOLDS")
print("(depths in nm; b values in um; alpha measured from the surface)")
print("=" * 78)

for x_pct in x_thresholds:
    f = 1.0 + x_pct / 100.0
    k_x = f * np.arccos(1.0 / f)
    dsig = abs(1.0 - sigma_bar_ratio(x_pct)) * 100.0
    print(f"\n(x = {x_pct}%:  a/b = {f:.3f},  k_x = {k_x:.5f},  "
          f"|dsigma_bar| = {dsig:.2f}%)")
    header = "  alpha   " + "".join(f"  b = {bv:>6.3f}" for bv in b_micron_list)
    print(header)
    for alpha_deg in alpha_degs:
        alpha_rad = np.radians(alpha_deg)
        row = [D_xpct(x_pct, bv * 1e-3, alpha_rad) * 1e6      # mm -> nm via *1e6
               for bv in b_micron_list]
        marker = "*" if alpha_deg == 60 else " "
        print(f" {marker}{alpha_deg:>4}d   " + "".join(f"  {d:>10.0f}" for d in row))

# Re-state the bolded row from the paper for sanity:
print("\nPaper-row spot-check (alpha = 60 deg, b = 10.728 um):")
for x_pct in x_thresholds:
    d = D_xpct(x_pct, 10.728e-3, np.radians(60.0)) * 1e6
    print(f"  D_{{{x_pct}%}} = {d:7.1f} nm "
          f"(paper: 831, 1177, 1442, 1864 for x = 0.1, 0.2, 0.3, 0.5)"
          if x_pct == 0.1 else f"  D_{{{x_pct}%}} = {d:7.1f} nm")

print("=" * 78)
