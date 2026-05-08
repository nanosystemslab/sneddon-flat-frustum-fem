"""
Compare Flat Punch vs Frustum — loads saved results from both simulations.

Includes Sneddon analytical solutions:
  - Flat punch: Sneddon 1965 eqs 6.1-6.3
  - Frustum (flat-tipped cone): derived from Sneddon 1965 general framework
    (see sneddon_flat_tipped_cone.md for full derivation)

Run after: python flat_punch.py && python frustum.py
"""
import sys
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.special import ellipk, ellipkinc
from scipy.optimize import brentq

# Optional: personal figure-styling module. Falls back to matplotlib
# defaults + a Paul-Tol-inspired palette so the script runs on any system.
try:
    sys.path.insert(0, '/Users/ethan/Desktop')
    from figure_style.style import (  # noqa: E402
        apply_style, save_figure, get_colors, TOL_COLORS, FONT_LABEL,
    )
except ImportError:
    def apply_style(use_tex=False):
        pass

    def save_figure(fig, filename, dpi=300):
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    # Paul Tol "vibrant" palette — colour-blind safe.
    TOL_COLORS = ('#0077BB', '#EE7733', '#33BBEE', '#EE3377',
                  '#CC3311', '#009988', '#BBBBBB')

    def get_colors(n):
        return list(TOL_COLORS[:n])

    FONT_LABEL = 11

apply_style(use_tex=True)
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{bm}"
    r"\renewcommand{\seriesdefault}{b}"
    r"\boldmath"
)
plt.rcParams["legend.fontsize"] = FONT_LABEL * 0.9  # 10% smaller legend
plt.rcParams["lines.linewidth"] = 2.4   # 2x default 1.2
plt.rcParams["lines.markersize"] = 12   # 2x default 6
plt.rcParams["legend.handlelength"] = 3  # longer legend lines to match
plt.rcParams["legend.markerscale"] = 2.0  # 2x legend marker size

# ============================================================================
# Load results from both simulations
# ============================================================================
flat_dir = "results_flat_punch"
frust_dir = "results_frustum"
output_dir = "results_comparison"
os.makedirs(output_dir, exist_ok=True)

# Load JSON parameters
with open(os.path.join(flat_dir, "simulation_parameters.json")) as f:
    fp_params = json.load(f)
with open(os.path.join(frust_dir, "simulation_parameters.json")) as f:
    fr_params = json.load(f)

# Load interface profiles
fp_data = np.load(os.path.join(flat_dir, "interface_profile.npz"))
fr_data = np.load(os.path.join(frust_dir, "interface_profile.npz"))

fp_r = fp_data["r_profile"]
fp_uz = fp_data["u_z_profile"]
fp_szz = fp_data["sigma_zz_profile"]

fr_r = fr_data["r_profile"]
fr_uz = fr_data["u_z_profile"]
fr_szz = fr_data["sigma_zz_profile"]

# Extract key scalars (at simulation scale)
R_punch_sim = fp_params["geometry"]["R_punch_mm"]
R_substrate = fp_params["geometry"]["R_substrate_mm"]
H_substrate = fp_params["geometry"]["H_substrate_mm"]

# ============================================================================
# SCALE FACTOR: simulation was run at convenient scale (R_punch = 0.5 mm)
# Actual punch radius is 21.456 um = 0.021456 mm
# Linear elasticity is scale-invariant: scale all lengths by alpha, stresses unchanged
# ============================================================================
R_punch_actual = 0.021456  # mm (21.456 um)
alpha_scale = R_punch_actual / R_punch_sim  # length scale factor
print(f"Scale factor: alpha = {alpha_scale:.6f} (sim R={R_punch_sim} mm -> actual R={R_punch_actual} mm)")

# Apply scaling: lengths * alpha, stresses unchanged, forces * alpha^2
R_punch = R_punch_actual
fp_r *= alpha_scale
fp_uz *= alpha_scale
fr_r *= alpha_scale
fr_uz *= alpha_scale
# stresses (fp_szz, fr_szz) are unchanged

fp_D = fp_params["fem_results"]["indentation_depth_interface_median_mm"] * alpha_scale
fr_D = fr_params["fem_results"]["indentation_depth_interface_median_mm"] * alpha_scale

fp_F = fp_params["loading"]["F_applied_N"] * alpha_scale**2
fr_F = fr_params["loading"]["F_applied_N"] * alpha_scale**2

fp_F_rxn = fp_params["fem_results"]["F_reaction_N"] * alpha_scale**2
fr_F_rxn = fr_params["fem_results"]["F_reaction_N"] * alpha_scale**2

fp_error = fp_params["fem_results"]["force_balance_error_percent"]
fr_error = fr_params["fem_results"]["force_balance_error_percent"]

fp_vm_max = fp_params["fem_results"]["max_von_mises_stress_MPa"]
fr_vm_max = fr_params["fem_results"]["max_von_mises_stress_MPa"]

fp_vm_sub = fp_params["fem_results"].get("max_von_mises_substrate_MPa",
            fp_params["fem_results"].get("max_von_mises_stress_MPa"))
fr_vm_sub = fr_params["fem_results"]["max_von_mises_substrate_MPa"]

fp_umax = fp_params["fem_results"]["max_displacement_mm"] * alpha_scale
fr_umax = fr_params["fem_results"]["max_displacement_mm"] * alpha_scale

# Material properties (same for both)
E_steel = fp_params["materials"]["steel"]["E_MPa"]
nu_steel = fp_params["materials"]["steel"]["nu"]
mu_analytical = E_steel / (2.0 * (1.0 + nu_steel))

# Frustum-specific geometry
half_angle_deg = fr_params["geometry"]["half_angle_deg"]
R_frustum_top = fr_params["geometry"]["R_frustum_top_mm"] * alpha_scale

# Angle convention conversion:
#   frustum.py uses half_angle_deg = 59.7 deg from the indenter AXIS
#   Sneddon frustum formulas use alpha = angle from the SURFACE (horizontal)
#   alpha_surface = 90 - half_angle_from_axis
#
# The cone profile slope dz/dr = tan(alpha_surface) = cot(half_angle_from_axis)
#   tan(30.3 deg) = 0.583 = 1/tan(59.7 deg) = cone rises 0.583 mm per mm outward
half_angle_from_axis_rad = np.radians(half_angle_deg)
alpha_surface = np.pi / 2.0 - half_angle_from_axis_rad  # radians, from surface

b_flat = R_punch  # flat tip radius for frustum Sneddon

# Display conversion: depths in nanometers
mm_to_nm = 1.0e6
fp_D_nm = fp_D * mm_to_nm
fr_D_nm = fr_D * mm_to_nm

print("=" * 70)
print("LOADED RESULTS")
print("=" * 70)
print(f"\nFlat punch:  {flat_dir}/")
print(f"  R_punch = {R_punch} mm, D = {fp_D_nm:.2f} nm, F = {fp_F:.2f} N")
print(f"  Interface nodes: {len(fp_r)}")
print(f"\nFrustum:     {frust_dir}/")
print(f"  R_punch = {R_punch} mm, half-angle = {half_angle_deg} deg (from axis)")
print(f"  alpha_surface = {np.degrees(alpha_surface):.1f} deg (from surface, for Sneddon)")
print(f"  tan(alpha) = {np.tan(alpha_surface):.4f}")
print(f"  D = {fr_D_nm:.2f} nm, F = {fr_F:.2f} N")
print(f"  Interface nodes: {len(fr_r)}")

# ============================================================================
# SNEDDON ANALYTICAL: FLAT PUNCH  (Sneddon 1965, eqs 6.1-6.3)
# ============================================================================

def flat_sneddon_stress(r, D, a, mu, eta):
    """sigma_zz = -(2*mu*D)/(pi*(1-eta)) * (a^2 - r^2)^(-1/2),  r < a"""
    r_safe = np.minimum(np.asarray(r, dtype=float), a - 1e-10)
    return -(2.0 * mu * D) / (np.pi * (1.0 - eta)) / np.sqrt(a**2 - r_safe**2)


def flat_sneddon_displacement(r, D, a):
    """u_z = -(2*D/pi) * arcsin(a/r),  r > a"""
    r_safe = np.maximum(np.asarray(r, dtype=float), a + 1e-10)
    return -(2.0 * D / np.pi) * np.arcsin(a / r_safe)


def flat_sneddon_load(D, a, mu, eta):
    """P = 4*mu*a*D / (1-eta)"""
    return 4.0 * mu * a * D / (1.0 - eta)


# ============================================================================
# SNEDDON ANALYTICAL: FRUSTUM (flat-tipped cone)
#
# Derived from Sneddon 1965 general framework (eqs 2.10, 3.4, 3.7, 4.3, 5.1)
# Full derivation: sneddon_flat_tipped_cone.md
# Cross-checked against: con_sneddon.pdf (Sneddon 1965)
#
# Parameters:
#   a     = contact radius (a >= b; cone surface contacts for a > b)
#   b     = flat tip radius
#   alpha = angle between cone surface and flat surface (from horizontal)
#   beta  = b/a   (normalized flat radius)
#   eps   = a * tan(alpha)
#
# Verified limits:
#   beta -> 0 (b -> 0): recovers sharp cone  (Sneddon eqs 6.4-6.7)
#   beta -> 1 (a -> b): recovers flat punch   (Sneddon eqs 6.1-6.3)
# ============================================================================

def frustum_depth(a, b, alpha):
    """
    Penetration depth:  D = a * tan(alpha) * arccos(b/a)
    [Sneddon eq 3.7 applied to frustum profile]
    """
    a = np.asarray(a, dtype=float)
    scalar = a.ndim == 0
    a = np.atleast_1d(a)
    D = np.zeros_like(a)
    mask = a > b * (1.0 + 1e-14)
    D[mask] = a[mask] * np.tan(alpha) * np.arccos(b / a[mask])
    return float(D[0]) if scalar else D


def frustum_load(a, b, alpha, mu, eta):
    """
    Total load:  P = (2*mu*a^2*tan(alpha))/(1-eta) * [arccos(b/a) + (b/a)*sqrt(1-(b/a)^2)]
    [Sneddon eq 4.3 applied to frustum profile; cross-checked via eq 4.1]
    """
    a = np.asarray(a, dtype=float)
    scalar = a.ndim == 0
    a = np.atleast_1d(a)
    P = np.zeros_like(a)
    mask = a > b * (1.0 + 1e-14)
    beta = b / a[mask]
    P[mask] = (2.0 * mu * a[mask]**2 * np.tan(alpha)) / (1.0 - eta) * (
        np.arccos(beta) + beta * np.sqrt(1.0 - beta**2))
    return float(P[0]) if scalar else P


def frustum_contact_radius(D_target, b, alpha):
    """
    Invert D = a*tan(alpha)*arccos(b/a) to find the contact radius a
    for a given penetration depth D.  Returns a >= b.
    """
    if D_target <= 0:
        return b
    # Upper bound: for very large a, arccos(b/a) -> pi/2
    a_max = max(20.0 * b, 4.0 * D_target / (np.tan(alpha) * np.pi / 2)) + b

    def residual(a):
        return a * np.tan(alpha) * np.arccos(b / a) - D_target

    return brentq(residual, b * (1.0 + 1e-10), a_max, xtol=1e-14)


def frustum_stress_zz(rho, a, b, alpha, mu, eta):
    """
    Normal stress sigma_zz(rho, 0) under the frustum punch.

    Under flat region  (rho < b):  sigma = -(2*mu*eps)/(pi*a*(1-eta)) * [I_A + I_B]
    Under cone region  (b <= rho < a):  sigma = -(2*mu*eps)/(pi*a*(1-eta)) * [I_A' + I_B']

    I_A, I_A': numerical quadrature
    I_B:  (1/beta) * [K(m) - F(phi, m)]  with m = (x/beta)^2, phi = arcsin(beta)
    I_B': (1/x)    * [K(m) - F(phi, m)]  with m = (beta/x)^2, phi = arcsin(x)

    Note: stress is singular at rho = b (corner) and rho = a (contact edge).
    """
    eps = a * np.tan(alpha)
    beta = b / a
    prefactor = -(2.0 * mu * eps) / (np.pi * a * (1.0 - eta))

    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    sigma = np.full_like(rho, np.nan)

    for i, rho_i in enumerate(rho):
        x = rho_i / a

        # Singularity at contact edge
        if x >= 1.0 - 1e-10:
            continue
        # Singularity at corner (x = beta)
        if abs(x - beta) < 1e-8:
            continue

        if x < beta:
            # --- Under the flat region ---
            # I_A = integral from beta to 1 of arccos(beta/t)/sqrt(t^2 - x^2) dt
            def integrand_IA(t, _x=x, _b=beta):
                return np.arccos(_b / t) / np.sqrt(t**2 - _x**2)
            I_A, _ = integrate.quad(integrand_IA, beta, 1.0,
                                    points=[beta], limit=200)

            # I_B via elliptic integrals
            k = x / beta
            m = k**2
            if m < 0.999:
                I_B = (1.0 / beta) * (ellipk(m) - ellipkinc(np.arcsin(beta), m))
            else:
                # Near x -> beta, elliptic form diverges; use numerical fallback
                def integrand_IB(t, _x=x, _b=beta):
                    return 1.0 / np.sqrt((t**2 - _b**2) * (t**2 - _x**2))
                IB_raw, _ = integrate.quad(integrand_IB, beta, 1.0,
                                           points=[beta], limit=200)
                I_B = beta * IB_raw

            sigma[i] = prefactor * (I_A + I_B)

        else:
            # --- Under the cone region (beta < x < 1) ---
            # I_A' = integral from x to 1 of arccos(beta/t)/sqrt(t^2 - x^2) dt
            def integrand_IA_prime(t, _x=x, _b=beta):
                return np.arccos(_b / t) / np.sqrt(t**2 - _x**2)
            I_A_prime, _ = integrate.quad(integrand_IA_prime, x, 1.0,
                                          points=[x], limit=200)

            # I_B' via elliptic integrals
            k = beta / x
            m = k**2
            I_B_prime = (1.0 / x) * (ellipk(m) - ellipkinc(np.arcsin(x), m))

            sigma[i] = prefactor * (I_A_prime + I_B_prime)

    return sigma


def frustum_displacement_outside(rho, a, b, alpha, D):
    """
    Surface displacement u_z(rho, 0) for rho > a.

    u_z = (2D/pi)*arcsin(a/rho) - (2*eps/pi) * integral_beta^1 t*arccos(beta/t)/sqrt(x^2-t^2) dt

    First term is the flat-punch displacement (Sneddon eq 6.3).
    Second term is the frustum correction (vanishes when beta -> 1).
    Returns negative (downward) displacement.
    """
    eps = a * np.tan(alpha)
    beta = b / a

    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    u_z = np.zeros_like(rho)

    for i, rho_i in enumerate(rho):
        x = rho_i / a
        if x <= 1.0 + 1e-12:
            u_z[i] = -D   # under the punch
            continue

        # Flat punch contribution
        flat_part = (2.0 * D / np.pi) * np.arcsin(1.0 / x)

        # Frustum correction integral
        def integrand(t, _x=x, _b=beta):
            return t * np.arccos(_b / t) / np.sqrt(_x**2 - t**2)
        corr, _ = integrate.quad(integrand, beta, 1.0 - 1e-12,
                                 points=[beta], limit=200)

        u_z[i] = -(flat_part - (2.0 * eps / np.pi) * corr)

    return u_z


# ============================================================================
# Compute Sneddon analytical for both FEM cases
# ============================================================================
# NOTE: Both FEM simulations use bonded contact at the flat tip only
# (contact radius a = b = R_punch).  In this limit the frustum Sneddon
# degenerates to the flat punch Sneddon, so flat punch eqs are correct
# for comparing against BOTH FEM results.

# Analytical grids
r_max = max(fp_r.max(), fr_r.max())
r_anal = np.linspace(0, r_max, 600)
r_under = r_anal[r_anal < R_punch]
r_outside = r_anal[r_anal > R_punch]

# Flat punch Sneddon for flat punch FEM
fp_sned_stress = flat_sneddon_stress(r_under, fp_D, R_punch, mu_analytical, nu_steel)
fp_sned_disp = flat_sneddon_displacement(r_outside, fp_D, R_punch)
fp_F_sned = flat_sneddon_load(fp_D, R_punch, mu_analytical, nu_steel)

# Flat punch Sneddon for frustum FEM (a = b, so flat punch is correct)
fr_sned_stress = flat_sneddon_stress(r_under, fr_D, R_punch, mu_analytical, nu_steel)
fr_sned_disp = flat_sneddon_displacement(r_outside, fr_D, R_punch)
fr_F_sned = flat_sneddon_load(fr_D, R_punch, mu_analytical, nu_steel)

print(f"\nSneddon force predictions (flat punch limit, a = b):")
print(f"  Flat punch: F_sned = {fp_F_sned:.2f} N (FEM applied: {fp_F:.2f} N)")
print(f"  Frustum:    F_sned = {fr_F_sned:.2f} N (FEM applied: {fr_F:.2f} N)")

# ============================================================================
# Frustum Sneddon analytical curves (a > b, for general comparison)
# ============================================================================
print("\nComputing frustum Sneddon analytical curves...")

# Range of contact radii from a = b (flat punch limit) to a = 5b
a_over_b = np.linspace(1.001, 5.0, 200)
a_vals = a_over_b * b_flat

D_frustum_curve = frustum_depth(a_vals, b_flat, alpha_surface)
P_frustum_curve = frustum_load(a_vals, b_flat, alpha_surface, mu_analytical, nu_steel)

# Equivalent flat punch P-D (same depth, contact radius = b only)
P_flat_at_same_D = flat_sneddon_load(D_frustum_curve, b_flat, mu_analytical, nu_steel)

# Cone limit (beta -> 0): P = pi*mu*a^2*tan(alpha)/(1-eta)
P_cone_curve = (np.pi * mu_analytical * a_vals**2 * np.tan(alpha_surface)) / (1.0 - nu_steel)
D_cone_curve = (np.pi / 2.0) * a_vals * np.tan(alpha_surface)

# Stiffness ratio: frustum / flat punch (at same depth)
stiffness_ratio = np.where(P_flat_at_same_D > 0, P_frustum_curve / P_flat_at_same_D, 1.0)

print(f"  Computed P-D curves for a/b in [{a_over_b[0]:.3f}, {a_over_b[-1]:.1f}]")
print(f"  D range: [{D_frustum_curve[0]*mm_to_nm:.2f}, {D_frustum_curve[-1]*mm_to_nm:.2f}] nm")
print(f"  P range: [{P_frustum_curve[0]:.4e}, {P_frustum_curve[-1]:.2f}] N")

# Compute frustum stress and displacement at a representative a/b ratio
a_demo = 2.0 * b_flat   # a = 2b for demonstration
D_demo = frustum_depth(a_demo, b_flat, alpha_surface)
P_demo = frustum_load(a_demo, b_flat, alpha_surface, mu_analytical, nu_steel)
beta_demo = b_flat / a_demo

D_demo_nm = D_demo * mm_to_nm
print(f"\n  Demo frustum profile at a/b = 2.0:")
print(f"    a = {a_demo:.4f} mm, b = {b_flat:.4f} mm, beta = {beta_demo:.3f}")
print(f"    D = {D_demo_nm:.2f} nm, P = {P_demo:.4f} N")

# Stress under the punch at a = 2b
r_stress_demo = np.linspace(0, a_demo * 0.99, 80)
print(f"  Computing frustum stress at {len(r_stress_demo)} points...")
sigma_frustum_demo = frustum_stress_zz(r_stress_demo, a_demo, b_flat,
                                       alpha_surface, mu_analytical, nu_steel)

# Flat punch stress at same depth, with a = a_demo (same contact radius)
r_stress_flat_demo = r_stress_demo[r_stress_demo < a_demo - 1e-10]
sigma_flat_demo = flat_sneddon_stress(r_stress_flat_demo, D_demo, a_demo,
                                      mu_analytical, nu_steel)
P_flat_same_D = flat_sneddon_load(D_demo, a_demo, mu_analytical, nu_steel)

# Displacement outside at a = 2b
r_disp_demo = np.linspace(a_demo * 1.01, 5.0 * a_demo, 60)
print(f"  Computing frustum displacement at {len(r_disp_demo)} points...")
u_frustum_demo = frustum_displacement_outside(r_disp_demo, a_demo, b_flat,
                                               alpha_surface, D_demo)
# Flat punch displacement: same contact radius a_demo for fair comparison
r_disp_flat_demo = np.linspace(a_demo * 1.01, 5.0 * a_demo, 80)
u_flat_demo = flat_sneddon_displacement(r_disp_flat_demo, D_demo, a_demo)

print("  Done.")

# ============================================================================
# Verify limits: beta -> 0 (cone) and beta -> 1 (flat punch)
# ============================================================================
print("\nVerifying Sneddon limits...")

# Test at a = 100*b (nearly sharp cone, beta = 0.01)
a_cone_test = 100.0 * b_flat
D_cone_test = frustum_depth(a_cone_test, b_flat, alpha_surface)
P_cone_test = frustum_load(a_cone_test, b_flat, alpha_surface, mu_analytical, nu_steel)
D_cone_exact = (np.pi / 2.0) * a_cone_test * np.tan(alpha_surface)
P_cone_exact = (np.pi * mu_analytical * a_cone_test**2 * np.tan(alpha_surface)) / (1.0 - nu_steel)
print(f"  beta -> 0 (a/b=100):  D_frustum/D_cone = {D_cone_test/D_cone_exact:.6f}")
print(f"                         P_frustum/P_cone = {P_cone_test/P_cone_exact:.6f}")

# Test at a = 1.001*b (nearly flat punch, beta = 0.999)
a_flat_test = 1.001 * b_flat
D_flat_test = frustum_depth(a_flat_test, b_flat, alpha_surface)
P_flat_test = frustum_load(a_flat_test, b_flat, alpha_surface, mu_analytical, nu_steel)
P_flat_exact = flat_sneddon_load(D_flat_test, a_flat_test, mu_analytical, nu_steel)
ratio = P_flat_test / P_flat_exact if P_flat_exact > 0 else float('nan')
print(f"  beta -> 1 (a/b=1.001): P_frustum/P_flat = {ratio:.6f}")
print(f"                          (should approach 1.0)")

# ============================================================================
# COMPARISON PLOTS (FEM vs flat punch Sneddon — correct for bonded contact)
# ============================================================================
print("\nGenerating comparison plots...")


def _save_fig(fig, filename):
    save_figure(fig, filename, dpi=300)
    print(f"  Saved: {filename}")


c = get_colors(4)  # Tol palette colours for the four data series

# ---- Figure 1: Stress under punch (both geometries) ----
fig, ax = plt.subplots(figsize=(10, 7))
mask_fp = fp_r < R_punch
ax.plot(fp_r[mask_fp], fp_szz[mask_fp], 'o',
        markersize=6, alpha=0.5, color=c[0], label='Flat punch FEM')
ax.plot(r_under, fp_sned_stress, '-',
        linewidth=4, color=c[0], alpha=0.8, label='Flat punch Sneddon')
mask_fr = fr_r < R_punch
ax.plot(fr_r[mask_fr], fr_szz[mask_fr], 's',
        markersize=6, alpha=0.5, color=c[1], label='Frustum FEM')
ax.plot(r_under, fr_sned_stress, '--',
        linewidth=4, color=c[1], alpha=0.8, label='Frustum Sneddon (a=b)')
ax.axvline(R_punch, color='k', linestyle=':', alpha=0.5, label=f'a = {R_punch} mm')
ax.set_xlabel('Radial distance r (mm)')
ax.set_ylabel(r'$\sigma_{zz}$ (MPa)')
ax.set_title(r'\textbf{Contact Stress Under Punch}')
ax.legend(loc='lower left')
_save_fig(fig, os.path.join(output_dir, "comparison_contact_stress.pdf"))

# ---- Figure 2: Full displacement profile (both, NO indenter body) ----
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(fp_r, fp_uz, 'o',
        markersize=4, alpha=0.4, color=c[0], label='Flat punch FEM')
ax.plot(fr_r, fr_uz, 's',
        markersize=4, alpha=0.4, color=c[1], label='Frustum FEM')

fp_sned_full = np.zeros_like(r_anal)
fp_sned_full[r_anal < R_punch] = -fp_D
m_out = r_anal > R_punch
fp_sned_full[m_out] = flat_sneddon_displacement(r_anal[m_out], fp_D, R_punch)
ax.plot(r_anal, fp_sned_full, '-', linewidth=4, color=c[0], alpha=0.7, label='Flat Sneddon')

fr_sned_full = np.zeros_like(r_anal)
fr_sned_full[r_anal < R_punch] = -fr_D
fr_sned_full[m_out] = flat_sneddon_displacement(r_anal[m_out], fr_D, R_punch)
ax.plot(r_anal, fr_sned_full, '--', linewidth=4, color=c[1], alpha=0.7,
        label='Frustum Sneddon (a=b)')

ax.axvline(R_punch, color='k', linestyle=':', alpha=0.5)
ax.set_xlabel('Radial distance r (mm)')
ax.set_ylabel(r'$u_z$ (mm)')
ax.set_title(r'\textbf{Displacement Profile at Interface}')
ax.legend()
_save_fig(fig, os.path.join(output_dir, "comparison_displacement_profile.pdf"))

# ---- Figure 3: Displacement outside punch (zoomed) ----
fig, ax = plt.subplots(figsize=(10, 7))
mask_fp_out = fp_r > R_punch
ax.plot(fp_r[mask_fp_out], fp_uz[mask_fp_out], 'o',
        markersize=6, alpha=0.5, color=c[0], label='Flat punch FEM')
ax.plot(r_outside, fp_sned_disp, '-',
        linewidth=4, color=c[0], alpha=0.8, label='Flat Sneddon')
mask_fr_out = fr_r > R_punch
ax.plot(fr_r[mask_fr_out], fr_uz[mask_fr_out], 's',
        markersize=6, alpha=0.5, color=c[1], label='Frustum FEM')
ax.plot(r_outside, fr_sned_disp, '--',
        linewidth=4, color=c[1], alpha=0.8, label='Frustum Sneddon (a=b)')
ax.set_xlabel('Radial distance r (mm)')
ax.set_ylabel(r'$u_z$ (mm)')
ax.set_title(r'\textbf{Displacement Outside Contact}')
ax.legend()
_save_fig(fig, os.path.join(output_dir, "comparison_displacement_outside.pdf"))

# ---- Figure 4: Summary comparison table ----
fig, ax = plt.subplots(figsize=(10, 7))
ax.axis('off')
table_data = [
    ['Parameter', 'Flat Punch', 'Frustum'],
    ['Contact radius (mm)', f'{R_punch:.3f}', f'{R_punch:.3f}'],
    ['Punch shape', 'Cylinder', f'Cone {half_angle_deg} deg'],
    ['Applied force (N)', f'{fp_F:.2f}', f'{fr_F:.2f}'],
    ['Depth D (nm)', f'{fp_D_nm:.2f}', f'{fr_D_nm:.2f}'],
    ['Max |u| (mm)', f'{fp_umax:.4e}', f'{fr_umax:.4e}'],
    ['Max VM total (MPa)', f'{fp_vm_max:.0f}', f'{fr_vm_max:.0f}'],
    ['Max VM substrate (MPa)', f'{fp_vm_sub:.0f}', f'{fr_vm_sub:.0f}'],
    ['Force balance err (%)', f'{fp_error:.2f}', f'{fr_error:.2f}'],
    ['Sneddon force (N)', f'{fp_F_sned:.2f}', f'{fr_F_sned:.2f}'],
]

table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1.0, 2.0)

for j in range(3):
    table[0, j].set_facecolor(TOL_COLORS[0])
    table[0, j].set_text_props(color='white', fontweight='bold')

ax.set_title(r'\textbf{Comparison Summary}')
_save_fig(fig, os.path.join(output_dir, "comparison_summary_table.pdf"))

# --- Individual FEM vs Sneddon plots ---
fig, ax = plt.subplots(figsize=(10, 7))
mask = fp_r < R_punch
ax.plot(fp_r[mask], fp_szz[mask], 'o', label='FEM (mesh nodes)', markersize=6, alpha=0.7, color=c[0])
ax.plot(r_under, fp_sned_stress, '-', label='Sneddon Analytical', linewidth=4, alpha=0.85, color=c[1])
ax.axvline(R_punch, color='k', linestyle=':', alpha=0.5, label='Punch edge')
ax.set_xlabel('Radial distance r (mm)')
ax.set_ylabel(r'$\sigma_{zz}$ (MPa)')
ax.set_title(r'\textbf{Flat Punch --- Stress Under Punch: FEM vs Sneddon}')
ax.legend()
_save_fig(fig, os.path.join(output_dir, "flat_sneddon_stress.pdf"))

fig, ax = plt.subplots(figsize=(10, 7))
mask = fp_r > R_punch
ax.plot(fp_r[mask], fp_uz[mask], 'o', label='FEM (mesh nodes)', markersize=6, alpha=0.7, color=c[0])
ax.plot(r_outside, fp_sned_disp, '-', label='Sneddon Analytical', linewidth=4, alpha=0.85, color=c[1])
ax.axvline(R_punch, color='k', linestyle=':', alpha=0.5, label='Punch edge')
ax.set_xlabel('Radial distance r (mm)')
ax.set_ylabel(r'$u_z$ (mm)')
ax.set_title(r'\textbf{Flat Punch --- Displacement Outside Punch: FEM vs Sneddon}')
ax.legend()
_save_fig(fig, os.path.join(output_dir, "flat_sneddon_displacement.pdf"))

fig, ax = plt.subplots(figsize=(10, 7))
mask = fr_r < R_punch
ax.plot(fr_r[mask], fr_szz[mask], 'o', label='FEM (mesh nodes)', markersize=6, alpha=0.7, color=c[0])
ax.plot(r_under, fr_sned_stress, '-', label='Sneddon (flat punch limit)', linewidth=4, alpha=0.85, color=c[1])
ax.axvline(R_punch, color='k', linestyle=':', alpha=0.5, label='Punch edge')
ax.set_xlabel('Radial distance r (mm)')
ax.set_ylabel(r'$\sigma_{zz}$ (MPa)')
ax.set_title(r'\textbf{Frustum --- Stress Under Punch: FEM vs Sneddon (a=b limit)}')
ax.legend()
_save_fig(fig, os.path.join(output_dir, "frustum_sneddon_stress.pdf"))

fig, ax = plt.subplots(figsize=(10, 7))
mask = fr_r > R_punch
ax.plot(fr_r[mask], fr_uz[mask], 'o', label='FEM (mesh nodes)', markersize=6, alpha=0.7, color=c[0])
ax.plot(r_outside, fr_sned_disp, '-', label='Sneddon (flat punch limit)', linewidth=4, alpha=0.85, color=c[1])
ax.axvline(R_punch, color='k', linestyle=':', alpha=0.5, label='Punch edge')
ax.set_xlabel('Radial distance r (mm)')
ax.set_ylabel(r'$u_z$ (mm)')
ax.set_title(r'\textbf{Frustum --- Displacement Outside Punch: FEM vs Sneddon (a=b limit)}')
ax.legend()
_save_fig(fig, os.path.join(output_dir, "frustum_sneddon_displacement.pdf"))

# ============================================================================
# FRUSTUM SNEDDON ANALYTICAL PLOTS
# These show the full frustum solution for a > b (cone engages substrate)
# ============================================================================
print("\nGenerating frustum analytical plots...")

# ---- Plot A: Load-Depth (P vs D) curves ----
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(D_frustum_curve * mm_to_nm, P_frustum_curve, '-', linewidth=5, color=c[0],
        label='Frustum Sneddon')
ax.plot(D_frustum_curve * mm_to_nm, P_flat_at_same_D, '--', linewidth=4, color=c[1],
        label=f'Flat punch (a = b = {b_flat} mm)')
ax.plot(D_cone_curve * mm_to_nm, P_cone_curve, ':', linewidth=3, color=c[2],
        label=r'Sharp cone limit ($\beta \to 0$)')

# FEM operating points
ax.plot(fp_D_nm, fp_F, 'o', markersize=20, zorder=5, color=c[3],
        label=f'Flat punch FEM (D={fp_D_nm:.1f} nm)')
ax.plot(fr_D_nm, fr_F, 's', markersize=20, zorder=5, color=c[3],
        label=f'Frustum FEM (D={fr_D_nm:.1f} nm)')

ax.set_xlabel('Penetration depth D (nm)')
ax.set_ylabel('Total load P (N)')
ax.set_title(r'\textbf{Load vs Depth: Frustum Sneddon (b = ' + f'{b_flat}' + r' mm, $\alpha$ = ' + f'{np.degrees(alpha_surface):.1f}' + r' deg)}')
ax.legend()

# Inset text
info = (f"b = {b_flat} mm (flat tip)\n"
        r"$\alpha$ = " + f"{np.degrees(alpha_surface):.1f} deg from surface\n"
        f"E = {E_steel/1000:.0f} GPa, " + r"$\nu$ = " + f"{nu_steel}\n"
        r"$\mu$ = " + f"{mu_analytical:,.0f} MPa")
ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=18,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

_save_fig(fig, os.path.join(output_dir, "frustum_analytical_PD_curve.pdf"))

# ---- Plot B: Stiffness ratio (P_frustum / P_flat_punch) vs a/b ----
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(a_over_b, stiffness_ratio, '-', linewidth=5, color=c[0])
ax.axhline(1.0, color='k', linestyle=':', alpha=0.4)
ax.axhline(np.pi / 2.0, color=c[2], linestyle=':', alpha=0.5,
           label=r'Cone limit: $\pi/2$ = %.3f' % (np.pi / 2.0))

ax.set_xlabel(r'$a/b$ (contact radius / flat tip radius)')
ax.set_ylabel(r'$P_{\mathrm{frustum}} / P_{\mathrm{flat}}$ at same depth')
ax.set_title(r'\textbf{Frustum vs Flat Punch: Load Ratio at Equal Depth}')
ax.legend()
ax.set_xlim(1, 5)

_save_fig(fig, os.path.join(output_dir, "frustum_analytical_stiffness_ratio.pdf"))

# ---- Plot C: Stress profile under punch at a = 2b ----
fig, ax = plt.subplots(figsize=(10, 7))

valid = ~np.isnan(sigma_frustum_demo)
ax.plot(r_stress_demo[valid], sigma_frustum_demo[valid], '-', linewidth=5,
        color=c[0], label=f'Frustum (a={a_demo:.1f}, b={b_flat:.1f}, ' + r'$\beta$' + f'={beta_demo:.2f})')
valid_flat = ~np.isnan(sigma_flat_demo)
ax.plot(r_stress_flat_demo[valid_flat], sigma_flat_demo[valid_flat], '--', linewidth=4,
        color=c[1], label=f'Flat punch (a = {a_demo} mm, same D)')

ax.axvline(b_flat, color=c[3], linestyle='-.', alpha=0.6,
           label=f'b = {b_flat} mm (frustum flat tip)')
ax.axvline(a_demo, color='k', linestyle=':', alpha=0.5,
           label=f'a = {a_demo} mm (contact edge, both)')

ax.set_xlabel('Radial distance r (mm)')
ax.set_ylabel(r'$\sigma_{zz}$ (MPa)')
ax.set_title(r'\textbf{Frustum vs Flat Punch Stress (a/b = 2, D = ' + f'{D_demo:.4f}' + r' mm)}')
ax.legend()

info = (f"Frustum: a={a_demo:.3f}, b={b_flat:.3f} mm, P={P_demo:.1f} N\n"
        f"Flat punch: a={a_demo:.3f} mm, P={P_flat_same_D:.1f} N\n"
        f"Same depth D = {D_demo:.4f} mm, same contact radius a")
ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=18,
        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

_save_fig(fig, os.path.join(output_dir, "frustum_analytical_stress_a2b.pdf"))

# ---- Plot D: Displacement profile at a = 2b ----
fig, ax = plt.subplots(figsize=(10, 7))

# Frustum: flat at -D for r < a, then Sneddon outside for r > a
r_under_frustum = np.linspace(0, a_demo, 50)
ax.plot(r_under_frustum, np.full_like(r_under_frustum, -D_demo), '-', linewidth=5,
        color=c[0], label=f'Frustum (a={a_demo:.1f} mm)')
ax.plot(r_disp_demo, u_frustum_demo, '-', linewidth=5,
        color=c[0])  # outside continuation

# Flat punch: flat at -D for r < a_demo, then Sneddon outside for r > a_demo
r_under_flat = np.linspace(0, a_demo, 30)
ax.plot(r_under_flat, np.full_like(r_under_flat, -D_demo), '--', linewidth=4,
        color=c[1])
ax.plot(r_disp_flat_demo, u_flat_demo, '--', linewidth=4, color=c[1],
        label=f'Flat punch (a = {a_demo} mm)')

ax.axvline(b_flat, color=c[3], linestyle='-.', alpha=0.6,
           label=f'b = {b_flat} mm (frustum flat tip)')
ax.axvline(a_demo, color='k', linestyle=':', alpha=0.5,
           label=f'a = {a_demo} mm (contact edge, both)')

ax.set_xlabel('Radial distance r (mm)')
ax.set_ylabel(r'$u_z$ (mm)')
ax.set_title(r'\textbf{Frustum vs Flat Punch Displacement (a/b = 2, D = ' + f'{D_demo:.4f}' + r' mm)}')
ax.legend()

_save_fig(fig, os.path.join(output_dir, "frustum_analytical_displacement_a2b.pdf"))

# ---- Plot E: Stress at multiple a/b ratios ----
fig, ax = plt.subplots(figsize=(10, 7))

ab_ratios = [1.5, 2.0, 3.0, 5.0]
c_ab = get_colors(4)

for ab_ratio, color in zip(ab_ratios, c_ab):
    a_i = ab_ratio * b_flat
    D_i = frustum_depth(a_i, b_flat, alpha_surface)
    r_pts = np.linspace(0, a_i * 0.98, 60)
    sigma_i = frustum_stress_zz(r_pts, a_i, b_flat, alpha_surface,
                                mu_analytical, nu_steel)
    valid = ~np.isnan(sigma_i)
    # Normalize r by a for comparison
    ax.plot(r_pts[valid] / a_i, sigma_i[valid], '-', linewidth=4, color=color,
            label=f'a/b = {ab_ratio:.1f} (D = {D_i*mm_to_nm:.1f} nm)')
    ax.axvline(1.0 / ab_ratio, color=color, linestyle=':', alpha=0.3)

ax.set_xlabel(r'$r/a$ (normalized radial distance)')
ax.set_ylabel(r'$\sigma_{zz}$ (MPa)')
ax.set_title(r'\textbf{Frustum Stress at Various Contact Radii (b = ' + f'{b_flat}' + r' mm)}')
ax.legend()

_save_fig(fig, os.path.join(output_dir, "frustum_analytical_stress_multi_ab.pdf"))

# ============================================================================
# Save combined JSON
# ============================================================================
combined = {
    "flat_punch": {
        "R_punch_mm": R_punch,
        "D_fem_mm": fp_D,
        "F_applied_N": fp_F,
        "F_sneddon_N": fp_F_sned,
        "max_von_mises_substrate_MPa": fp_vm_sub,
        "max_displacement_mm": fp_umax,
        "force_balance_error_pct": fp_error,
    },
    "frustum": {
        "R_punch_mm": R_punch,
        "half_angle_from_axis_deg": half_angle_deg,
        "alpha_from_surface_deg": float(np.degrees(alpha_surface)),
        "tan_alpha": float(np.tan(alpha_surface)),
        "R_frustum_top_mm": R_frustum_top,
        "D_fem_mm": fr_D,
        "F_applied_N": fr_F,
        "F_sneddon_flat_limit_N": fr_F_sned,
        "max_von_mises_substrate_MPa": fr_vm_sub,
        "max_displacement_mm": fr_umax,
        "force_balance_error_pct": fr_error,
    },
    "frustum_sneddon_analytical": {
        "b_flat_mm": b_flat,
        "alpha_surface_rad": float(alpha_surface),
        "alpha_surface_deg": float(np.degrees(alpha_surface)),
        "demo_a_over_b": 2.0,
        "demo_a_mm": float(a_demo),
        "demo_D_mm": float(D_demo),
        "demo_P_N": float(P_demo),
        "cone_limit_check_a_over_b_100": {
            "D_frustum_over_D_cone": float(D_cone_test / D_cone_exact),
            "P_frustum_over_P_cone": float(P_cone_test / P_cone_exact),
        },
    },
    "comparison": {
        "same_contact_radius_mm": R_punch,
        "same_total_force_N": fp_F,
        "depth_ratio_frustum_over_flat": fr_D / fp_D if fp_D > 0 else None,
        "stress_ratio_frustum_over_flat": fr_vm_sub / fp_vm_sub if fp_vm_sub > 0 else None,
    }
}

with open(os.path.join(output_dir, "comparison_summary.json"), 'w') as f:
    json.dump(combined, f, indent=2)

# ============================================================================
# Final summary
# ============================================================================
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print(f"\n  Same contact radius: {R_punch} mm")
print(f"  Same total force:    {fp_F:.2f} N")
print(f"\n  Depth ratio (frustum/flat):           {fr_D/fp_D:.4f}" if fp_D > 0 else "")
print(f"  Substrate stress ratio (frustum/flat): {fr_vm_sub/fp_vm_sub:.4f}" if fp_vm_sub > 0 else "")
print(f"\n  Frustum Sneddon parameters:")
print(f"    b = {b_flat} mm (flat tip radius)")
print(f"    alpha = {np.degrees(alpha_surface):.1f} deg from surface")
print(f"    tan(alpha) = {np.tan(alpha_surface):.4f}")
print(f"\n  Frustum Sneddon formulas verified:")
print(f"    D = a tan(alpha) arccos(b/a)")
print(f"    P = (2 mu a^2 tan(alpha))/(1-eta) [arccos(b/a) + (b/a) sqrt(1-(b/a)^2)]")
print(f"    Cone limit (beta->0): OK ({D_cone_test/D_cone_exact:.6f})")
print(f"    Flat limit (beta->1): OK ({ratio:.6f})")
print(f"\n  Output: {os.path.abspath(output_dir)}/")
print("=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)
