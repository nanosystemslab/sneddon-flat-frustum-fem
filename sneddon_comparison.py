"""
Sneddon analytical comparison: Flat punch vs Frustum (flat-ended cone, FEC).

Standalone script — no FEM results needed.
Both solutions evaluated at the SAME total force F.
Both punches have the SAME flat contact radius (a_flat = b_fec = 0.1 mm).

At low loads the FEC cone doesn't engage, so a_fec ≈ b and both should
give identical results (validation test).

Flat punch:  Sneddon 1965 eqs 6.1-6.3
Frustum:     Sneddon 1965 general framework (flat-tipped cone)
"""
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.special import ellipk, ellipkinc
from scipy.optimize import brentq
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

# ============================================================================
# Parameters
# ============================================================================
# Material (steel substrate — Sneddon assumes rigid indenter, elastic half-space)
E = 200_000.0       # MPa
nu = 0.28
mu = E / (2.0 * (1.0 + nu))

# Geometry — BOTH punches have the same flat radius
R_flat = 0.1        # mm  flat punch radius = frustum flat tip radius
b_fec = R_flat      # mm  frustum flat tip radius (same!)

# Frustum cone angle
half_angle_from_axis_deg = 59.7
alpha_surface = np.radians(90.0 - half_angle_from_axis_deg)  # from surface

# Target force
F_target = 100.0    # N  low load to validate FEC ≈ flat punch

output_dir = "results_comparison"
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("SNEDDON ANALYTICAL COMPARISON")
print("=" * 70)
print(f"  E = {E/1000:.0f} GPa, nu = {nu}, mu = {mu:,.0f} MPa")
print(f"  Flat punch radius  a = {R_flat} mm")
print(f"  FEC flat tip       b = {b_fec} mm  (same!)")
print(f"  Cone angle from surface: {np.degrees(alpha_surface):.1f} deg")
print(f"  tan(alpha) = {np.tan(alpha_surface):.4f}")
print(f"  Target force F = {F_target} N")

# ============================================================================
# Flat punch Sneddon (eqs 6.1-6.3)
# ============================================================================

def flat_depth_from_load(F, a, mu, nu):
    """D = F*(1-nu) / (4*mu*a)"""
    return F * (1.0 - nu) / (4.0 * mu * a)


def flat_load(D, a, mu, nu):
    """P = 4*mu*a*D / (1-nu)"""
    return 4.0 * mu * a * D / (1.0 - nu)


def flat_stress(r, D, a, mu, nu):
    """sigma_zz under punch (r < a)"""
    r_safe = np.minimum(np.asarray(r, dtype=float), a - 1e-10)
    return -(2.0 * mu * D) / (np.pi * (1.0 - nu)) / np.sqrt(a**2 - r_safe**2)


def flat_displacement(r, D, a):
    """u_z outside punch (r > a)"""
    r_safe = np.maximum(np.asarray(r, dtype=float), a + 1e-10)
    return -(2.0 * D / np.pi) * np.arcsin(a / r_safe)


# ============================================================================
# Frustum / FEC Sneddon (flat-tipped cone)
# ============================================================================

def fec_depth(a, b, alpha):
    """D = a * tan(alpha) * arccos(b/a)"""
    if a <= b * (1.0 + 1e-14):
        return 0.0
    return a * np.tan(alpha) * np.arccos(b / a)


def fec_load(a, b, alpha, mu, nu):
    """P = (2*mu*a^2*tan(alpha))/(1-nu) * [arccos(b/a) + (b/a)*sqrt(1-(b/a)^2)]"""
    if a <= b * (1.0 + 1e-14):
        return 0.0
    beta = b / a
    return (2.0 * mu * a**2 * np.tan(alpha)) / (1.0 - nu) * (
        np.arccos(beta) + beta * np.sqrt(1.0 - beta**2))


def fec_contact_radius_from_load(F_target, b, alpha, mu, nu):
    """Invert fec_load to find a for a given force F."""
    if F_target <= 0:
        return b
    a_max = max(50.0 * b, 10.0)

    def residual(a):
        return fec_load(a, b, alpha, mu, nu) - F_target

    return brentq(residual, b * (1.0 + 1e-10), a_max, xtol=1e-14)


def fec_stress_zz(rho, a, b, alpha, mu, nu):
    """sigma_zz under FEC punch."""
    eps = a * np.tan(alpha)
    beta = b / a
    prefactor = -(2.0 * mu * eps) / (np.pi * a * (1.0 - nu))

    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    sigma = np.full_like(rho, np.nan)

    for i, rho_i in enumerate(rho):
        x = rho_i / a
        if x >= 1.0 - 1e-10:
            continue
        if abs(x - beta) < 1e-8:
            continue

        if x < beta:
            def integrand_IA(t, _x=x, _b=beta):
                return np.arccos(_b / t) / np.sqrt(t**2 - _x**2)
            I_A, _ = integrate.quad(integrand_IA, beta, 1.0,
                                    points=[beta], limit=200)
            k = x / beta
            m = k**2
            if m < 0.999:
                I_B = (1.0 / beta) * (ellipk(m) - ellipkinc(np.arcsin(beta), m))
            else:
                def integrand_IB(t, _x=x, _b=beta):
                    return 1.0 / np.sqrt((t**2 - _b**2) * (t**2 - _x**2))
                IB_raw, _ = integrate.quad(integrand_IB, beta, 1.0,
                                           points=[beta], limit=200)
                I_B = beta * IB_raw
            sigma[i] = prefactor * (I_A + I_B)
        else:
            def integrand_IA_prime(t, _x=x, _b=beta):
                return np.arccos(_b / t) / np.sqrt(t**2 - _x**2)
            I_A_prime, _ = integrate.quad(integrand_IA_prime, x, 1.0,
                                          points=[x], limit=200)
            k = beta / x
            m = k**2
            I_B_prime = (1.0 / x) * (ellipk(m) - ellipkinc(np.arcsin(x), m))
            sigma[i] = prefactor * (I_A_prime + I_B_prime)

    return sigma


def fec_displacement_outside(rho, a, b, alpha, D):
    """u_z outside FEC contact (rho > a)."""
    eps = a * np.tan(alpha)
    beta = b / a

    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    u_z = np.zeros_like(rho)

    for i, rho_i in enumerate(rho):
        x = rho_i / a
        if x <= 1.0 + 1e-12:
            u_z[i] = -D
            continue
        flat_part = (2.0 * D / np.pi) * np.arcsin(1.0 / x)
        def integrand(t, _x=x, _b=beta):
            return t * np.arccos(_b / t) / np.sqrt(_x**2 - t**2)
        corr, _ = integrate.quad(integrand, beta, 1.0 - 1e-12,
                                 points=[beta], limit=200)
        u_z[i] = -(flat_part - (2.0 * eps / np.pi) * corr)

    return u_z


# ============================================================================
# Compute both solutions at F = F_target
# ============================================================================

# --- Flat punch at F_target (a = R_flat) ---
D_flat = flat_depth_from_load(F_target, R_flat, mu, nu)
F_flat_check = flat_load(D_flat, R_flat, mu, nu)

print(f"\nFlat punch (a = {R_flat} mm):")
print(f"  D = {D_flat:.6f} mm")
print(f"  F = {F_flat_check:.2f} N (check)")

# --- FEC at F_target (b = R_flat, solve for a) ---
a_fec = fec_contact_radius_from_load(F_target, b_fec, alpha_surface, mu, nu)
D_fec = fec_depth(a_fec, b_fec, alpha_surface)
F_fec_check = fec_load(a_fec, b_fec, alpha_surface, mu, nu)
beta_fec = b_fec / a_fec

print(f"\nFEC (b = {b_fec} mm):")
print(f"  a = {a_fec:.6f} mm (contact radius, solved from F)")
print(f"  a/b = {a_fec/b_fec:.6f}")
print(f"  beta = {beta_fec:.6f}")
print(f"  D = {D_fec:.6f} mm")
print(f"  F = {F_fec_check:.2f} N (check)")

print(f"\nComparison at F = {F_target} N:")
print(f"  D_flat = {D_flat:.6f} mm")
print(f"  D_fec  = {D_fec:.6f} mm")
print(f"  D ratio (fec/flat) = {D_fec/D_flat:.6f}")
print(f"  a_flat = {R_flat:.4f} mm")
print(f"  a_fec  = {a_fec:.6f} mm")
print(f"  a_fec - b = {(a_fec - b_fec):.2e} mm (cone engagement beyond flat tip)")

# ============================================================================
# Plot 1: FEC indenter profile (shape check)
# ============================================================================
print("\nPlotting FEC indenter profile...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: physical indenter shape
ax = axes[0]
r_profile = np.linspace(0, 0.4, 500)
z_flat = np.zeros_like(r_profile)
z_fec = np.zeros_like(r_profile)

for i, r in enumerate(r_profile):
    # Flat punch: flat face at z=0 for r < R_flat, then punch wall goes up
    if r <= R_flat:
        z_flat[i] = 0.0
    else:
        z_flat[i] = np.nan  # no contact outside

    # FEC: flat at z=0 for r < b, then cone rises at angle alpha
    if r <= b_fec:
        z_fec[i] = 0.0
    else:
        z_fec[i] = (r - b_fec) * np.tan(alpha_surface)

ax.plot(r_profile, z_fec, '-', linewidth=2.5, color='C0', label='FEC profile')
ax.plot(r_profile[r_profile <= R_flat], z_flat[r_profile <= R_flat], '--',
        linewidth=2, color='C1', label='Flat punch profile')

# Mark key dimensions
ax.axvline(b_fec, color='C3', linestyle='-.', alpha=0.6,
           label=f'b = {b_fec} mm (flat tip edge)')
ax.axvline(R_flat, color='C1', linestyle=':', alpha=0.4,
           label=f'a_flat = {R_flat} mm')
if a_fec > b_fec * 1.001:
    ax.axvline(a_fec, color='C0', linestyle=':', alpha=0.4,
               label=f'a_fec = {a_fec:.4f} mm')

# Show the deformed surface level
ax.axhline(-D_flat, color='C1', linestyle='--', alpha=0.3, linewidth=1)
ax.axhline(-D_fec, color='C0', linestyle='-', alpha=0.3, linewidth=1)

ax.set_xlabel('Radial distance r (mm)')
ax.set_ylabel('Indenter surface z (mm)')
ax.set_title('Indenter Profiles (Sneddon rigid punch shape)')
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 0.35)
ax.set_ylim(-0.02, 0.15)
ax.set_aspect('equal')

# Right: zoomed near contact edge to show cone engagement
ax = axes[1]

r_zoom = np.linspace(0.08, 0.18, 500)
z_fec_zoom = np.zeros_like(r_zoom)
for i, r in enumerate(r_zoom):
    if r <= b_fec:
        z_fec_zoom[i] = 0.0
    else:
        z_fec_zoom[i] = (r - b_fec) * np.tan(alpha_surface)

ax.plot(r_zoom, z_fec_zoom, '-', linewidth=2.5, color='C0', label='FEC profile')
ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
ax.axvline(b_fec, color='C3', linestyle='-.', alpha=0.6,
           label=f'b = {b_fec} mm')
if a_fec > b_fec * 1.001:
    ax.axvline(a_fec, color='C0', linestyle=':', alpha=0.6,
               label=f'a_fec = {a_fec:.4f} mm')
    # Shade the cone engagement region
    r_cone = np.linspace(b_fec, a_fec, 100)
    z_cone = (r_cone - b_fec) * np.tan(alpha_surface)
    ax.fill_between(r_cone, 0, z_cone, alpha=0.15, color='C0',
                    label='Cone engagement zone')

ax.set_xlabel('Radial distance r (mm)')
ax.set_ylabel('z (mm)')
ax.set_title(f'Zoom: Cone Engagement at F = {F_target} N')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

info = (f"a_fec = {a_fec:.6f} mm\n"
        f"b = {b_fec} mm\n"
        f"a/b = {a_fec/b_fec:.6f}\n"
        f"Cone engagement = {(a_fec-b_fec)*1000:.3f} um")
ax.text(0.98, 0.98, info, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle(f'FEC Indenter Profile — alpha = {np.degrees(alpha_surface):.1f} deg from surface',
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "sneddon_fec_profile.pdf"), dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {output_dir}/sneddon_fec_profile.pdf")

# ============================================================================
# Plot 2: Stress comparison
# ============================================================================
print("Computing stress profiles...")

a_max = max(R_flat, a_fec)

# Flat punch stress
r_stress_flat = np.linspace(0, R_flat * 0.99, 100)
sigma_flat = flat_stress(r_stress_flat, D_flat, R_flat, mu, nu)

# FEC stress
r_stress_fec = np.linspace(0, a_fec * 0.99, 100)
sigma_fec = fec_stress_zz(r_stress_fec, a_fec, b_fec, alpha_surface, mu, nu)

fig, ax = plt.subplots(figsize=(8, 6))

valid = ~np.isnan(sigma_fec)
ax.plot(r_stress_fec[valid], sigma_fec[valid], '-', linewidth=2.5,
        color='C0', label=f'FEC (a={a_fec:.4f}, b={b_fec})')
ax.plot(r_stress_flat, sigma_flat, '--', linewidth=2,
        color='C1', label=f'Flat punch (a={R_flat})')

ax.axvline(b_fec, color='C3', linestyle='-.', alpha=0.6,
           label=f'b = {b_fec} mm (flat tip edge)')

ax.set_xlabel('Radial distance r (mm)')
ax.set_ylabel('$\\sigma_{zz}$ (MPa)')
ax.set_title(f'Contact Stress — Same Force F = {F_target} N')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

info = (f"F = {F_target} N for both\n"
        f"Flat: a={R_flat}, D={D_flat:.6f} mm\n"
        f"FEC:  a={a_fec:.4f}, b={b_fec}, D={D_fec:.6f} mm\n"
        f"D ratio = {D_fec/D_flat:.6f}")
ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=8,
        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.tight_layout()
fig.savefig(os.path.join(output_dir, "sneddon_same_force_stress.pdf"), dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {output_dir}/sneddon_same_force_stress.pdf")

# ============================================================================
# Plot 3: Displacement comparison
# ============================================================================
print("Computing displacement profiles...")

r_disp = np.linspace(a_max * 1.01, 5.0 * a_max, 80)
u_flat_outside = flat_displacement(r_disp, D_flat, R_flat)
u_fec_outside = fec_displacement_outside(r_disp, a_fec, b_fec,
                                          alpha_surface, D_fec)

fig, ax = plt.subplots(figsize=(8, 6))

# FEC
r_under_fec = np.linspace(0, a_fec, 50)
ax.plot(r_under_fec, np.full_like(r_under_fec, -D_fec), '-',
        linewidth=2.5, color='C0', label=f'FEC (a={a_fec:.4f})')
ax.plot(r_disp, u_fec_outside, '-', linewidth=2.5, color='C0')

# Flat punch
r_under_flat = np.linspace(0, R_flat, 30)
ax.plot(r_under_flat, np.full_like(r_under_flat, -D_flat), '--',
        linewidth=2, color='C1')
ax.plot(r_disp, u_flat_outside, '--', linewidth=2, color='C1',
        label=f'Flat punch (a={R_flat})')

ax.axvline(b_fec, color='C3', linestyle='-.', alpha=0.6,
           label=f'b = {b_fec} mm')

ax.set_xlabel('Radial distance r (mm)')
ax.set_ylabel('$u_z$ (mm)')
ax.set_title(f'Surface Displacement — Same Force F = {F_target} N')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

info = (f"F = {F_target} N for both\n"
        f"Flat: D={D_flat:.6f} mm\n"
        f"FEC:  D={D_fec:.6f} mm")
ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=8,
        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.tight_layout()
fig.savefig(os.path.join(output_dir, "sneddon_same_force_displacement.pdf"), dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {output_dir}/sneddon_same_force_displacement.pdf")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY — Same Force Comparison")
print("=" * 70)
print(f"  F = {F_target} N")
print(f"  Material: E={E/1000:.0f} GPa, nu={nu}")
print(f"  Flat punch:  a = {R_flat} mm,  D = {D_flat:.6f} mm")
print(f"  FEC:         a = {a_fec:.6f} mm (b={b_fec}),  D = {D_fec:.6f} mm")
print(f"  a_fec/b = {a_fec/b_fec:.6f}")
print(f"  Cone engagement beyond flat tip: {(a_fec - b_fec)*1e3:.4f} um")
print(f"  D ratio (fec/flat) = {D_fec/D_flat:.6f}")
if abs(D_fec/D_flat - 1.0) < 0.01:
    print(f"  --> FEC matches flat punch within 1% (as expected at low load)")
else:
    print(f"  --> FEC differs from flat punch by {abs(D_fec/D_flat - 1.0)*100:.1f}%")
print(f"\n  Output: {os.path.abspath(output_dir)}/")
print("=" * 70)
