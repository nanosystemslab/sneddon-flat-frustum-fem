"""
Sneddon's Analytical Solutions for Axisymmetric Indentation

Reference: Sneddon, I.N. (1965) "The relation between load and penetration
in the axisymmetric Boussinesq problem for a punch of arbitrary profile"
Int. J. Engng Sci., Vol 3, pp. 47-57

This module implements analytical solutions for flat-end cylindrical punch
indentation (Section 6a, equations 6.1-6.3).
"""

import numpy as np


def flat_punch_force_displacement(a, D, E, nu):
    """
    Equation 6.1: Total load for given penetration depth

    P = 4μaD / (1-ν)

    Parameters:
    -----------
    a : float
        Punch radius (mm)
    D : float
        Penetration depth (mm)
    E : float
        Young's modulus (MPa)
    nu : float
        Poisson's ratio

    Returns:
    --------
    P : float
        Total load (N)
    """
    mu = E / (2 * (1 + nu))  # Shear modulus
    P = (4 * mu * a * D) / (1 - nu)
    return P


def flat_punch_stress(r, a, D, E, nu):
    """
    Equation 6.2: Stress distribution under punch

    σ_zz(r, 0) = -(2μD) / (π(1-ν)√(a²-r²))    for 0 ≤ r < a

    Note: Stress has singularity as r → a⁻ (punch edge)

    Parameters:
    -----------
    r : array-like
        Radial coordinates (mm) - must satisfy 0 ≤ r < a
    a : float
        Punch radius (mm)
    D : float
        Penetration depth (mm)
    E : float
        Young's modulus (MPa)
    nu : float
        Poisson's ratio

    Returns:
    --------
    sigma_zz : array-like
        Normal stress at z=0 (MPa), negative = compression
    """
    r = np.asarray(r)

    # Check bounds
    if np.any(r < 0):
        raise ValueError("r must be non-negative")
    if np.any(r >= a):
        raise ValueError(f"r must be less than a={a}")

    mu = E / (2 * (1 + nu))

    # Stress formula (note: becomes very large as r → a)
    sigma_zz = -(2 * mu * D) / (np.pi * (1 - nu) * np.sqrt(a**2 - r**2))

    return sigma_zz


def flat_punch_displacement(r, a, D):
    """
    Equation 6.3: Displacement outside contact area

    u_z(r, 0) = (2D/π) sin⁻¹(a/r)    for r > a

    Parameters:
    -----------
    r : array-like
        Radial coordinates (mm) - must satisfy r > a
    a : float
        Punch radius (mm)
    D : float
        Penetration depth (mm)

    Returns:
    --------
    u_z : array-like
        Vertical displacement at z=0 (mm), positive = upward
    """
    r = np.asarray(r)

    # Check bounds
    if np.any(r <= a):
        raise ValueError(f"r must be greater than a={a}")

    # Displacement formula
    u_z = (2 * D / np.pi) * np.arcsin(a / r)

    return u_z


def flat_punch_full_solution(a, D, E, nu, r_contact=None, r_free=None):
    """
    Complete analytical solution for flat-end cylindrical punch

    Combines equations 6.1, 6.2, 6.3 for convenience

    Parameters:
    -----------
    a : float
        Punch radius (mm)
    D : float
        Penetration depth (mm)
    E : float
        Young's modulus (MPa)
    nu : float
        Poisson's ratio
    r_contact : array-like, optional
        Radial coordinates under punch (0 ≤ r < a) for stress evaluation
    r_free : array-like, optional
        Radial coordinates outside contact (r > a) for displacement evaluation

    Returns:
    --------
    dict with keys:
        'P': Total load (N)
        'mu': Shear modulus (MPa)
        'stress_r': Radial coords for stress (mm), if r_contact provided
        'stress': σ_zz values (MPa), if r_contact provided
        'disp_r': Radial coords for displacement (mm), if r_free provided
        'disp': u_z values (mm), if r_free provided
    """
    # Material properties
    mu = E / (2 * (1 + nu))

    # Total load
    P = flat_punch_force_displacement(a, D, E, nu)

    result = {
        'P': P,
        'mu': mu,
        'a': a,
        'D': D,
        'E': E,
        'nu': nu
    }

    # Stress distribution under punch
    if r_contact is not None:
        r_contact = np.asarray(r_contact)
        # Filter to valid range (avoid singularity)
        r_valid = r_contact[r_contact < a * 0.999]  # Stay away from edge
        if len(r_valid) > 0:
            sigma_zz = flat_punch_stress(r_valid, a, D, E, nu)
            result['stress_r'] = r_valid
            result['stress'] = sigma_zz

    # Displacement outside contact
    if r_free is not None:
        r_free = np.asarray(r_free)
        r_valid = r_free[r_free > a]
        if len(r_valid) > 0:
            u_z = flat_punch_displacement(r_valid, a, D)
            result['disp_r'] = r_valid
            result['disp'] = u_z

    return result


def penetration_from_force(P, a, E, nu):
    """
    Inverse of equation 6.1: compute penetration depth from force

    D = P(1-ν) / (4μa)

    Parameters:
    -----------
    P : float
        Total load (N)
    a : float
        Punch radius (mm)
    E : float
        Young's modulus (MPa)
    nu : float
        Poisson's ratio

    Returns:
    --------
    D : float
        Penetration depth (mm)
    """
    mu = E / (2 * (1 + nu))
    D = P * (1 - nu) / (4 * mu * a)
    return D


def print_solution_summary(sol):
    """
    Print formatted summary of analytical solution

    Parameters:
    -----------
    sol : dict
        Result from flat_punch_full_solution()
    """
    print("="*70)
    print("SNEDDON ANALYTICAL SOLUTION - FLAT-END CYLINDRICAL PUNCH")
    print("="*70)

    print("\nGeometry:")
    print(f"  Punch radius: a = {sol['a']:.4f} mm")
    print(f"  Penetration depth: D = {sol['D']:.6f} mm")

    print("\nMaterial Properties:")
    print(f"  Young's modulus: E = {sol['E']:,.0f} MPa = {sol['E']/1000:.0f} GPa")
    print(f"  Poisson's ratio: ν = {sol['nu']:.3f}")
    print(f"  Shear modulus: μ = {sol['mu']:,.0f} MPa")

    print("\nLoad (Equation 6.1):")
    print(f"  P = 4μaD/(1-ν) = {sol['P']:.6f} N")

    if 'stress_r' in sol:
        print("\nStress Under Punch (Equation 6.2):")
        print(f"  σ_zz(r, 0) = -(2μD) / (π(1-ν)√(a²-r²))")
        print(f"  Evaluated at {len(sol['stress_r'])} points")
        print(f"  Range: r ∈ [0, {sol['stress_r'].max():.4f}] mm")
        print(f"  σ_zz range: [{sol['stress'].max():.2f}, {sol['stress'].min():.2f}] MPa")
        print(f"  (Note: singularity at r = a)")

    if 'disp_r' in sol:
        print("\nDisplacement Outside Contact (Equation 6.3):")
        print(f"  u_z(r, 0) = (2D/π) sin⁻¹(a/r)")
        print(f"  Evaluated at {len(sol['disp_r'])} points")
        print(f"  Range: r ∈ [{sol['disp_r'].min():.4f}, {sol['disp_r'].max():.4f}] mm")
        print(f"  u_z range: [{sol['disp'].min():.6f}, {sol['disp'].max():.6f}] mm")

    print("\n" + "="*70)


# Example usage
if __name__ == "__main__":
    # Example parameters (matching FEM simulation)
    a = 0.1       # mm - punch radius
    D = 0.001     # mm - penetration depth (will be computed from FEM)
    E = 200_000   # MPa - Young's modulus (steel)
    nu = 0.28     # Poisson's ratio

    # Compute full solution
    r_contact = np.linspace(0, 0.099, 50)  # Under punch (avoid edge)
    r_free = np.linspace(0.11, 3.0, 50)    # Outside contact

    sol = flat_punch_full_solution(a, D, E, nu,
                                     r_contact=r_contact,
                                     r_free=r_free)

    print_solution_summary(sol)

    # Example: Given force, compute penetration
    P = 31.416  # N (from FEM)
    D_computed = penetration_from_force(P, a, E, nu)
    print(f"\nInverse calculation:")
    print(f"  Given P = {P:.3f} N")
    print(f"  Computed D = {D_computed:.6f} mm")
