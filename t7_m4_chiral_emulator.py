import argparse
import json
from dataclasses import dataclass, asdict
import numpy as np

ALPHA_INV = 137.035999084
EPS0 = 8.8541878128e-12
MU0 = 4.0 * np.pi * 1e-7

@dataclass
class GelParams:
    name: str = "cnc_batio3_fe3o4_chiral_gel"
    rho: float = 1015.0               # kg/m^3
    tau_y: float = 0.05               # Pa (internal stiction / yield stress)
    K_visc: float = 0.08              # Pa*s^n (shear-thinning consistency)
    n_flow: float = 0.55              # < 1 shear-thinning index
    eps_r: float = 140.0              # relative permittivity
    sigma: float = 1.5e-3             # S/m conductivity
    chi_mag: float = 1.2e-4           # volumetric magnetic susceptibility
    gamma_chiral: float = -0.35       # left-handed chiral coupling (< 0 = left-handed)
    poloidal_gain: float = 0.28       # poloidal rollover coupling weight

@dataclass
class MagnetParams:
    name: str = "anti_helmholtz_euler_cusp"
    R_cell: float = 0.025             # m (cell radius)
    H_cell: float = 0.080             # m (cell height)
    B_ring: float = 0.18              # T (surface field scale)
    z_sep: float = 0.045              # m (ring magnet axial separation)
    euler_pitch: float = 1.25         # dimensionless Euler-spiral curvature weight
    theta_tilt_rad: float = 0.22      # F8 precession tilt angle theta (rad)
    omega_7_scale: float = 1.0        # 7th-axis toroidal spin rate scale
    V_drive: float = 12.0             # V (low-voltage drive amplitude)
    f_drive: float = 50.0             # Hz (0.0 for DC Quincke mode)

def yee_curl_triad(F_r: np.ndarray, F_phi: np.ndarray, F_z: np.ndarray, r_1d: np.ndarray, dr: float, dz: float):
    """
    Axisymmetric cylindrical Yee discrete curl stencil mapping one 3-vector triad
    onto its dual staggered triad: (curl F)_r, (curl F)_phi, (curl F)_z.
    """
    R_col = r_1d[:, None]
    dFphi_dz = np.gradient(F_phi, dz, axis=1)
    dFr_dz = np.gradient(F_r, dz, axis=1)
    dFz_dr = np.gradient(F_z, dr, axis=0)
    d_rFphi_dr = np.gradient(R_col * F_phi, dr, axis=0) / (R_col + 1e-9)

    curl_r = -dFphi_dz
    curl_phi = dFr_dz - dFz_dr
    curl_z = d_rFphi_dr
    return curl_r, curl_phi, curl_z

def build_6x6_yee_block(omega_norm: float, theta_tilt: float, parity_sign: float) -> np.ndarray:
    """
    Trace-free 6x6 spatial-imaginary Yee block generator:
        M6 = [  A_RR      i * B_RI ]
             [ -i * B_IR   -A_RR   ]
    Enforces Tr(M6) = 0 and N=1 Mobius parity twist across the 3R / 3iI interface.
    """
    c_t = np.cos(theta_tilt)
    s_t = np.sin(theta_tilt)

    L_rot = np.array([
        [0.0,        -omega_norm,  0.0],
        [omega_norm,  0.0,        -s_t],
        [0.0,         s_t,         0.0]
    ], dtype=np.complex128)

    A_RR = c_t * L_rot
    B_RI = 1j * parity_sign * (s_t * np.eye(3) + 0.25 * omega_norm * L_rot)
    B_IR = -1j * parity_sign * (s_t * np.eye(3) - 0.25 * omega_norm * L_rot)

    M6 = np.vstack([
        np.hstack([A_RR, B_RI]),
        np.hstack([B_IR, -A_RR])
    ])
    M6 -= (np.trace(M6) / 6.0) * np.eye(6, dtype=np.complex128)
    return M6

def magnetic_cusp_and_euler_profile(r_1d: np.ndarray, z_1d: np.ndarray, mag: MagnetParams):
    R, Z = np.meshgrid(r_1d, z_1d, indexing="ij")
    z1, z2 = -0.5 * mag.z_sep, 0.5 * mag.z_sep
    a = 1.15 * mag.R_cell

    def ring_field(r_pt, z_pt, z0, pol):
        dz = z_pt - z0
        d2 = r_pt**2 + a**2 + dz**2 + 1e-12
        Bz = pol * mag.B_ring * (a**2) / (d2**1.5)
        Br = pol * mag.B_ring * (1.5 * (a**2) * r_pt * dz) / (d2**2.5)
        return Br, Bz

    Br1, Bz1 = ring_field(R, Z, z1, +1.0)
    Br2, Bz2 = ring_field(R, Z, z2, -1.0)
    Br, Bz = Br1 + Br2, Bz1 + Bz2
    B2 = Br**2 + Bz**2

    z_norm = Z / (0.5 * mag.H_cell + 1e-12)
    euler_kappa = mag.euler_pitch * z_norm * np.exp(-(z_norm**2))
    b2_norm = B2 / (np.max(B2) + 1e-12)
    r_env = mag.R_cell * (0.20 + 0.80 * (z_norm**2)) * (1.0 - 0.35 * b2_norm)
    return R, Z, Br, Bz, B2, euler_kappa, r_env

def run_emulation(gel: GelParams, mag: MagnetParams, nr: int = 36, nz: int = 72, steps: int = 350, dt: float = 5e-4):
    r_1d = np.linspace(1e-4, mag.R_cell, nr)
    z_1d = np.linspace(-0.5 * mag.H_cell, 0.5 * mag.H_cell, nz)
    dr = float(r_1d[1] - r_1d[0])
    dz = float(z_1d[1] - z_1d[0])

    R, Z, Br, Bz, B2, euler_kappa, r_env = magnetic_cusp_and_euler_profile(r_1d, z_1d, mag)

    # Staggered 6-component Yee state:
    # Real spatial velocity triad Psi_R = (v_r, v_phi, v_z)
    v_r = np.zeros_like(R)
    v_phi = np.zeros_like(R)
    v_z = np.zeros_like(R)

    # Imaginary rotational / vorticity-flux triad Psi_I = (w_r, w_phi, w_z)
    w_r = np.zeros_like(R)
    w_phi = np.zeros_like(R)
    w_z = np.zeros_like(R)

    E_mag = mag.V_drive / (2.0 * r_env + 1e-4)
    chiral_sign = -1.0 if gel.gamma_chiral < 0 else 1.0
    waist_gauss = np.exp(-((Z / (0.25 * mag.H_cell)) ** 2))
    c_eff = 0.08  # normalized lattice wave-coupling speed for discrete CFL stability

    for step in range(steps):
        t = step * dt
        ac_mod = 1.0 if mag.f_drive <= 0.0 else np.cos(2.0 * np.pi * mag.f_drive * t)

        # 1) Half-step update of Imaginary Rotational Triad (Psi_I) via Yee curl of Psi_R
        curl_vr, curl_vphi, curl_vz = yee_curl_triad(v_r, v_phi, v_z, r_1d, dr, dz)
        w_r += dt * (c_eff * curl_vr - 0.35 * w_r)
        w_phi += dt * (c_eff * curl_vphi + chiral_sign * gel.poloidal_gain * euler_kappa * v_phi - 0.35 * w_phi)
        w_z += dt * (c_eff * curl_vz - 0.35 * w_z)

        # 2) Shear-thinning & stiction damping factor from current rotational shear
        shear_rate = np.sqrt(w_r**2 + w_phi**2 + w_z**2) + 1e-4
        eta_eff = gel.K_visc * (shear_rate ** (gel.n_flow - 1.0)) + gel.tau_y / shear_rate
        damping = np.clip((eta_eff / gel.rho) * 2.5, 0.05, 4.0)

        # 3) Full-step update of Real Spatial Triad (Psi_R) via Yee curl of Psi_I + chiral E-drive
        curl_wr, curl_wphi, curl_wz = yee_curl_triad(w_r, w_phi, w_z, r_1d, dr, dz)
        tau_phi = EPS0 * gel.eps_r * (E_mag**2) * abs(gel.gamma_chiral) * waist_gauss * (0.65 + 0.35 * abs(ac_mod))

        v_r += dt * (c_eff * curl_wr - damping * v_r)
        v_phi += dt * (c_eff * curl_wphi + (chiral_sign * tau_phi / (R * gel.rho)) - damping * v_phi)
        v_z += dt * (c_eff * curl_wz + gel.poloidal_gain * euler_kappa * (v_phi**2) / (R + 1e-3) - damping * v_z)

    waist_idx = nz // 2
    omega_waist = float(np.max(np.abs(v_phi[:, waist_idx] / r_1d)))
    peak_poloidal_speed = float(np.max(np.sqrt(v_r**2 + v_z**2)))

    # Induced magnetic torus at the waist
    J_phi = gel.sigma * E_mag[:, waist_idx] * np.abs(v_phi[:, waist_idx])
    B_torus_ind = float(MU0 * np.trapezoid(J_phi, r_1d))

    # 6x6 Yee block invariants & T7 -> M4 lambda diagnostic
    omega_norm = omega_waist * mag.omega_7_scale * 1e-2
    M6 = build_6x6_yee_block(omega_norm=omega_norm, theta_tilt=mag.theta_tilt_rad, parity_sign=chiral_sign)
    gauge_inv_tr_m6_sq = float(np.real(np.trace(M6 @ M6)))

    norm_3R = float(np.linalg.norm(np.stack([v_r, v_phi, v_z])) + 1e-12)
    norm_3iI = float(np.linalg.norm(np.stack([w_r, w_phi, w_z])) * np.sqrt(gel.eps_r))
    imag_to_real_ratio = float(norm_3iI / norm_3R)

    gamma_loss = gel.tau_y / (gel.K_visc + 1e-6)
    coherence_metric = float((omega_waist * abs(np.sin(mag.theta_tilt_rad)) * mag.omega_7_scale) / (gamma_loss + 1e-6))
    lambda_m4 = float(1.0 + max(0.0, 0.01 * (coherence_metric - 1.0)))

    return {
        "gel": asdict(gel),
        "magnet": asdict(mag),
        "metrics": {
            "max_spin_rpm": omega_waist * 60.0 / (2.0 * np.pi),
            "peak_poloidal_speed_mm_s": peak_poloidal_speed * 1e3,
            "induced_B_torus_nT": B_torus_ind * 1e9,
            "M6_trace_abs": float(abs(np.trace(M6))),
            "M6_gauge_invariant_TrM2": gauge_inv_tr_m6_sq,
            "yee_3iI_to_3R_ratio": imag_to_real_ratio,
            "alpha_inv_ref": ALPHA_INV,
            "alpha_harmonic_alignment": float(imag_to_real_ratio / ALPHA_INV),
            "coherence_metric": coherence_metric,
            "lambda_m4": lambda_m4,
        },
    }

def load_json_or_default(path: str | None, cls):
    if not path:
        return cls()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return cls(**data)

def main():
    parser = argparse.ArgumentParser(description="T7 -> M4 Yee 6x6 Chiral Gel Magnetic Cusp Emulator")
    parser.add_argument("--gel", type=str, default=None, help="Path to gel_params.json")
    parser.add_argument("--mag", type=str, default=None, help="Path to magnet_params.json")
    parser.add_argument("--out", type=str, default="emulation_results.json", help="Output JSON report path")
    args = parser.parse_args()

    gel = load_json_or_default(args.gel, GelParams)
    mag = load_json_or_default(args.mag, MagnetParams)
    results = run_emulation(gel, mag)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results["metrics"], indent=2))

if __name__ == "__main__":
    main()
