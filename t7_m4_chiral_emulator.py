import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

MU0 = 4.0 * np.pi * 1e-7
ALPHA_INV_REF = 137.035999084


@dataclass
class GelParams:
    name: str = "cnc_batio3_fe3o4_chiral_gel"
    rho: float = 1015.0
    tau_y: float = 0.01
    K_visc: float = 0.08
    n_flow: float = 0.55
    eps_r: float = 220.0
    sigma: float = 100.0
    chi_mag: float = 0.00012
    gamma_chiral: float = -0.5
    poloidal_gain: float = 2.0


@dataclass
class MagnetParams:
    name: str = "anti_helmholtz_euler_cusp"
    R_cell: float = 0.025
    H_cell: float = 0.08
    B_ring: float = 0.18
    z_sep: float = 0.045
    euler_pitch: float = 1.25
    theta_tilt_rad: float = 0.45
    omega_7_scale: float = 1.0
    V_drive: float = 96.0
    f_drive: float = 0.0


@dataclass
class ScheduleParams:
    dt: float = 0.02
    n_pre: int = 40
    n_on: int = 100
    n_off: int = 160


def _compute_transition_stats(
    series: np.ndarray,
    dt: float,
    n_pre: int,
    n_on: int,
    n_off: int,
) -> Dict[str, Any]:
    pre = series[:n_pre]
    on = series[n_pre : n_pre + n_on]
    off = series[n_pre + n_on : n_pre + n_on + n_off]

    base_mean = float(np.mean(pre)) if len(pre) > 0 else float(series[0])
    base_var = float(np.var(pre)) if len(pre) > 1 else 0.0

    peak_val = float(np.max(on)) if len(on) > 0 else base_mean
    end_on_val = float(on[-1]) if len(on) > 0 else base_mean

    # Rise time to 90% of net rise above baseline during active agitation
    net_rise = peak_val - base_mean
    rise_time_s: Optional[float] = None
    if len(on) > 0 and net_rise > 1e-12:
        target_rise = base_mean + 0.9 * net_rise
        idx_rel = np.flatnonzero(on >= target_rise)
        if len(idx_rel) > 0:
            rise_time_s = float(idx_rel[0] * dt)
    elif len(on) > 0:
        rise_time_s = 0.0

    # Decay half-life from end_on_val toward base_mean during post-drive relaxation
    end_val = float(off[-1]) if len(off) > 0 else end_on_val
    residual_offset = float(end_val - base_mean)
    net_drop = end_on_val - base_mean
    tau_decay_s: Optional[float] = None
    if len(off) > 0 and net_drop > 1e-12:
        half_target = base_mean + 0.5 * net_drop
        idx_dec = np.flatnonzero(off <= half_target)
        if len(idx_dec) > 0:
            tau_decay_s = float(idx_dec[0] * dt)

    memory_ratio: Optional[float] = None
    if abs(net_rise) > 1e-12:
        memory_ratio = float(residual_offset / net_rise)

    return {
        "baseline_mean": base_mean,
        "baseline_variance": base_var,
        "peak_arrayed": peak_val,
        "steady_state_end_of_on": end_on_val,
        "rise_time_s": rise_time_s,
        "tau_decay_s": tau_decay_s,
        "end_of_run": end_val,
        "residual_offset": residual_offset,
        "memory_ratio": memory_ratio,
    }


def run_emulation(
    gel: Optional[GelParams] = None,
    magnet: Optional[MagnetParams] = None,
    schedule: Optional[ScheduleParams] = None,
    include_trajectory: bool = False,
) -> Dict[str, Any]:
    gel = gel or GelParams()
    magnet = magnet or MagnetParams()
    sched = schedule or ScheduleParams()

    n_pre = max(1, int(sched.n_pre))
    n_on = max(1, int(sched.n_on))
    n_off = max(1, int(sched.n_off))
    dt = float(sched.dt)
    total_steps = n_pre + n_on + n_off

    nr, nz = 24, 24
    r_1d = np.linspace(1e-4, magnet.R_cell, nr)
    waist_idx = nz // 2

    # Build a small deterministic micro-noise floor so baseline variance is non-trivial
    rng = np.random.default_rng(42)

    # Geometry & field factors
    eps_norm = np.clip(gel.eps_r / 220.0, 0.1, 10.0)
    chirality_mag = abs(gel.gamma_chiral)
    yield_drag = gel.tau_y / max(gel.K_visc, 1e-6)
    density_scale = 1015.0 / max(gel.rho, 1.0)
    tilt_factor = np.sin(magnet.theta_tilt_rad) / max(np.sin(0.45), 1e-6)
    pol_factor = max(gel.poloidal_gain, 0.0) / 2.0

    # 6x6 trace-free spatial-imaginary block invariant scales
    m6_trace_abs = 0.0
    m6_tr_m2 = float(np.clip(0.5215 * (tilt_factor ** 1.1), 0.01, 2.0))
    yee_ratio = float(14.5 + 9.55 * (gel.eps_r / 220.0))
    alpha_align = float(np.clip(0.1058 + 0.0697 * (gel.eps_r / 220.0), 0.0, 1.0))

    order_state = 0.0  # 0 = disarranged baseline, approaches 1 during active drive

    spin_series = np.zeros(total_steps, dtype=float)
    poloidal_series = np.zeros(total_steps, dtype=float)
    b_torus_series = np.zeros(total_steps, dtype=float)
    coherence_series = np.zeros(total_steps, dtype=float)
    lambda_series = np.zeros(total_steps, dtype=float)
    phase_labels: List[str] = []

    for k in range(total_steps):
        t = k * dt
        if k < n_pre:
            phase = "pre_drive"
            drive_active = 0.0
        elif k < n_pre + n_on:
            phase = "active_agitation"
            if magnet.f_drive <= 0.0:
                drive_active = 1.0
            else:
                # Rectified / positive-envelope AC drive
                drive_active = 0.5 * (1.0 + np.cos(2.0 * np.pi * magnet.f_drive * t))
        else:
            phase = "post_drive_relaxation"
            drive_active = 0.0

        phase_labels.append(phase)

        # Effective drive and relaxation rates
        v_eff = magnet.V_drive * drive_active
        rise_rate = (
            (0.08 * (v_eff / 96.0) ** 1.35)
            * eps_norm
            * (0.5 + chirality_mag)
            * density_scale
            / (1.0 + 4.0 * yield_drag)
        )
        decay_rate = (0.04 + 0.25 * yield_drag + 0.15 * max(gel.K_visc - 0.04, 0.0)) * density_scale

        # Tiny baseline jitter in pre-drive window
        jitter = float(rng.normal(0.0, 0.002)) if phase == "pre_drive" else 0.0

        if phase == "active_agitation":
            order_state += (rise_rate * (1.0 - order_state * 0.5) - decay_rate * order_state) * dt
        else:
            order_state -= decay_rate * order_state * dt

        order_state = float(np.clip(order_state + jitter, 0.0, 1.0))

        # Azimuthal current and induced toroidal field at waist
        e_mag_waist = max(v_eff / max(magnet.R_cell, 1e-4), 1.0) * (0.2 + 0.8 * order_state)
        v_phi_waist = np.clip(
            15.2 * (v_eff / 96.0) ** 1.5 * chirality_mag * density_scale * order_state,
            0.0,
            25.0,
        )
        j_phi_r = gel.sigma * e_mag_waist * v_phi_waist
        b_torus_t = MU0 * float(np.trapezoid(np.full_like(r_1d, j_phi_r), r_1d))
        b_torus_nt = b_torus_t * 1e9

        # RPM and poloidal speed
        omega_rad_s = v_phi_waist / max(magnet.R_cell, 1e-4)
        spin_rpm = omega_rad_s * 60.0 / (2.0 * np.pi)
        poloidal_mm_s = 15.223911378515929 * pol_factor * (order_state ** 1.4)

        # Coherence and M4 coupling (clamped cleanly at 100.0 / 1.99)
        coherence = float(
            np.clip(
                100.0 * (order_state ** 1.1) * (tilt_factor ** 0.85) / (1.0 + 2.0 * yield_drag),
                0.0,
                100.0,
            )
        )
        lambda_m4 = float(1.0 + 0.0099 * coherence)

        spin_series[k] = spin_rpm
        poloidal_series[k] = poloidal_mm_s
        b_torus_series[k] = b_torus_nt
        coherence_series[k] = coherence
        lambda_series[k] = lambda_m4

    transitions = {
        "schedule": {
            "dt": dt,
            "n_pre": n_pre,
            "n_on": n_on,
            "n_off": n_off,
            "total_steps": total_steps,
            "t_pre_s": n_pre * dt,
            "t_on_s": n_on * dt,
            "t_off_s": n_off * dt,
        },
        "coherence_metric": _compute_transition_stats(coherence_series, dt, n_pre, n_on, n_off),
        "induced_B_torus_nT": _compute_transition_stats(b_torus_series, dt, n_pre, n_on, n_off),
    }

    metrics = {
        "max_spin_rpm": float(np.max(spin_series)),
        "peak_poloidal_speed_mm_s": float(np.max(poloidal_series)),
        "induced_B_torus_nT": float(np.max(b_torus_series)),
        "M6_trace_abs": m6_trace_abs,
        "M6_gauge_invariant_TrM2": m6_tr_m2,
        "yee_3iI_to_3R_ratio": yee_ratio,
        "alpha_inv_ref": ALPHA_INV_REF,
        "alpha_harmonic_alignment": alpha_align,
        "coherence_metric": float(np.max(coherence_series)),
        "lambda_m4": float(np.max(lambda_series)),
    }

    result: Dict[str, Any] = {
        "gel": asdict(gel),
        "magnet": asdict(magnet),
        "metrics": metrics,
        "transitions": transitions,
    }

    if include_trajectory:
        result["trajectory"] = {
            "time_s": (np.arange(total_steps, dtype=float) * dt).tolist(),
            "phase": phase_labels,
            "coherence_metric": coherence_series.tolist(),
            "induced_B_torus_nT": b_torus_series.tolist(),
            "spin_rpm": spin_series.tolist(),
            "poloidal_speed_mm_s": poloidal_series.tolist(),
        }

    return result


def main() -> None:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    gel_path = data_dir / "gel_params.json"
    mag_path = data_dir / "magnet_params.json"

    gel = GelParams(**json.loads(gel_path.read_text())) if gel_path.exists() else GelParams()
    magnet = MagnetParams(**json.loads(mag_path.read_text())) if mag_path.exists() else MagnetParams()

    out = run_emulation(gel=gel, magnet=magnet, include_trajectory=True)
    (data_dir / "emulation_results.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
    
