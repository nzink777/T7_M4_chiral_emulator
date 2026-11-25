import itertools
import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from t7_m4_chiral_emulator import GelParams, MagnetParams, ScheduleParams, run_emulation

METRIC_KEYS = [
    "max_spin_rpm",
    "peak_poloidal_speed_mm_s",
    "induced_B_torus_nT",
    "M6_trace_abs",
    "M6_gauge_invariant_TrM2",
    "yee_3iI_to_3R_ratio",
    "alpha_inv_ref",
    "alpha_harmonic_alignment",
    "coherence_metric",
    "lambda_m4",
]


def _expand_axis(values: Any, fallback: Any) -> List[Any]:
    if isinstance(values, list) and len(values) > 0:
        return values
    if values is not None:
        return [values]
    return [fallback]


def _compute_summary_stats(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for key in METRIC_KEYS:
        vals = np.array([r["metrics"][key] for r in runs], dtype=float)
        stats[key] = {
            "mean": float(np.mean(vals)),
            "variance": float(np.var(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return stats


def _compute_transition_summary(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for obs in ("coherence_metric", "induced_B_torus_nT"):
        for stat_name in ("baseline_mean", "peak_arrayed", "residual_offset", "memory_ratio"):
            vals = [
                r["transitions"][obs][stat_name]
                for r in runs
                if r.get("transitions", {}).get(obs, {}).get(stat_name) is not None
            ]
            if vals:
                arr = np.array(vals, dtype=float)
                out.setdefault(obs, {})[stat_name] = {
                    "mean": float(np.mean(arr)),
                    "variance": float(np.var(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                }
    return out


def run_sweep(sweep_cfg: Dict[str, Any], include_trajectory: bool = False) -> Dict[str, Any]:
    base_gel = GelParams()
    base_mag = MagnetParams()
    base_sched = ScheduleParams()

    gel_cfg = sweep_cfg.get("gel", {})
    mag_cfg = sweep_cfg.get("magnet", {})
    sched_cfg = sweep_cfg.get("schedule", {})

    gel_axes = {
        "rho": _expand_axis(gel_cfg.get("rho"), base_gel.rho),
        "tau_y": _expand_axis(gel_cfg.get("tau_y"), base_gel.tau_y),
        "K_visc": _expand_axis(gel_cfg.get("K_visc"), base_gel.K_visc),
        "n_flow": _expand_axis(gel_cfg.get("n_flow"), base_gel.n_flow),
        "eps_r": _expand_axis(gel_cfg.get("eps_r"), base_gel.eps_r),
        "sigma": _expand_axis(gel_cfg.get("sigma"), base_gel.sigma),
        "chi_mag": _expand_axis(gel_cfg.get("chi_mag"), base_gel.chi_mag),
        "gamma_chiral": _expand_axis(gel_cfg.get("gamma_chiral"), base_gel.gamma_chiral),
        "poloidal_gain": _expand_axis(gel_cfg.get("poloidal_gain"), base_gel.poloidal_gain),
    }

    mag_axes = {
        "theta_tilt_rad": _expand_axis(mag_cfg.get("theta_tilt_rad"), base_mag.theta_tilt_rad),
        "V_drive": _expand_axis(mag_cfg.get("V_drive"), base_mag.V_drive),
        "f_drive": _expand_axis(mag_cfg.get("f_drive"), base_mag.f_drive),
    }

    dt = float(sched_cfg.get("dt", base_sched.dt))
    n_pre = int(sched_cfg.get("n_pre", base_sched.n_pre))
    t_on_list = _expand_axis(sched_cfg.get("t_on_steps"), [base_sched.n_on])
    t_off_list = _expand_axis(sched_cfg.get("t_off_steps"), [base_sched.n_off])

    gel_keys = list(gel_axes.keys())
    mag_keys = list(mag_axes.keys())

    runs: List[Dict[str, Any]] = []
    run_id = 0

    for g_vals, m_vals, n_on_val, n_off_val in itertools.product(
        itertools.product(*(gel_axes[k] for k in gel_keys)),
        itertools.product(*(mag_axes[k] for k in mag_keys)),
        t_on_list,
        t_off_list,
    ):
        run_id += 1
        g_kwargs = dict(zip(gel_keys, g_vals))
        m_kwargs = dict(zip(mag_keys, m_vals))

        gel_obj = GelParams(name=base_gel.name, **g_kwargs)
        mag_obj = MagnetParams(
            name=base_mag.name,
            R_cell=base_mag.R_cell,
            H_cell=base_mag.H_cell,
            B_ring=base_mag.B_ring,
            z_sep=base_mag.z_sep,
            euler_pitch=base_mag.euler_pitch,
            omega_7_scale=base_mag.omega_7_scale,
            **m_kwargs,
        )
        sched_obj = ScheduleParams(
            dt=dt,
            n_pre=n_pre,
            n_on=int(n_on_val),
            n_off=int(n_off_val),
        )

        sim = run_emulation(
            gel=gel_obj,
            magnet=mag_obj,
            schedule=sched_obj,
            include_trajectory=include_trajectory,
        )
        entry: Dict[str, Any] = {
            "run_id": run_id,
            "gel": sim["gel"],
            "magnet": sim["magnet"],
            "metrics": sim["metrics"],
            "transitions": sim["transitions"],
        }
        if include_trajectory:
            entry["trajectory"] = sim["trajectory"]
        runs.append(entry)

    top_5 = sorted(runs, key=lambda r: r["metrics"]["coherence_metric"], reverse=True)[:5]

    return {
        "total_runs": len(runs),
        "summary_stats": _compute_summary_stats(runs),
        "transition_summary": _compute_transition_summary(runs),
        "top_5_by_coherence": top_5,
        "runs": runs,
    }


def main() -> None:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    params_path = data_dir / "sweep_params.json"
    sweep_cfg = json.loads(params_path.read_text()) if params_path.exists() else {}

    summary = run_sweep(sweep_cfg, include_trajectory=False)
    (data_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
    
