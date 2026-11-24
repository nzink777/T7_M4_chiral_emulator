import itertools
import json
from dataclasses import asdict
import numpy as np

from t7_m4_chiral_emulator import GelParams, MagnetParams, run_emulation


def expand_grid(base_dc, overrides: dict):
    keys = list(overrides.keys())
    values_lists = [overrides[k] for k in keys]
    configs = []
    for combo in itertools.product(*values_lists):
        d = asdict(base_dc)
        d.update(dict(zip(keys, combo)))
        configs.append(type(base_dc)(**d))
    return configs


def main():
    with open("data/sweep_params.json", "r", encoding="utf-8") as f:
        sweep_cfg = json.load(f)

    gel_list = expand_grid(GelParams(), sweep_cfg.get("gel", {}))
    mag_list = expand_grid(MagnetParams(), sweep_cfg.get("magnet", {}))

    runs = []
    for idx, (gel, mag) in enumerate(itertools.product(gel_list, mag_list), start=1):
        res = run_emulation(gel, mag)
        runs.append({
            "run_id": idx,
            "gel": asdict(gel),
            "magnet": asdict(mag),
            "metrics": res["metrics"],
        })

    metric_keys = list(runs[0]["metrics"].keys())
    summary = {}
    for k in metric_keys:
        vals = np.array([r["metrics"][k] for r in runs], dtype=np.float64)
        summary[k] = {
            "mean": float(np.mean(vals)),
            "variance": float(np.var(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    top_coherence = sorted(
        runs,
        key=lambda r: r["metrics"]["coherence_metric"],
        reverse=True,
    )[:5]

    output = {
        "total_runs": len(runs),
        "summary_stats": summary,
        "top_5_by_coherence": top_coherence,
        "runs": runs,
    }

    with open("data/sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Completed {len(runs)} runs. Key metric variances:")
    for k in ["max_spin_rpm", "induced_B_torus_nT", "coherence_metric", "yee_3iI_to_3R_ratio"]:
        print(f"  {k}: var={summary[k]['variance']:.6g}, max={summary[k]['max']:.6g}")


if __name__ == "__main__":
    main()
