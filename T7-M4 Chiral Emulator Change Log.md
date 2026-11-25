# Change Log — `nzink777/T7_M4_chiral_emulator`

## [Unreleased] — Three-Phase Transition Cycle & Run #7 Benchmark Lock

### Added
* **Explicit Three-Phase Drive/Relaxation Schedule (`t7_m4_chiral_emulator.py`):**
  * Introduced `ScheduleParams` (`dt`, `n_pre`, `n_on`, `n_off`) to partition each simulation into three sequential windows:
    1. **Pre-drive baseline (`pre_drive`):** Zero electric drive (`V_eff = 0.0`) with static anti-Helmholtz cusp confinement to measure un-driven baseline floor and variance.
    2. **Active agitation (`active_agitation`):** Applied drive (`V_drive`, `f_drive`) to track organization from disarranged to arrayed states.
    3. **Post-drive relaxation (`post_drive_relaxation`):** Drive returned to zero (`V_eff = 0.0`) to observe decay dynamics and hysteresis.
* **Transition Metrics Extraction (`_compute_transition_stats`):**
  * Added automated per-run transition observables for both `coherence_metric` and `induced_B_torus_nT`:
    * `baseline_mean` and `baseline_variance` (pre-drive window).
    * `peak_arrayed`, `steady_state_end_of_on`, and `rise_time_s` (90% net rise threshold during active agitation).
    * `tau_decay_s` (decay half-life), `end_of_run`, `residual_offset`, and `memory_ratio` (post-drive relaxation window).
* **Sweep Transition Aggregation (`sweep_emulator.py`):**
  * Added `_compute_transition_summary()` to aggregate mean, variance, min, and max across `baseline_mean`, `peak_arrayed`, `residual_offset`, and `memory_ratio` in `data/sweep_summary.json`.
  * Added `schedule` axis expansion (`t_on_steps`, `t_off_steps`) to `run_sweep()`.
* **Trajectory Logging Controls:**
  * Enabled full step-by-step `trajectory` output (`time_s`, `phase`, `coherence_metric`, `induced_B_torus_nT`, `spin_rpm`, `poloidal_speed_mm_s`) in `data/emulation_results.json` (`include_trajectory=True`).
  * Kept `include_trajectory=False` by default in `sweep_emulator.py` to keep `data/sweep_summary.json` lightweight and GitHub Actions CI runs bounded.

### Changed
* **Default Reference Parameters Locked to Coupled Sweep Run #7:**
  * Updated `GelParams` defaults to `rho = 1015.0`, `tau_y = 0.01`, `K_visc = 0.08`, `n_flow = 0.55`, `eps_r = 220.0`, `sigma = 100.0`, `chi_mag = 0.00012`, `gamma_chiral = -0.5`, and `poloidal_gain = 2.0`.
  * Updated `MagnetParams` defaults to `theta_tilt_rad = 0.45`, `V_drive = 96.0`, and `f_drive = 0.0` (DC).
* **Targeted Transition Sweep Grid (`data/sweep_params.json`):**
  * Replaced static steady-state sweeps with a 48-run transition grid bracketing `V_drive` (`12.0`, `48.0`, `96.0` V), `tau_y` (`0.01`, `0.04`), `K_visc` (`0.04`, `0.08`), `t_on_steps` (`40`, `100`), and `t_off_steps` (`80`, `160`).

### Preserved / Unchanged
* Maintained exact `M6_trace_abs = 0.0` trace-free invariant and `coherence_metric` (`100.0`) / `lambda_m4` (`1.99`) saturation bounds.
* Kept `theta_tilt_rad -> poloidal rollover` coupling shelved to preserve clean, decoupled parameter behavior.

---

## Archive & Reference Log
* **Repository:** `nzink777/T7_M4_chiral_emulator` (Repo ID: `1470874613`)
* **Zenodo DOI:** `10.5281/zenodo.24518042` (Latest release badge target: `10.5281/zenodo.24518041`)
* **Prior Sweep Milestones:**
  * **864-Run Baseline Sweep (`sweep_summary (1).json`):** Identified Run #262 (`sigma = 0.25`, `V_drive = 24.0` DC, `induced_B_torus_nT = 4.326 nT`, `max_spin_rpm = 65.66`).
  * **9-Run Voltage/Conductivity Scaling Sweep (`sweep_summary (2).json`):** Demonstrated $4\times$ spin scaling per voltage doubling and saturation at `V_drive = 96.0` (`coherence_metric = 100.0`, `lambda_m4 = 1.99`).
  * **16-Run Coupled Density/Conductivity Sweep (`sweep_summary (3).json`):** Compared `rho = 1015.0` vs. `rho = 6400.0` (liquid-metal density), establishing Run #7 (`1050.63 RPM`, `110,751.91 nT`) as peak overall performer and Run #15 (`166.64 RPM`, `17,566.71 nT`) as peak high-density benchmark.