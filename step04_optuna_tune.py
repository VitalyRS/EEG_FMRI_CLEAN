"""
STEP 04: Bayesian Optimization of Bergen AAS Parameters via Optuna TPE (Fast 1-Channel)
========================================================================================
Extracts 1 target channel (Fz, ~15 MB in RAM) for the working segment interval,
loads slice triggers directly from slice_triggers.txt, and executes fast Optuna trials.
Zero duplicate multi-channel files created on disk.
"""
from pathlib import Path
import subprocess
import gc
import json
import numpy as np
import mne
from scipy.io import savemat
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

try:
    from .config import (MATLAB_BIN, EEGLAB_DIR, BERGEN_DIR, PROJECT_ROOT,
                         DEFAULT_SEGMENT_DIR, DEFAULT_TARGET_CH, DEFAULT_N_TRIALS,
                         GRADIENT_HARMONICS, ALPHA_BAND, BG_BANDS)
except ImportError:
    from config import (MATLAB_BIN, EEGLAB_DIR, BERGEN_DIR, PROJECT_ROOT,
                         DEFAULT_SEGMENT_DIR, DEFAULT_TARGET_CH, DEFAULT_N_TRIALS,
                         GRADIENT_HARMONICS, ALPHA_BAND, BG_BANDS)


# The slice-artifact comb lands on exact multiples of the slice-repetition
# frequency (sfreq / nominal_slice_samples = 10.0 Hz for this sequence). At
# those bins physiological signal and artifact are mathematically inseparable,
# so AAS necessarily zeroes them. They must be EXCLUDED from any alpha metric,
# otherwise the dead 10 Hz bin drags the measured peak and preservation down.
ALPHA_COMB_NOTCH_HZ = 0.15
ALPHA_RETENTION_TARGET = 0.90  # keep >= 90% of off-comb alpha shoulders


def _comb_notch_mask(f: np.ndarray, comb_hz: float, half_width: float = ALPHA_COMB_NOTCH_HZ) -> np.ndarray:
    """True where f sits ON a comb harmonic (to be excluded)."""
    if comb_hz <= 0:
        return np.zeros_like(f, dtype=bool)
    nearest_harm = np.round(f / comb_hz) * comb_hz
    return (np.abs(f - nearest_harm) <= half_width) & (nearest_harm > 0)


def compute_alpha_and_gradient_metrics(data_clean: np.ndarray, data_raw: np.ndarray,
                                       sfreq: float, comb_hz: float = 10.0):
    nperseg = int(4 * sfreq)
    f, psd_clean = welch(data_clean, sfreq, nperseg=nperseg, noverlap=nperseg // 2)
    _, psd_raw   = welch(data_raw, sfreq, nperseg=nperseg, noverlap=nperseg // 2)

    # 1. Gradient suppression across non-alpha harmonics (20, 30, 40, 50, 60 Hz)
    p_raw_harm = []
    p_clean_harm = []
    for hf in GRADIENT_HARMONICS:
        if hf > f[-1]:
            continue
        idx = np.argmin(np.abs(f - hf))
        p_raw_harm.append(psd_raw[idx])
        p_clean_harm.append(psd_clean[idx])

    mean_raw_harm = float(np.mean(p_raw_harm)) if p_raw_harm else 1.0
    mean_clean_harm = float(np.mean(p_clean_harm)) if p_clean_harm else 1.0
    g_suppression = 1.0 - (mean_clean_harm / max(mean_raw_harm, 1e-12))
    g_suppression = float(np.clip(g_suppression, 0.0, 1.0))

    # Exclude the dead comb bin(s) (10 Hz) from every alpha computation
    on_comb = _comb_notch_mask(f, comb_hz)
    m_alpha = (f >= ALPHA_BAND[0]) & (f <= ALPHA_BAND[1])
    m_alpha_off = m_alpha & ~on_comb   # alpha "shoulders" only
    m_bg1   = (f >= BG_BANDS[0][0]) & (f <= BG_BANDS[0][1])
    m_bg2   = (f >= BG_BANDS[1][0]) & (f <= BG_BANDS[1][1])

    # 2. Alpha Prominence in Clean, measured on off-comb shoulders only
    p_alpha = float(np.mean(psd_clean[m_alpha_off])) if np.any(m_alpha_off) else 1e-6
    p_bg    = float(0.5 * (np.mean(psd_clean[m_bg1]) + np.mean(psd_clean[m_bg2]))) if (np.any(m_bg1) and np.any(m_bg2)) else 1e-6
    alpha_prominence = float(p_alpha / max(p_bg, 1e-12))

    # 3. Alpha Peak Frequency (off-comb, so it can't be pinned to 10 Hz)
    if np.any(m_alpha_off):
        alpha_f = f[m_alpha_off]
        alpha_psd = psd_clean[m_alpha_off]
        alpha_peak_freq = float(alpha_f[np.argmax(alpha_psd)])
    else:
        alpha_peak_freq = 10.0

    # 4. Alpha retention: how much of the off-comb alpha shoulders survived
    # cleaning relative to raw. This is the physically meaningful "did we eat
    # the real alpha?" signal — over-aggressive AAS (small win_k) drives it
    # toward 0, faithful cleaning keeps it near 1.0.
    p_alpha_raw_off = float(np.mean(psd_raw[m_alpha_off])) if np.any(m_alpha_off) else 1e-6
    alpha_retention = float(p_alpha / max(p_alpha_raw_off, 1e-12))
    alpha_retention = float(np.clip(alpha_retention, 0.0, 1.5))

    # 5. Multi-objective composite score to MINIMIZE:
    #   - drive gradient suppression -> 1.0  (small when supp is high)
    #   - protect physiological alpha: penalize retention below target
    grad_term = (1.0 - g_suppression) * 100.0
    alpha_penalty = max(0.0, ALPHA_RETENTION_TARGET - alpha_retention) ** 2
    loss = float(grad_term + 200.0 * alpha_penalty)

    return {
        "loss": loss,
        "g_suppression": g_suppression,
        "alpha_prominence": alpha_prominence,
        "alpha_peak_freq": alpha_peak_freq,
        "alpha_retention": alpha_retention,
        "p_alpha": p_alpha,
        "p_bg": p_bg
    }


def plot_trial_spectrum(data_clean: np.ndarray, data_raw: np.ndarray, sfreq: float,
                        comb_hz: float, res: dict, params: dict, out_png: Path):
    """Per-trial PSD (raw vs clean) with gradient harmonics, alpha band and the
    slice comb marked, so trials can be compared visually side by side."""
    nperseg = int(4 * sfreq)
    f, psd_raw = welch(data_raw, sfreq, nperseg=nperseg, noverlap=nperseg // 2)
    _, psd_clean = welch(data_clean, sfreq, nperseg=nperseg, noverlap=nperseg // 2)
    m = f <= 70.0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(f[m], psd_raw[m], color="#d63031", lw=0.8, alpha=0.65, label="Raw (inside MRI)")
    ax.semilogy(f[m], psd_clean[m], color="#00b894", lw=1.1, label="Clean (Bergen AAS)")
    ax.axvspan(ALPHA_BAND[0], ALPHA_BAND[1], color="#0984e3", alpha=0.08, label="Alpha 8-13 Hz")
    for hf in GRADIENT_HARMONICS:
        if hf <= 70.0:
            ax.axvline(hf, color="#636e72", ls=":", lw=0.7)
    ax.axvline(comb_hz, color="#e17055", ls="--", lw=0.9, label=f"Slice comb {comb_hz:.1f} Hz")
    ax.set_xlim(0, 70)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (uV^2/Hz)")
    ax.set_title(
        f"Trial {params['trial']} | shift={params['shift']:+d}, k={params['win_k']}, "
        f"thresh={params['motion_thresh']:.2f}\n"
        f"GradSupp={res['g_suppression']*100:.2f}%  "
        f"AlphaRet={res['alpha_retention']*100:.1f}%  "
        f"AlphaPeak={res['alpha_peak_freq']:.1f}Hz  Loss={res['loss']:.3f}"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run_single_channel_matlab(segment_dir: Path, mat_1ch_path: Path, triggers_path: Path,
                              rp_path: Path | None, shift: int, win_k: int,
                              motion_thresh: float, slices_per_volume: int = 25,
                              nominal_slice_samples: int = 500,
                              out_dir: Path | None = None) -> np.ndarray | None:
    shift_str = f"p{shift}" if shift >= 0 else f"m{-shift}"
    tag = f"opt_trial_sh_{shift_str}_k{win_k}"
    # When out_dir is given (per-trial persistence) keep the executed .m script
    # and the cleaned .mat there for later inspection; otherwise fall back to
    # scratch files in segment_dir that are deleted right after loading.
    keep = out_dir is not None
    work_dir = out_dir if keep else segment_dir
    out_mat = work_dir / ("clean_1ch.mat" if keep else f"{tag}_clean.mat")
    m_file = work_dir / ("bergen_clean.m" if keep else f"{tag}.m")

    helper_dir = Path(__file__).parent.resolve()

    rp_code = ""
    if rp_path and rp_path.exists():
        rp_code = f"""
        [motiondata, W_vol] = m_rp_info('{rp_path.resolve()}', n_slices / {slices_per_volume}, {motion_thresh}, {win_k});
        W = kron(W_vol, eye({slices_per_volume}));
        clear W_vol motiondata;
        """
    else:
        rp_code = f"""
        W_vol = m_moving_average(n_slices / {slices_per_volume}, {win_k});
        W = kron(W_vol, eye({slices_per_volume}));
        clear W_vol;
        """

    lines = [
        f"addpath('{EEGLAB_DIR.resolve()}');",
        f"addpath('{BERGEN_DIR.resolve()}');",
        f"addpath('{helper_dir}');",
        f"addpath('{PROJECT_ROOT.resolve()}');",
        f"load('{mat_1ch_path.resolve()}', 'data', 'srate', 'pnts');",
        "EEG.data = data;",
        "EEG.pnts = pnts;",
        "EEG.nbchan = 1;",
        "EEG.srate = srate;",
        f"Peak_slices = load('{triggers_path.resolve()}');",
        "Peak_slices = Peak_slices(:)';",
    ]
    lines += [f"TR_sl = {nominal_slice_samples};"]  # fixed nominal period, never derived from clipped triggers
    if shift != 0:
        # Only shift. Do NOT clip to [1,pnts] (that shrank TR_sl and left a
        # periodic uncorrected gap) and do NOT drop slices (that breaks the
        # n_slices = n_vols*spv invariant kron() needs). bergen_fast_correction
        # skips the 1-2 out-of-bounds boundary slices internally via its own
        # valid_slices mask, so the full trigger count is preserved for W.
        lines += [f"Peak_slices = Peak_slices + ({shift});"]
    lines += [
        "n_slices = length(Peak_slices);",
        rp_code,
        "onset_val  = 0;",
        "offset_val = onset_val + TR_sl - 1;",
        "EEG = bergen_fast_correction(EEG, W, Peak_slices, onset_val, offset_val);",
        "clear W;",
        "clean_data = EEG.data;",
        f"save('{out_mat.resolve()}', 'clean_data', '-v7');",
        "clear EEG clean_data;",
        "exit(0);",
    ]

    with open(m_file, "w", encoding="ascii") as fh:
        fh.write("\n".join(lines) + "\n")

    res = subprocess.run(
        [str(MATLAB_BIN), "-nodesktop", "-nosplash", "-batch", f"run('{m_file.resolve()}')"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if not keep:
        m_file.unlink(missing_ok=True)

    if res.returncode == 0 and out_mat.exists():
        from scipy.io import loadmat
        try:
            m_res = loadmat(out_mat)
            clean_arr = m_res["clean_data"].flatten()
            if not keep:
                out_mat.unlink(missing_ok=True)
            return clean_arr
        except Exception:
            if not keep:
                out_mat.unlink(missing_ok=True)
            return None
    return None


def run_optuna_tuning(segment_dir: Path = DEFAULT_SEGMENT_DIR,
                      n_trials: int = DEFAULT_N_TRIALS,
                      target_ch: str = DEFAULT_TARGET_CH):
    segment_dir = Path(segment_dir).resolve()
    print("=" * 70)
    print(f"[STEP 04] Optuna TPE Optimization (Lean 1-Channel) for: {segment_dir.name}")
    print("=" * 70)

    work_info_path = segment_dir / "segment_work_info.json"
    if not work_info_path.exists():
        raise FileNotFoundError(f"Missing {work_info_path}. Run step03 first!")

    with open(work_info_path, "r", encoding="utf-8") as f:
        work_info = json.load(f)

    raw_vhdr = Path(work_info["raw_vhdr"]).resolve()
    t_start = float(work_info["t_work_start_sec"])
    t_stop  = float(work_info["t_work_stop_sec"])
    slices_per_volume = int(work_info.get("slices_per_volume", 25))
    rp_file_str = work_info.get("rp_file")
    rp_path = Path(rp_file_str).resolve() if rp_file_str else None
    triggers_path = segment_dir / "slice_triggers.txt"

    # Extract 1 channel in memory
    print(f"  Extracting target channel '{target_ch}' interval [{t_start:.2f}s .. {t_stop:.2f}s]...")
    raw = mne.io.read_raw_brainvision(raw_vhdr, preload=False, verbose=False)
    sfreq = float(raw.info["sfreq"])
    ch_pick = target_ch if target_ch in raw.ch_names else raw.ch_names[0]
    raw_crop = raw.copy().crop(tmin=t_start, tmax=min(raw.times[-1], t_stop)).pick([ch_pick])
    data_1ch = raw_crop.get_data(units="uV")[0].astype(np.float64)
    del raw, raw_crop; gc.collect()

    mat_1ch_path = segment_dir / "temp_1ch_target.mat"
    savemat(mat_1ch_path, {"data": data_1ch, "srate": sfreq, "pnts": len(data_1ch)})
    print(f"  Cached 1-channel MAT ({len(data_1ch)} samples, {mat_1ch_path.stat().st_size / (1024*1024):.2f} MB)")

    db_path = segment_dir / "optuna_study.db"
    storage_url = f"sqlite:///{db_path.resolve()}"
    # Version tag bumped when the search space or loss definition changes, so
    # stale trials from the old win_k range / old loss are never mixed in.
    # v4: the TR_sl / trigger-shift bug fix changed the artifact-suppression
    # landscape, so the v3 trials (computed on the buggy MATLAB code) are stale
    # and must not be reused -> force a fresh 20-trial search on the fixed code.
    study_name = f"bergen_{segment_dir.name}_v4_trslfix"

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True
    )

    # Slice-comb fundamental for this sequence (10.0 Hz here). Bins on this
    # comb are unrecoverable and excluded from the alpha metrics.
    nominal_slice_samples = int(work_info.get("nominal_slice_samples", 500))
    comb_hz = sfreq / nominal_slice_samples

    def objective(trial: optuna.Trial) -> float:
        shift = trial.suggest_int("shift", -6, 6)
        # This subject has head motion, so the gradient artifact is NON-stationary.
        # A SMALL template window (few neighbouring volumes) tracks the changing
        # artifact shape and suppresses the 20/30/40 Hz comb far better than a wide
        # window, which averages over epochs where the artifact already drifted and
        # leaves a large flat residual. Empirically k~4 clears harmonics to <4 uV^2,
        # while k=39 leaves ~85. Range kept tight around the proven optimum.
        win_k = trial.suggest_int("win_k", 3, 14)
        motion_thresh = trial.suggest_float("motion_thresh", 0.20, 1.50, step=0.10)

        # Each trial gets its own subfolder (segment4/trial0, trial1, ...),
        # named by Optuna's 0-based trial index so it maps 1:1 to best_trial.
        trial_dir = segment_dir / f"trial{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        data_c = run_single_channel_matlab(segment_dir, mat_1ch_path, triggers_path,
                                           rp_path, shift, win_k, motion_thresh, slices_per_volume,
                                           nominal_slice_samples=nominal_slice_samples,
                                           out_dir=trial_dir)
        if data_c is None:
            raise optuna.exceptions.TrialPruned()

        res = compute_alpha_and_gradient_metrics(data_c, data_1ch, sfreq, comb_hz=comb_hz)

        # Persist this trial's parameters, metrics and a comparison spectrum.
        params = {"trial": trial.number, "shift": shift, "win_k": win_k,
                  "motion_thresh": motion_thresh}
        with open(trial_dir / "params.json", "w", encoding="utf-8") as fh:
            json.dump(params, fh, indent=2)
        with open(trial_dir / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        try:
            plot_trial_spectrum(data_c, data_1ch, sfreq, comb_hz, res, params,
                                trial_dir / "spectrum.png")
        except Exception as e:
            print(f"    [warn] trial {trial.number} spectrum plot failed: {e}")

        del data_c; gc.collect()

        print(f"  Trial {trial.number:3d} | shift={shift:+d}, k={win_k:2d}, thresh={motion_thresh:.2f} -> GradSupp={res['g_suppression']*100:.2f}%, AlphaRet={res['alpha_retention']*100:.1f}%, AlphaProm={res['alpha_prominence']:.2f}, AlphaPeak={res['alpha_peak_freq']:.1f}Hz (Loss={res['loss']:.4f}) [{trial_dir.name}/]")
        return res["loss"]

    remaining_trials = max(0, n_trials - len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]))
    if remaining_trials > 0:
        print(f"Running {remaining_trials} Optuna trials...")
        study.optimize(objective, n_trials=remaining_trials)

    # Cleanup temporary 1-ch MAT
    mat_1ch_path.unlink(missing_ok=True)

    print("\n" + "=" * 70)
    print("OPTUNA OPTIMIZATION COMPLETE!")
    print(f"Best Trial #{study.best_trial.number}:")
    print(f"  shift         = {study.best_params['shift']} samples")
    print(f"  win_k         = {study.best_params['win_k']} volumes")
    print(f"  motion_thresh = {study.best_params['motion_thresh']:.2f} mm")
    print(f"  Best Loss     = {study.best_value:.4f}")
    best_trial_dir = segment_dir / f"trial{study.best_trial.number}"
    print(f"  Artifacts     = {best_trial_dir}/  (params/metrics/spectrum/clean_1ch)")
    print("=" * 70)

    # Save best parameters to JSON
    best_params_json = segment_dir / "optuna_best_params.json"
    with open(best_params_json, "w", encoding="utf-8") as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_trial_dir": str(best_trial_dir),
            "best_params": study.best_params,
            "best_value": study.best_value
        }, f, indent=2)

    # Plot optimization summary
    plot_optuna_summary(study, segment_dir)
    return study.best_params


def plot_optuna_summary(study: optuna.Study, segment_dir: Path):
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        return
    shifts  = [t.params["shift"] for t in trials]
    ks      = [t.params["win_k"] for t in trials]
    threshs = [t.params["motion_thresh"] for t in trials]
    scores  = [t.value for t in trials]
    idx     = list(range(len(trials)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Optuna TPE Bergen Optimization Summary\n"
        f"Best: shift={study.best_params['shift']:+d} | "
        f"k={study.best_params['win_k']} | "
        f"thresh={study.best_params['motion_thresh']:.2f} | "
        f"AvgSpike={study.best_value:.4f}",
        fontsize=12, fontweight="bold"
    )

    axes[0, 0].plot(idx, scores, "o-", lw=1, ms=4, color="#636e72", alpha=0.6, label="Trial")
    axes[0, 0].plot(idx, np.minimum.accumulate(scores), lw=2.5, color="#00b894", label="Best so far")
    axes[0, 0].set_xlabel("Trial"); axes[0, 0].set_ylabel("Avg Spike Ratio (10-40 Hz)")
    axes[0, 0].set_title("Optimization Progress"); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(shifts, scores, c=scores, cmap="RdYlGn_r", s=60, alpha=0.8, edgecolors="k", lw=0.3)
    axes[0, 1].axvline(study.best_params["shift"], color="green", lw=2, ls="--")
    axes[0, 1].set_xlabel("shift (samples)"); axes[0, 1].set_ylabel("Avg Spike Ratio")
    axes[0, 1].set_title("Shift vs Score"); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].scatter(ks, scores, c=scores, cmap="RdYlGn_r", s=60, alpha=0.8, edgecolors="k", lw=0.3)
    axes[1, 0].axvline(study.best_params["win_k"], color="green", lw=2, ls="--")
    axes[1, 0].set_xlabel("win_k (volumes)"); axes[1, 0].set_ylabel("Avg Spike Ratio")
    axes[1, 0].set_title("Window k vs Score"); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(threshs, scores, c=scores, cmap="RdYlGn_r", s=60, alpha=0.8, edgecolors="k", lw=0.3)
    axes[1, 1].axvline(study.best_params["motion_thresh"], color="green", lw=2, ls="--")
    axes[1, 1].set_xlabel("motion_thresh (mm FD)"); axes[1, 1].set_ylabel("Avg Spike Ratio")
    axes[1, 1].set_title("Motion Threshold vs Score"); axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_png = segment_dir / "optuna_result.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Optuna plot: {out_png.name}")


if __name__ == "__main__":
    run_optuna_tuning()
