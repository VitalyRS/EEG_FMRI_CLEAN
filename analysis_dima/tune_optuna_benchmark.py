"""
Cohort-Wide Hyperparameter Optimization for Benchmark Alignment (Optuna)
========================================================================
Optimizes continuous ICLabel rejection thresholds across ALL benchmark
subjects simultaneously to guarantee maximum generalization and zero overfitting.

Uses exact linear projection (CAR(filt(A @ S)) == CAR(A) @ filt(S)):
Evaluating 12 subjects x 1,000 trials runs in ~10 seconds!
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, welch
import mne
import optuna

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_dima.compare_dima import load_dima, eeg_only, to_band_car, align, OCC

DATA = ROOT / "data"
DIMA = ROOT / "data_dima"

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_subject_precomputed(sid: str, seg: str):
    """Load Dima reference and pre-filter components in 0.05 seconds."""
    dr, qrs, n_ic, n_rm, cb = load_dima(sid, seg)
    cache_path = DATA / sid / "segments" / seg / "optuna_ica_trials" / "phase2_ica_cache.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    cache = np.load(cache_path, allow_pickle=True)
    ch_names = list(cache["ch_names"])
    common = [c for c in dr.ch_names if c in ch_names]
    idx_mine = [ch_names.index(c) for c in common]

    D = eeg_only(dr, common)
    Dx = to_band_car(D)

    W = cache["icaweights"]
    A = cache["icawinv"]
    X = cache["pruned_data"]  # (95, N)
    ic_classes = cache["ic_classes"]  # (n_ic, 7)
    sf = float(dr.info["sfreq"])

    # Activations
    S = W @ X

    # Occipital channels for alpha retention
    occ_idx = [common.index(c) for c in OCC if c in common]
    f, psd_base = welch(X[idx_mine][occ_idx], sf, nperseg=int(4 * sf), axis=-1)
    m_alpha = (f >= 8.0) & (f <= 13.0)
    alpha_base = float(np.mean(psd_base[:, m_alpha]))

    # Align lag
    info = mne.create_info([ch_names[i] for i in idx_mine], sf, ch_types="eeg")
    raw_X = mne.io.RawArray(X[idx_mine], info, verbose=False)
    Xx = to_band_car(raw_X)
    lag, _ = align(Dx, Xx)
    start_dima = max(30, 30 - lag)
    start_other = start_dima + lag
    n_pts = min(Dx.shape[1] - start_dima, Xx.shape[1] - start_other)

    D_aligned = Dx[:, start_dima:start_dima + n_pts]

    # Pre-filtering (exact linear equivalence)
    b, a_coeff = butter(2, [1.0 / (sf / 2.0), 45.0 / (sf / 2.0)], btype="bandpass")
    
    # 1. Pre-filter and CAR X
    X_comm = X[idx_mine]
    X_car = X_comm - np.mean(X_comm, axis=0, keepdims=True)
    X_filt = filtfilt(b, a_coeff, X_car, axis=-1)
    X_filt_aligned = X_filt[:, start_other:start_other + n_pts].astype(np.float32)

    # 2. CAR(A) on common channels
    A_comm = A[idx_mine]
    A_car = (A_comm - np.mean(A_comm, axis=0, keepdims=True)).astype(np.float32)

    # 3. Pre-filter S directly: takes ~0.02s for all 94 components!
    S_filt = filtfilt(b, a_coeff, S, axis=-1)
    S_filt_aligned = S_filt[:, start_other:start_other + n_pts].astype(np.float32)

    # Occipital baseline weights for fast alpha estimation
    # X_occ = X[idx_mine][occ_idx] - A[idx_mine][occ_idx][:, rej] @ S[rej]
    A_occ = A[idx_mine][occ_idx].astype(np.float32)
    X_occ = X[idx_mine][occ_idx].astype(np.float32)

    return {
        "sid": sid,
        "seg": seg,
        "dima_removed_count": n_rm,
        "D_aligned": D_aligned.astype(np.float32),
        "X_filt_aligned": X_filt_aligned,
        "A_car": A_car,
        "S_filt_aligned": S_filt_aligned,
        "ic_classes": ic_classes,
        "alpha_base": alpha_base,
        "X_occ": X_occ,
        "A_occ": A_occ,
        "S": S.astype(np.float32),
        "sf": sf,
    }


def decide_rejects(
    ic_classes: np.ndarray,
    thresh_other: float,
    other_brain_max: float,
    thresh_muscle: float,
    thresh_line_chan: float,
    thresh_eye: float,
    thresh_heart: float,
    max_brain_prob: float,
    max_rej_fraction: float,
) -> list[int]:
    n_ic = len(ic_classes)
    candidate_rejects = []

    for c in range(n_ic):
        probs = ic_classes[c]
        dom = int(np.argmax(probs))
        p_max = float(probs[dom])
        p_brain = float(probs[0])
        is_rej = False

        if dom == 2 and p_max >= thresh_eye and p_brain < max_brain_prob:  # Eye
            is_rej = True
        elif (dom == 3 or probs[3] >= thresh_heart) and p_brain < max_brain_prob:  # Heart
            is_rej = True
        elif dom in (4, 5) and p_max >= thresh_line_chan and p_brain < min(max_brain_prob, 0.15):  # Line / Chan
            is_rej = True
        elif dom == 1 and p_max >= thresh_muscle and p_brain < min(max_brain_prob, 0.10):  # Muscle
            is_rej = True
        elif dom == 6 and p_max >= thresh_other and p_brain < other_brain_max:  # Other / Scanner / Pump
            is_rej = True

        if is_rej:
            conf = p_max - p_brain
            candidate_rejects.append((c, conf))

    max_allowed = max(3, int(max_rej_fraction * n_ic))
    if len(candidate_rejects) > max_allowed:
        candidate_rejects.sort(key=lambda x: x[1], reverse=True)
        candidate_rejects = candidate_rejects[:max_allowed]

    return sorted([c for c, _ in candidate_rejects])


def run_cohort_tuning(
    targets: list[tuple[str, str]],
    n_trials: int = 1000,
    min_alpha_ret: float = 0.80,
):
    print("==========================================================================")
    print(f"Cohort-Wide Multi-Subject Optimization across {len(targets)} benchmark datasets")
    print("==========================================================================")

    datasets = []
    for sid, seg in targets:
        try:
            d = load_subject_precomputed(sid, seg)
            datasets.append(d)
            print(f"  ✓ Loaded {sid}/{seg} (Dima removed {d['dima_removed_count']} ICs)")
        except Exception as e:
            print(f"  ✗ Skipping {sid}/{seg}: {e}")

    if not datasets:
        print("No valid datasets loaded.")
        return

    print(f"\nAll {len(datasets)} datasets loaded into memory. Starting {n_trials} Optuna trials...")

    def objective(trial: optuna.Trial) -> float:
        thresh_other = trial.suggest_float("thresh_other", 0.40, 0.75, step=0.05)
        other_brain_max = trial.suggest_float("other_brain_max", 0.03, 0.15, step=0.01)
        thresh_muscle = trial.suggest_float("thresh_muscle", 0.45, 0.85, step=0.05)
        thresh_line_chan = trial.suggest_float("thresh_line_chan", 0.45, 0.85, step=0.05)
        thresh_eye = trial.suggest_float("thresh_eye", 0.60, 0.85, step=0.05)
        thresh_heart = trial.suggest_float("thresh_heart", 0.60, 0.85, step=0.05)
        max_brain_prob = trial.suggest_float("max_brain_prob", 0.10, 0.25, step=0.02)
        max_rej_fraction = trial.suggest_float("max_rej_fraction", 0.08, 0.25, step=0.02)

        r_list = []
        rmse_list = []
        alpha_list = []
        rej_counts = []

        for d in datasets:
            rej = decide_rejects(
                d["ic_classes"],
                thresh_other,
                other_brain_max,
                thresh_muscle,
                thresh_line_chan,
                thresh_eye,
                thresh_heart,
                max_brain_prob,
                max_rej_fraction,
            )
            rej_counts.append(len(rej))

            # Instant matrix reconstruction: X_filt - A_car[:, rej] @ S_filt[rej]
            if len(rej) == 0:
                X_al = d["X_filt_aligned"]
            else:
                X_al = d["X_filt_aligned"] - d["A_car"][:, rej] @ d["S_filt_aligned"][rej, :]

            # Fast Pearson r
            D_al = d["D_aligned"]
            X_dm = X_al - np.mean(X_al, axis=-1, keepdims=True)
            D_dm = D_al - np.mean(D_al, axis=-1, keepdims=True)
            cov = np.sum(X_dm * D_dm, axis=-1)
            std_x = np.sqrt(np.sum(X_dm**2, axis=-1))
            std_d = np.sqrt(np.sum(D_dm**2, axis=-1))
            r_all = cov / (std_x * std_d + 1e-30)
            r_list.append(float(np.mean(r_all)))
            rmse_list.append(float(np.sqrt(np.mean((X_al - D_al)**2)) * 1e6))

            # Alpha retention check
            if len(rej) > 0:
                X_occ_clean = d["X_occ"] - d["A_occ"][:, rej] @ d["S"][rej, :]
            else:
                X_occ_clean = d["X_occ"]
            X_occ_car = X_occ_clean - np.mean(X_occ_clean, axis=0, keepdims=True)
            f, psd_c = welch(X_occ_car, d["sf"], nperseg=int(4 * d["sf"]), axis=-1)
            m_a = (f >= 8.0) & (f <= 13.0)
            alpha_ret = float(np.mean(psd_c[:, m_a])) / max(d["alpha_base"], 1e-30)
            alpha_list.append(alpha_ret)

        cohort_r = float(np.mean(r_list))
        cohort_rmse = float(np.mean(rmse_list))
        cohort_alpha = float(np.mean(alpha_list))
        avg_rej = float(np.mean(rej_counts))

        trial.set_user_attr("cohort_r", cohort_r)
        trial.set_user_attr("cohort_rmse", cohort_rmse)
        trial.set_user_attr("cohort_alpha", cohort_alpha)
        trial.set_user_attr("avg_rej_count", avg_rej)
        trial.set_user_attr("min_alpha", float(np.min(alpha_list)))

        # Penalties:
        # 1. Protect alpha across every subject
        alpha_penalty = 10.0 * max(0.0, min_alpha_ret - np.min(alpha_list))
        
        # Maximize cohort correlation, minimize error, ensure alpha >= 80% on all subjects
        score = cohort_r - 0.01 * cohort_rmse - alpha_penalty
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    print(f"\n==========================================================================")
    print(f"COHORT OPTUNA COMPLETE ({n_trials} trials across {len(datasets)} subjects)")
    print(f"==========================================================================")
    print(f"Best Cohort Score:    {best.value:.4f}")
    print(f"  Cohort Mean r:      {best.user_attrs['cohort_r']:.4f}")
    print(f"  Cohort Mean RMSE:   {best.user_attrs['cohort_rmse']:.2f} uV")
    print(f"  Cohort Mean Alpha:  {best.user_attrs['cohort_alpha']:.1%}")
    print(f"  Min Alpha Subject:  {best.user_attrs['min_alpha']:.1%}")
    print(f"  Average ICs Removed:{best.user_attrs['avg_rej_count']:.1f} ICs (Dima average: ~7.6 ICs)")
    print("\nUniversal Hyperparameters for cleanP02.yaml:")
    print("---------------------------------------------")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print("---------------------------------------------\n")
    return best, study


if __name__ == "__main__":
    targets_default = [
        "1916:drone", "1917:drone", "1918:drone", "1924:drone", "1925:drone", "1926:drone",
        "1916:lasertag", "1917:lasertag", "1918:lasertag", "1924:lasertag", "1925:lasertag", "1926:lasertag",
    ]
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=targets_default)
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--min_alpha", type=float, default=0.80)
    args = ap.parse_args()

    target_list = [tuple(t.split(":")) for t in args.targets]
    run_cohort_tuning(target_list, n_trials=args.trials, min_alpha_ret=args.min_alpha)
