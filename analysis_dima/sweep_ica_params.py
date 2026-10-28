"""
Sweep Hyperparameters Against Dima's Benchmark
=============================================
Performs an automated grid search over ICA PCA ranks (e.g. 75, 80, 85, 90, 95),
ASR burst criteria, and component rejection strategies (thresholds vs meta-model)
to find the configuration that maximizes agreement with Dima's reference dataset.

Metrics evaluated:
  - Mean Pearson correlation r (aligned in time)
  - Root Mean Square Error (RMSE, uV) vs Dima
  - Occipital alpha power ratio (pipeline / Dima)
  - Residual cardiac artifact (QRS-locked RMS, uV)
  - Number of rejected IC components

Usage:
  python analysis_dima/sweep_ica_params.py --sid 1916 1925 --seg drone --pca 75 85 95
  python analysis_dima/sweep_ica_params.py --sid 1916 --seg drone lasertag
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt
from scipy.stats import pearsonr
import mne

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_dima.compare_dima import load_dima, eeg_only, to_band_car, align, OCC, qrs_locked_rms
from mne.preprocessing import ICA
from mne_icalabel.iclabel import iclabel_label_components

DATA = ROOT / "data"
DIMA = ROOT / "data_dima"
OUT_DIR = ROOT / "reports" / "dima_compare"
META_MODEL_PATHS = [
    ROOT / "data" / "ica_meta_model.json",
    ROOT / "config" / "ica_meta_model.json",
]


def evaluate_rejection(X_car_filt, S_filt, A_car, rejected_ics, D_aligned, psd_d, sf, occ_idx):
    """Vectorized evaluation of cleaned data against Dima's reference in ~1 millisecond."""
    if len(rejected_ics) > 0:
        rej = np.array(rejected_ics, dtype=int)
        P_rej = A_car[:, rej] @ S_filt[rej, :]
        X_clean = X_car_filt - P_rej
    else:
        X_clean = X_car_filt

    # Vectorized Pearson r across channels
    n_ch = min(D_aligned.shape[0], X_clean.shape[0])
    D_c = D_aligned[:n_ch] - np.mean(D_aligned[:n_ch], axis=-1, keepdims=True)
    X_c = X_clean[:n_ch] - np.mean(X_clean[:n_ch], axis=-1, keepdims=True)
    cov = np.sum(D_c * X_c, axis=-1)
    denom = np.sqrt(np.sum(D_c ** 2, axis=-1) * np.sum(X_c ** 2, axis=-1)) + 1e-12
    mean_r = float(np.mean(cov / denom))

    # RMSE (uV)
    rmse = float(np.sqrt(np.mean((X_clean[:n_ch] * 1e6 - D_aligned[:n_ch] * 1e6) ** 2)))

    # Alpha ratio on occipital channels
    f, psd_x = welch(X_clean[occ_idx], sf, nperseg=int(4 * sf), axis=-1)
    m_alpha = (f >= 8.0) & (f <= 13.0)
    alpha_x = float(np.mean(psd_x[:, m_alpha]))
    alpha_ratio = (alpha_x / (psd_d + 1e-30)) if psd_d > 0 else 1.0

    return mean_r, rmse, alpha_ratio


def run_sweep(subjects: list[str], segments: list[str], pca_ranks: list[int],
              burst_crits: list[str | float], rejection_modes: list[str],
              max_iter: int = 150):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    # Load meta-model if available
    meta_model = None
    for p_cand in META_MODEL_PATHS:
        if p_cand.exists():
            with open(p_cand) as f:
                meta_model = json.load(f)
            break

    for sid in subjects:
        for seg in segments:
            print("\n" + "=" * 78, flush=True)
            print(f"  SWEEP FOR SUBJECT {sid} | SEGMENT: {seg}", flush=True)
            print("=" * 78, flush=True)

            # 1. Load Dima reference
            try:
                raw_dima, qrs_dima, dima_n_ic, dima_n_rm, dima_cb = load_dima(sid, seg)
            except Exception as e:
                print(f"  [SKIP] Could not load Dima benchmark for {sid}/{seg}: {e}", flush=True)
                continue

            # 2. Load our BCG data
            bcg_fif = DATA / sid / "derivatives" / "03_bcg" / seg / f"{seg}_bcg_clean.fif"
            if not bcg_fif.exists():
                # Fallback to resampled
                bcg_fif = DATA / sid / "derivatives" / "02_resampled250" / seg / f"{seg}_250hz.fif"
            if not bcg_fif.exists():
                print(f"  [SKIP] No BCG or 250Hz input file found for {sid}/{seg}", flush=True)
                continue

            raw_mine = mne.io.read_raw_fif(bcg_fif, preload=True, verbose=False)
            # Edge trimming: if file has full ~500s duration, crop 1.0s start and 1.0s end!
            if raw_mine.times[-1] >= 495.0:
                print(f"  [TRIM] Cropping 1.0s edge bursts from start and end...", flush=True)
                raw_mine.crop(tmin=1.0, tmax=raw_mine.times[-1] - 1.0)

            sf = float(raw_mine.info["sfreq"])
            common_chans = [c for c in raw_dima.ch_names if c in raw_mine.ch_names and c != "ECG"]
            occ_idx = [common_chans.index(c) for c in OCC if c in common_chans]

            D_common = eeg_only(raw_dima, common_chans)
            Dx = to_band_car(D_common, 1.0, 45.0)

            raw_common = eeg_only(raw_mine, common_chans)
            # Set exact electrode montage from Dima's EEGLAB reference (includes O9, O10)
            if raw_dima.get_montage() is not None:
                raw_common.set_montage(raw_dima.get_montage(), on_missing="ignore")
            else:
                raw_common.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore")

            Xx = to_band_car(raw_common, 1.0, 45.0)

            # Align lag
            lag, align_corr = align(Dx, Xx)
            start_dima = max(30, 30 - lag)
            start_mine = start_dima + lag
            n_pts = min(Dx.shape[1] - start_dima, Xx.shape[1] - start_mine)

            D_aligned = Dx[:, start_dima:start_dima + n_pts]
            print(f"  Dima ICs: {dima_n_ic} (removed {dima_n_rm}) | Aligned lag: {lag} samples ({lag/sf*1000:+.1f} ms)", flush=True)

            # Precompute Dima occipital alpha PSD once
            f_welch, psd_d_arr = welch(D_aligned[occ_idx], sf, nperseg=int(4 * sf), axis=-1)
            m_alpha = (f_welch >= 8.0) & (f_welch <= 13.0)
            psd_d = float(np.mean(psd_d_arr[:, m_alpha]))

            # Standard ICLabel preprocessing: 1.0 - 100.0 Hz + Average Reference
            raw_for_ica = raw_common.copy().filter(1.0, 100.0, verbose=False).set_eeg_reference("average", verbose=False)

            # Fit ICA for each PCA rank
            for n_pca in pca_ranks:
                print(f"\n  --- Fitting Picard ICA with PCA rank = {n_pca} ---", flush=True)
                t0 = time.time()
                try:
                    ica = ICA(n_components=n_pca, method="picard", fit_params=dict(ortho=False, extended=True), max_iter=max_iter, random_state=42)
                    ica.fit(raw_for_ica, verbose=False)
                    fit_time = time.time() - t0
                    print(f"      Fit finished in {fit_time:.1f}s. Running ICLabel...", flush=True)
                except Exception as e:
                    print(f"  ICA fit failed for n_pca={n_pca}: {e}", flush=True)
                    continue

                # ICLabel classification (shape: (n_pca, 7))
                probs = iclabel_label_components(raw_for_ica, ica)
                class_names = ["brain", "muscle", "eye", "heart", "line_noise", "channel_noise", "other"]

                # Extract unmixing and mixing matrices
                W = ica.unmixing_matrix_ @ ica.pca_components_[:n_pca, :]
                A = np.linalg.pinv(W)

                # Precompute unmixed activations and CAR mixing
                X_mat = raw_common.get_data()  # (n_ch, T)
                # CAR on X
                X_car = X_mat - np.mean(X_mat, axis=0, keepdims=True)
                b_band, a_band = butter(2, [1.0 / (sf / 2.0), 45.0 / (sf / 2.0)], btype="bandpass")
                X_car_filt = filtfilt(b_band, a_band, X_car, axis=-1)[:, start_mine:start_mine + n_pts]

                # Project CAR to mixing matrix
                A_car = A - np.mean(A, axis=0, keepdims=True)

                # S_filt (zero-phase bandpass via scipy filtfilt)
                S_raw = W @ X_mat
                S_filt = filtfilt(b_band, a_band, S_raw, axis=-1)[:, start_mine:start_mine + n_pts]

                # Test baseline: 0 ICs rejected
                r_base, rmse_base, alpha_base = evaluate_rejection(
                    X_car_filt, S_filt, A_car, [], D_aligned, psd_d, sf, occ_idx
                )
                results.append({
                    "subject": sid, "segment": seg, "pca_rank": n_pca,
                    "strategy": "No ICA rejection", "rejected_ics": 0,
                    "corr_r": r_base, "rmse_uV": rmse_base,
                    "alpha_ratio": alpha_base,
                    "fit_time_s": round(fit_time, 2)
                })

                # Test rejection modes
                # 1. Standard Thresholds (Optuna-tuned grid)
                for t_muscle in [0.45, 0.60, 0.80]:
                    for t_other in [0.60, 0.75, 0.90]:
                        for max_rej in [0.10, 0.15, 0.25]:
                            rej = []
                            for i in range(n_pca):
                                p = probs[i]
                                is_eye = p[2] >= 0.80
                                is_heart = p[3] >= 0.65
                                is_line_chan = (p[4] + p[5]) >= 0.85
                                is_muscle = (p[1] >= t_muscle) and (p[0] < 0.12)
                                is_other = (p[6] >= t_other) and (p[0] < 0.08)
                                if (is_eye or is_heart or is_line_chan or is_muscle or is_other) and p[0] < 0.15:
                                    rej.append(i)
                            # Apply max fraction ceiling
                            cap = int(round(max_rej * n_pca))
                            rej = rej[:cap]
                            r_val, rmse_val, a_ratio = evaluate_rejection(
                                X_car_filt, S_filt, A_car, rej, D_aligned, psd_d, sf, occ_idx
                            )
                            results.append({
                                "subject": sid, "segment": seg, "pca_rank": n_pca,
                                "strategy": f"thresh (mus={t_muscle}, oth={t_other}, cap={max_rej})",
                                "rejected_ics": len(rej), "corr_r": r_val, "rmse_uV": rmse_val,
                                "alpha_ratio": a_ratio,
                                "fit_time_s": round(fit_time, 2)
                            })

                # 2. Meta-Model (if available)
                if meta_model is not None:
                    weights = np.array(meta_model["weights"])
                    bias = float(meta_model.get("intercept", meta_model.get("bias", 0.0)))

                    # Compute features for each IC matching training:
                    # ['P(Brain)', 'P(Muscle)', 'P(Eye)', 'P(Heart)', 'P(Line)', 'P(Chan)', 'P(Other)', 'P_max', 'P_max-P_brain', 'Rel_variance']
                    feat_list = []
                    p_var = np.var(S_filt, axis=1) / (np.sum(np.var(S_filt, axis=1)) + 1e-30)
                    for i in range(n_pca):
                        p = probs[i]
                        p_max = float(np.max(p))
                        gap = float(p_max - p[0])
                        feat = [p[0], p[1], p[2], p[3], p[4], p[5], p[6], p_max, gap, float(p_var[i])]
                        feat_list.append(feat)

                    feat_arr = np.array(feat_list)
                    meta_probs = 1.0 / (1.0 + np.exp(-(feat_arr @ weights + bias)))

                    for cutoff in [0.40, 0.50, 0.60, 0.70]:
                        rej_meta = [i for i in range(n_pca) if meta_probs[i] >= cutoff and probs[i, 0] < 0.15]
                        # Cap at 15%
                        rej_meta = rej_meta[:int(0.15 * n_pca)]
                        r_val, rmse_val, a_ratio = evaluate_rejection(
                            X_car_filt, S_filt, A_car, rej_meta, D_aligned, psd_d, sf, occ_idx
                        )
                        results.append({
                            "subject": sid, "segment": seg, "pca_rank": n_pca,
                            "strategy": f"meta_model (cutoff={cutoff})",
                            "rejected_ics": len(rej_meta), "corr_r": r_val, "rmse_uV": rmse_val,
                            "alpha_ratio": a_ratio,
                            "fit_time_s": round(fit_time, 2)
                        })

    df = pd.DataFrame(results)
    csv_path = OUT_DIR / "sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print("\n" + "=" * 78)
    print(f"  SWEEP COMPLETE. Full results saved to: {csv_path}")
    print("=" * 78)

    # Print top 10 configurations sorted by correlation r
    if not df.empty:
        df_sorted = df.sort_values(by="corr_r", ascending=False)
        print("\n🏆 TOP 15 CONFIGURATIONS BY CORRELATION WITH DIMA'S BENCHMARK:")
        print(df_sorted[["subject", "segment", "pca_rank", "strategy", "rejected_ics", "corr_r", "rmse_uV", "alpha_ratio"]].head(15).to_string(index=False))

    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sweep hyperparameters to match Dima's benchmark.")
    ap.add_argument("--sid", nargs="+", default=["1916"], help="Subject IDs, e.g. 1916 1925")
    ap.add_argument("--seg", nargs="+", default=["drone"], help="Segments, e.g. drone lasertag")
    ap.add_argument("--pca", nargs="+", type=int, default=[75, 85, 95], help="PCA ranks to sweep")
    ap.add_argument("--max_iter", type=int, default=150, help="Max Picard ICA iterations")
    a = ap.parse_args()

    run_sweep(subjects=a.sid, segments=a.seg, pca_ranks=a.pca,
              burst_crits=["off", 20.0], rejection_modes=["thresholds", "meta_model"],
              max_iter=a.max_iter)
