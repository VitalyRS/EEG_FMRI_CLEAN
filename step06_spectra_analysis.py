"""
STEP 06: Quantitative Alpha-Preservation & Gradient Suppression Quality Analysis
================================================================================
Evaluates dual-objective criteria:
  1. Gradient artifact suppression across non-alpha harmonics (20, 30, 40, 50, 60 Hz).
  2. Quantitative Alpha Rhythm preservation (8-13 Hz):
     - Alpha Prominence (P_alpha / P_background)
     - Alpha Peak Frequency (f_alpha)
     - Alpha Preservation Ratio vs outside-MRI EEG21 reference
     - Eyes-Closed / Eyes-Open (EC/EO) physiological alpha reactivity
Exports summary.csv, metrics.csv, step03_spectra.png, and alpha_quality_check.png.
"""
from pathlib import Path
import json
import csv
import numpy as np
import mne
import gc
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .config import (DEFAULT_SEGMENT_DIR, EVAL_CHANNELS, ALPHA_BAND,
                         BG_BANDS, GRADIENT_HARMONICS)
except ImportError:
    from config import (DEFAULT_SEGMENT_DIR, EVAL_CHANNELS, ALPHA_BAND,
                         BG_BANDS, GRADIENT_HARMONICS)

FMAX_COMPARE = 40.0
NPERSEG_SEC = 4


def _time_to_s(tstr):
    tstr = str(tstr).replace(",", ".")
    h, m, rest = tstr.split(":")
    s, *ms = rest.split(".")
    ms_val = int(ms[0]) if ms else 0
    return int(h) * 3600 + int(m) * 60 + int(s) + ms_val / 1000.0


def parse_blocks(blocks_file: Path):
    blocks = {}
    with open(blocks_file, "r", encoding="cp1251", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line or "." not in line:
                continue
            idx_dot = line.index(".")
            idx_eq  = line.index("=")
            idx_str = int(line[:idx_dot])
            key     = line[idx_dot+1:idx_eq]
            val     = line[idx_eq+1:]
            blocks.setdefault(idx_str, {})[key] = val

    result = []
    for idx in sorted(blocks):
        b = blocks[idx]
        caption = b.get("Caption", "")
        begin_s = _time_to_s(b.get("begin", "0"))
        end_s   = _time_to_s(b.get("end", "0"))
        result.append((caption, begin_s, end_s))
    return result


def psd_for_channel(data_1d: np.ndarray, sfreq: float, nperseg_sec: int = 4):
    nperseg = int(nperseg_sec * sfreq)
    return welch(data_1d, sfreq, nperseg=nperseg, noverlap=nperseg // 2)


# Slice-comb harmonics (10 Hz here) are mathematically unrecoverable after AAS.
# They are excluded from alpha peak / prominence so the dead bin cannot pin the
# measured peak to 10 Hz or artificially deflate preservation.
ALPHA_COMB_NOTCH_HZ = 0.15
SLICE_COMB_HZ = 10.0


def _comb_notch_mask(f: np.ndarray, comb_hz: float, half_width: float = ALPHA_COMB_NOTCH_HZ) -> np.ndarray:
    if comb_hz <= 0:
        return np.zeros_like(f, dtype=bool)
    nearest_harm = np.round(f / comb_hz) * comb_hz
    return (np.abs(f - nearest_harm) <= half_width) & (nearest_harm > 0)


def compute_alpha_metrics(f: np.ndarray, psd: np.ndarray, comb_hz: float = SLICE_COMB_HZ):
    on_comb = _comb_notch_mask(f, comb_hz)
    m_alpha = (f >= ALPHA_BAND[0]) & (f <= ALPHA_BAND[1])
    m_alpha_off = m_alpha & ~on_comb   # exclude dead comb bin(s)
    m_bg1   = (f >= BG_BANDS[0][0]) & (f <= BG_BANDS[0][1])
    m_bg2   = (f >= BG_BANDS[1][0]) & (f <= BG_BANDS[1][1])

    df = f[1] - f[0] if len(f) > 1 else 1.0
    p_alpha = float(np.sum(psd[m_alpha_off]) * df) if np.any(m_alpha_off) else 1e-6
    p_bg    = float(0.5 * (np.mean(psd[m_bg1]) + np.mean(psd[m_bg2]))) if (np.any(m_bg1) and np.any(m_bg2)) else 1e-6
    prominence = float(np.mean(psd[m_alpha_off]) / max(p_bg, 1e-12)) if np.any(m_alpha_off) else 1.0

    if np.any(m_alpha_off):
        alpha_f = f[m_alpha_off]
        alpha_psd = psd[m_alpha_off]
        peak_freq = float(alpha_f[np.argmax(alpha_psd)])
    else:
        peak_freq = 10.0

    return {
        "p_alpha": p_alpha,
        "p_bg": p_bg,
        "prominence": prominence,
        "peak_freq": peak_freq
    }


def compute_spectra(segment_dir: Path = DEFAULT_SEGMENT_DIR):
    segment_dir = Path(segment_dir).resolve()
    print("=" * 75)
    print(f"[STEP 06] Quantitative Alpha Quality & Spectra Analysis: {segment_dir.name}")
    print("=" * 75)

    work_info_path = segment_dir / "segment_work_info.json"
    if not work_info_path.exists():
        raise FileNotFoundError(f"Missing {work_info_path}. Run step03 first!")

    with open(work_info_path, "r", encoding="utf-8") as f:
        work_info = json.load(f)

    raw_vhdr = Path(work_info["raw_vhdr"]).resolve()
    t_start = float(work_info["t_work_start_sec"])
    t_stop  = float(work_info["t_work_stop_sec"])

    # Cleaned file (.set) - pick newest
    set_files = list(segment_dir.glob("*bergen*optuna*.set"))
    if not set_files:
        set_files = list(segment_dir.glob("*.set"))
    if not set_files:
        raise FileNotFoundError(f"No cleaned .set files found in {segment_dir}. Run step05 first!")
    clean_set = max(set_files, key=lambda p: p.stat().st_mtime)

    out_dict = {}

    # 1. Compute Raw PSD directly from continuous raw VHDR
    print(f"  Reading raw EEG window [{t_start:.2f}s .. {t_stop:.2f}s] from: {raw_vhdr.name}")
    raw_raw = mne.io.read_raw_brainvision(raw_vhdr, preload=False, verbose=False)
    sfreq_raw = float(raw_raw.info["sfreq"])
    raw_crop = raw_raw.copy().crop(tmin=t_start, tmax=min(raw_raw.times[-1], t_stop))

    for ch in EVAL_CHANNELS:
        picks = [c for c in raw_crop.ch_names if c.upper() == ch.upper()]
        if not picks:
            continue
        data = raw_crop.get_data(picks=picks[:1], units="uV")[0]
        f, psd = psd_for_channel(data, sfreq_raw, NPERSEG_SEC)
        out_dict[f"raw_{ch}_f"] = f
        out_dict[f"raw_{ch}_psd"] = psd
        out_dict[f"raw_{ch}_std"] = float(np.std(data))
    del raw_raw, raw_crop; gc.collect()

    # 2. Compute Cleaned PSD from EEGLAB .set
    print(f"  Loading cleaned EEG: {clean_set.name}")
    clean_raw = mne.io.read_raw_eeglab(clean_set, preload=False, verbose=False)
    sfreq_clean = float(clean_raw.info["sfreq"])
    for ch in EVAL_CHANNELS:
        picks = [c for c in clean_raw.ch_names if c.upper() == ch.upper()]
        if not picks:
            continue
        data = clean_raw.get_data(picks=picks[:1], units="uV")[0]
        f, psd = psd_for_channel(data, sfreq_clean, NPERSEG_SEC)
        out_dict[f"clean_{ch}_f"] = f
        out_dict[f"clean_{ch}_psd"] = psd
        out_dict[f"clean_{ch}_std"] = float(np.std(data))
    del clean_raw; gc.collect()

    # 3. Process EEG21 Reference (Outside MRI) with EO/EC Reactivity
    eeg21_dirs = [segment_dir / "add" / "eeg21", segment_dir / "eeg21"]
    eeg21_dir = next((d for d in eeg21_dirs if d.exists()), None)

    eeg21_available = False
    eo_ec_data = {}
    if eeg21_dir is not None:
        edf_files = list(eeg21_dir.glob("*.edf"))
        blocks_files = list(eeg21_dir.glob("*.blocks"))
        if edf_files and blocks_files:
            try:
                print(f"  [INFO] Found EEG21 reference files: {edf_files[0].name}")
                edf = mne.io.read_raw_edf(str(edf_files[0]), preload=True, verbose=False)
                sfreq_edf = float(edf.info["sfreq"])
                blocks = parse_blocks(blocks_files[0])
                g1_blocks = [(cap, b, e) for cap, b, e in blocks if cap.startswith("G1.")]  # Eyes Open
                g2_blocks = [(cap, b, e) for cap, b, e in blocks if cap.startswith("G2.")]  # Eyes Closed

                # Overall EEG21 PSD
                for ch in EVAL_CHANNELS:
                    ch_edf = [c for c in edf.ch_names if c.upper() == ch.upper()]
                    if not ch_edf:
                        continue
                    # Eyes closed segments
                    ec_segs = []
                    for cap, b_s, e_s in (g2_blocks if g2_blocks else blocks):
                        s0 = max(0, int(b_s * sfreq_edf))
                        s1 = min(edf.n_times, int(e_s * sfreq_edf))
                        if s1 > s0:
                            ec_segs.append(edf.get_data(picks=ch_edf[:1], start=s0, stop=s1, units="uV")[0])
                    if ec_segs:
                        concat_ec = np.concatenate(ec_segs)
                        f_e, psd_e = psd_for_channel(concat_ec, sfreq_edf, NPERSEG_SEC)
                        out_dict[f"eeg21_{ch}_f"] = f_e
                        out_dict[f"eeg21_{ch}_psd"] = psd_e
                        out_dict[f"eeg21_{ch}_std"] = float(np.std(concat_ec))

                    # Eyes open segments for EO/EC ratio
                    eo_segs = []
                    for cap, b_s, e_s in g1_blocks:
                        s0 = max(0, int(b_s * sfreq_edf))
                        s1 = min(edf.n_times, int(e_s * sfreq_edf))
                        if s1 > s0:
                            eo_segs.append(edf.get_data(picks=ch_edf[:1], start=s0, stop=s1, units="uV")[0])
                    if eo_segs and ec_segs:
                        concat_eo = np.concatenate(eo_segs)
                        f_eo, psd_eo = psd_for_channel(concat_eo, sfreq_edf, NPERSEG_SEC)
                        m_a = (f_e >= 8.0) & (f_e <= 13.0)
                        p_ec = np.mean(psd_e[m_a]) if np.any(m_a) else 1e-6
                        p_eo = np.mean(psd_eo[m_a]) if np.any(m_a) else 1e-6
                        eo_ec_data[ch] = float(p_ec / max(p_eo, 1e-12))

                del edf; gc.collect()
                eeg21_available = True
                print("  [SUCCESS] EEG21 outside-MRI reference loaded and processed.")
            except Exception as ex:
                print(f"  [WARNING] Could not parse EEG21 reference: {ex}. Skipping reference comparison.")
                eeg21_available = False

    out_dict["eeg21_available"] = np.array([eeg21_available])

    # 4. Quantitative Metrics Calculation & Summary Table
    rows = []
    print("\n" + "=" * 85)
    print(f"{'Channel':<8} | {'GradSupp':<10} | {'AlphaClean':<12} | {'AlphaPromClean':<14} | {'AlphaPeak':<10} | {'AlphaPreserv':<12} | {'Status'}")
    print("-" * 85)

    for ch in EVAL_CHANNELS:
        f_r = out_dict.get(f"raw_{ch}_f")
        p_r = out_dict.get(f"raw_{ch}_psd")
        f_c = out_dict.get(f"clean_{ch}_f")
        p_c = out_dict.get(f"clean_{ch}_psd")

        if f_r is None or f_c is None or p_r is None or p_c is None:
            continue

        # Gradient suppression (20, 30, 40, 50, 60 Hz)
        p_r_harm = [p_r[np.argmin(np.abs(f_r - h))] for h in GRADIENT_HARMONICS if h <= f_r[-1]]
        p_c_harm = [p_c[np.argmin(np.abs(f_c - h))] for h in GRADIENT_HARMONICS if h <= f_c[-1]]
        m_r_h = float(np.mean(p_r_harm)) if p_r_harm else 1.0
        m_c_h = float(np.mean(p_c_harm)) if p_c_harm else 1.0
        g_supp = float(np.clip(1.0 - (m_c_h / max(m_r_h, 1e-12)), 0.0, 1.0))

        # Alpha metrics
        m_raw   = compute_alpha_metrics(f_r, p_r)
        m_clean = compute_alpha_metrics(f_c, p_c)

        alpha_preserv = 1.0
        prom_eeg21 = 0.0
        peak_eeg21 = 0.0
        if eeg21_available and f"eeg21_{ch}_f" in out_dict:
            f_e = out_dict[f"eeg21_{ch}_f"]
            p_e = out_dict[f"eeg21_{ch}_psd"]
            m_eeg21 = compute_alpha_metrics(f_e, p_e)
            prom_eeg21 = m_eeg21["prominence"]
            peak_eeg21 = m_eeg21["peak_freq"]
            alpha_preserv = float(m_clean["prominence"] / max(prom_eeg21, 1e-6))
        else:
            alpha_preserv = float(min(1.0, m_clean["prominence"] / 2.0))

        status = "EXCELLENT" if (g_supp >= 0.98 and alpha_preserv >= 0.80) else ("PASS" if g_supp >= 0.95 else "REVIEW")

        row = {
            "segment": segment_dir.name,
            "channel": ch,
            "alpha_raw": f"{m_raw['p_alpha']:.2f}",
            "alpha_clean": f"{m_clean['p_alpha']:.2f}",
            "alpha_prominence_clean": f"{m_clean['prominence']:.2f}",
            "alpha_prominence_eeg21": f"{prom_eeg21:.2f}" if eeg21_available else "N/A",
            "alpha_peak_clean": f"{m_clean['peak_freq']:.1f}",
            "alpha_peak_eeg21": f"{peak_eeg21:.1f}" if eeg21_available else "N/A",
            "gradient_suppression": f"{g_supp * 100:.2f}%",
            "alpha_preservation": f"{alpha_preserv * 100:.1f}%",
            "eo_ec_ratio_eeg21": f"{eo_ec_data.get(ch, 1.0):.2f}" if eeg21_available else "N/A",
            "status": status
        }
        rows.append(row)
        print(f"{ch:<8} | {row['gradient_suppression']:<10} | {row['alpha_clean']:<12} | {row['alpha_prominence_clean']:<14} | {row['alpha_peak_clean']:<10} | {row['alpha_preservation']:<12} | {status}")

    print("=" * 85)

    # Save summary.csv & metrics.csv
    summary_csv = segment_dir / "summary_alpha_quality.csv"
    metrics_csv = segment_dir / "metrics.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved quality metrics tables: {summary_csv.name} and {metrics_csv.name}")

    # Save spectra npz
    out_npz = segment_dir / "step03_spectra_data.npz"
    np.savez_compressed(out_npz, **out_dict)
    print(f"  Saved spectra data array: {out_npz.name}")

    # Plot Full Spectra & Zoomed Alpha Preservation Quality Check
    plot_spectra_comparison(out_dict, eeg21_available, segment_dir)
    plot_alpha_quality_check(out_dict, eeg21_available, segment_dir, rows)
    return out_dict


def plot_spectra_comparison(data_dict: dict, eeg21_available: bool, segment_dir: Path):
    plot_chs = ["Fz", "Cz", "Pz", "Oz", "O1", "O2"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Power Spectral Density (PSD 0.5 - 40 Hz) - {segment_dir.name}\n"
        "Comparison: Raw (inside MRI) vs Bergen AAS Cleaned (Optuna Winner)"
        + (" vs EEG21 Outside MRI" if eeg21_available else ""),
        fontsize=13, fontweight="bold"
    )

    for idx, ch in enumerate(plot_chs):
        ax = axes[idx // 3, idx % 3]
        f_raw = data_dict.get(f"raw_{ch}_f")
        psd_raw = data_dict.get(f"raw_{ch}_psd")
        f_clean = data_dict.get(f"clean_{ch}_f")
        psd_clean = data_dict.get(f"clean_{ch}_psd")

        if f_raw is not None and psd_raw is not None:
            m_r = (f_raw >= 0.5) & (f_raw <= FMAX_COMPARE)
            ax.semilogy(f_raw[m_r], psd_raw[m_r], color="#d63031", lw=1.2, label="Raw (inside MRI)", alpha=0.6)

        if f_clean is not None and psd_clean is not None:
            m_c = (f_clean >= 0.5) & (f_clean <= FMAX_COMPARE)
            ax.semilogy(f_clean[m_c], psd_clean[m_c], color="#00b894", lw=1.8, label="Bergen Clean (Optuna)")

        if eeg21_available:
            f_e = data_dict.get(f"eeg21_{ch}_f")
            psd_e = data_dict.get(f"eeg21_{ch}_psd")
            if f_e is not None and psd_e is not None:
                m_e = (f_e >= 0.5) & (f_e <= FMAX_COMPARE)
                ax.semilogy(f_e[m_e], psd_e[m_e], color="#6c5ce7", lw=1.4, ls="--", label="EEG21 (outside MRI)")

        for h in [10, 20, 30, 40]:
            ax.axvline(h, color="gray", ls=":", alpha=0.4)

        ax.set_title(f"Channel: {ch}", fontweight="bold", fontsize=11)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (uV^2/Hz)")
        ax.set_xlim(0.5, 40)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    out_png = segment_dir / "step03_spectra.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved full spectra plot: {out_png.name}")


def plot_alpha_quality_check(data_dict: dict, eeg21_available: bool, segment_dir: Path, rows: list):
    focus_chs = ["O1", "Oz", "O2", "Pz"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Alpha Rhythm Preservation Dashboard (5 - 20 Hz) - {segment_dir.name}\n"
        "Quantitative Evaluation of Occipital/Parietal Alpha Structure after Bergen AAS",
        fontsize=13, fontweight="bold"
    )

    for idx, ch in enumerate(focus_chs):
        ax = axes[idx // 2, idx % 2]
        f_clean = data_dict.get(f"clean_{ch}_f")
        psd_clean = data_dict.get(f"clean_{ch}_psd")
        f_raw = data_dict.get(f"raw_{ch}_f")
        psd_raw = data_dict.get(f"raw_{ch}_psd")

        # Highlight Alpha Band
        ax.axvspan(8.0, 13.0, color="#ffeaa7", alpha=0.35, label="Alpha Band (8-13 Hz)")

        if f_raw is not None and psd_raw is not None:
            m_r = (f_raw >= 5.0) & (f_raw <= 20.0)
            ax.semilogy(f_raw[m_r], psd_raw[m_r], color="#d63031", lw=1.2, alpha=0.4, label="Raw MRI")

        if f_clean is not None and psd_clean is not None:
            m_c = (f_clean >= 5.0) & (f_clean <= 20.0)
            ax.semilogy(f_clean[m_c], psd_clean[m_c], color="#00b894", lw=2.2, label="Clean MRI (Bergen)")

        if eeg21_available:
            f_e = data_dict.get(f"eeg21_{ch}_f")
            psd_e = data_dict.get(f"eeg21_{ch}_psd")
            if f_e is not None and psd_e is not None:
                m_e = (f_e >= 5.0) & (f_e <= 20.0)
                ax.semilogy(f_e[m_e], psd_e[m_e], color="#6c5ce7", lw=1.6, ls="--", label="EEG21 Outside MRI")

        # Find row for badge
        row = next((r for r in rows if r["channel"] == ch), None)
        title_extra = ""
        if row:
            title_extra = f"\nGradSupp: {row['gradient_suppression']} | AlphaProm: {row['alpha_prominence_clean']} | AlphaPeak: {row['alpha_peak_clean']}Hz"

        ax.set_title(f"Channel: {ch} {title_extra}", fontweight="bold", fontsize=10)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (uV^2/Hz)")
        ax.set_xlim(5.0, 20.0)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    out_png = segment_dir / "alpha_quality_check.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved alpha quality dashboard plot: {out_png.name}")


if __name__ == "__main__":
    compute_spectra()
