"""
Quantitative comparison: reference ("Dima") cleaned EEG vs. this pipeline.

For every subject/segment found in data_dima/<seg>/<sid>/ it loads
  * Dima's  EEGLAB .set  (MR_corr -> LPF80 -> SR250 -> qrs -> CB_corr_N -> asr -> ICA -> av_ref)
  * mine    02_resampled250 (input to BCG), 03_bcg, 05_ica
aligns them in time, puts everything in the same band (1-45 Hz) + average
reference and reports:
  - time lag between the two recordings
  - per-channel Pearson r (mine vs Dima) for BCG / ICA stage
  - band power ratio  mine/Dima   (delta/theta/alpha/beta/gamma + out-of-band)
  - QRS-locked residual (cardiac-averaged RMS, uV) for every stage
  - occipital alpha power, total RMS

Usage:  python analysis_dima/compare_dima.py [--seg drone lasertag] [--sid 1916 ...]
Output: reports/dima_compare/dima_compare.csv
"""
from __future__ import annotations
import argparse, glob, os, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import welch, correlate, fftconvolve

warnings.filterwarnings("ignore")
import mne
mne.set_log_level("ERROR")

ROOT = Path(__file__).resolve().parent.parent
DIMA = ROOT / "data_dima"
DATA = ROOT / "data"
OUT = ROOT / "reports" / "dima_compare"

BANDS = {"sub1": (0.1, 1), "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45), "hf45_80": (45, 80)}
OCC = ["O1", "Oz", "O2", "POz", "Pz", "P3", "P4", "P1", "P2"]


def load_dima(sid, seg):
    p = glob.glob(str(DIMA / seg / sid / "*.set"))[0]
    raw = mne.io.read_raw_eeglab(p, preload=True)
    m = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
    qrs = np.array([float(e.latency) - 1 for e in np.atleast_1d(m["event"]) if str(e.type) == "qrs"])
    hist = m["history"] if isinstance(m["history"], str) else "\n".join(map(str, np.atleast_1d(m["history"])))
    rm = re.search(r"subcomp\( EEG, \[([^\]]*)\]", hist)
    n_removed = len(rm.group(1).split()) if rm else 0
    return raw, qrs, int(np.shape(m["icaweights"])[0]), n_removed, int(re.search(r"CB_corr_(\d+)", p).group(1))


def load_mine(sid, seg):
    base = DATA / sid / "derivatives"
    out = {}
    for k, rel in {"resampled": f"02_resampled250/{seg}/{seg}_250hz.fif",
                   "bcg": f"03_bcg/{seg}/{seg}_bcg_clean.fif",
                   "ica": f"05_ica/{seg}/{seg}_ica_clean.fif"}.items():
        p = base / rel
        out[k] = mne.io.read_raw_fif(p, preload=True) if p.exists() else None
    return out


def eeg_only(raw, chans):
    r = raw.copy().pick([c for c in chans if c in raw.ch_names])
    return r.reorder_channels([c for c in chans if c in r.ch_names])


def to_band_car(raw, lo=1.0, hi=45.0):
    r = raw.copy().filter(lo, hi, verbose=False)
    r.set_eeg_reference("average", verbose=False)
    return r.get_data()


def band_power(x, sf):
    f, p = welch(x, sf, nperseg=int(4 * sf), axis=-1)
    return {k: float(np.mean(p[:, (f >= a) & (f < b)])) for k, (a, b) in BANDS.items()}


def qrs_locked_rms(x, sf, qrs, tmin=-0.2, tmax=0.8):
    a, b = int(tmin * sf), int(tmax * sf)
    ep = [x[:, int(q) + a:int(q) + b] for q in qrs if int(q) + a >= 0 and int(q) + b < x.shape[1]]
    if len(ep) < 20:
        return np.nan
    ep = np.stack(ep)
    ep = ep - ep.mean(axis=2, keepdims=True)
    return float(np.sqrt(np.mean(ep.mean(axis=0) ** 2)) * 1e6)


def align(ref, x, max_lag=500):
    """lag (samples) s.t. x[t+lag] ~ ref[t]. FFT cross-correlation (1-45Hz CAR data)
    averaged over every 6th channel. Dima's files are offset from ours by up to ~140
    samples (different dummy-scan trimming), so the search window must be wide."""
    n = min(ref.shape[1], x.shape[1])
    acc = np.zeros(2 * max_lag + 1)
    chs = list(range(0, ref.shape[0], 6))
    for i in chs:
        r = ref[i, :n] - ref[i, :n].mean()
        y = x[i, :n] - x[i, :n].mean()
        c = fftconvolve(y, r[::-1], mode="full") / (n * r.std() * y.std() + 1e-30)
        acc += c[n - 1 - max_lag:n + max_lag]
    acc /= len(chs)
    k = int(np.argmax(acc))
    return k - max_lag, float(acc[k])


def run(sid, seg):
    dr, qrs, n_ic, n_rm, cb = load_dima(sid, seg)
    mine = load_mine(sid, seg)
    chans = [c for c in dr.ch_names if c in mine["ica"].ch_names]
    sf = dr.info["sfreq"]
    D = eeg_only(dr, chans)
    row = dict(sid=sid, seg=seg, dima_CB=cb, dima_n_ic=n_ic, dima_ic_removed=n_rm, n_qrs=len(qrs))

    # --- raw spectra out-of-band (no filtering) to expose pipeline filter differences
    d_raw = D.get_data()
    for k in ("bcg", "ica"):
        if mine[k] is None:
            continue
        mr = eeg_only(mine[k], chans).get_data()
        pd_, pm_ = band_power(d_raw, sf), band_power(mr, sf)
        row[f"rawband_sub1_ratio_{k}"] = pm_["sub1"] / pd_["sub1"]
        row[f"rawband_hf_ratio_{k}"] = pm_["hf45_80"] / max(pd_["hf45_80"], 1e-30)

    Dx = to_band_car(D)
    stages = {"dima": Dx}
    for k in ("resampled", "bcg", "ica"):
        if mine[k] is not None:
            stages[k] = to_band_car(eeg_only(mine[k], chans))

    # --- alignment (Dima vs mine ICA)
    ref_key = "ica" if "ica" in stages else "bcg"
    lag, c = align(Dx, stages[ref_key])
    row["lag_samples"], row["align_r"] = lag, c
    start_dima = max(30, 30 - lag)
    start_other = start_dima + lag
    other_stages = [v for k, v in stages.items() if k != "dima"]
    n = min(Dx.shape[1] - start_dima, min(v.shape[1] - start_other for v in other_stages))
    stages["dima"] = stages["dima"][:, start_dima:start_dima + n]
    for k in list(stages):
        if k != "dima":
            stages[k] = stages[k][:, start_other:start_other + n]
    qrs_a = qrs - start_dima

    occ = [chans.index(c) for c in OCC if c in chans]
    for k, x in stages.items():
        bp = band_power(x, sf)
        row[f"{k}_rms_uV"] = float(np.sqrt(np.mean(x ** 2)) * 1e6)
        row[f"{k}_cardiac_rms_uV"] = qrs_locked_rms(x, sf, qrs_a)
        row[f"{k}_occ_alpha"] = float(np.mean(welch(x[occ], sf, nperseg=int(4 * sf), axis=-1)[1][:, 8 * 4:13 * 4 + 1]))
        for b in ("delta", "theta", "alpha", "beta", "gamma"):
            row[f"{k}_{b}"] = bp[b]
    for k in ("bcg", "ica"):
        if k in stages:
            r = [np.corrcoef(stages["dima"][i], stages[k][i])[0, 1] for i in range(len(chans))]
            row[f"r_{k}_vs_dima_mean"] = float(np.mean(r))
            row[f"r_{k}_vs_dima_min"] = float(np.min(r))
            for b in ("delta", "theta", "alpha", "beta", "gamma"):
                row[f"ratio_{k}_{b}"] = row[f"{k}_{b}"] / row[f"dima_{b}"]
            row[f"ratio_{k}_rms"] = row[f"{k}_rms_uV"] / row["dima_rms_uV"]
            row[f"ratio_{k}_cardiac"] = row[f"{k}_cardiac_rms_uV"] / row["dima_cardiac_rms_uV"]
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", nargs="+", default=["drone", "lasertag"])
    ap.add_argument("--sid", nargs="+", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seg in a.seg:
        for sd in sorted((DIMA / seg).glob("*")):
            if a.sid and sd.name not in a.sid:
                continue
            try:
                r = run(sd.name, seg)
                rows.append(r)
                print(f"{sd.name}/{seg}: lag={r['lag_samples']} r_ica={r.get('r_ica_vs_dima_mean', np.nan):.3f} "
                      f"card dima={r['dima_cardiac_rms_uV']:.2f} bcg={r.get('bcg_cardiac_rms_uV', np.nan):.2f} "
                      f"ica={r.get('ica_cardiac_rms_uV', np.nan):.2f} uV", flush=True)
            except Exception as e:  # noqa
                print(f"{sd.name}/{seg}: FAILED {e}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "dima_compare.csv", index=False)
    print("saved", OUT / "dima_compare.csv")
