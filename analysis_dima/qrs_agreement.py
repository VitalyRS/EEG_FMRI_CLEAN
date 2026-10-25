"""
Compare QRS detections: my pipeline (fMRIB pop_fmrib_qrsdetect on ECG of 02_resampled250)
vs. the QRS events stored in Dima's .set files.

Finds the clock shift that maximises agreement and reports recall/precision/HR.
Caches my detections in reports/dima_compare/my_qrs_<sid>_<seg>.npy.

Usage: python analysis_dima/qrs_agreement.py --seg drone lasertag --sid 1924 1925
"""
from __future__ import annotations
import argparse, glob, shutil, sys, warnings
from pathlib import Path
import numpy as np
import scipy.io as sio
import pandas as pd

warnings.filterwarnings("ignore")
import mne
mne.set_log_level("ERROR")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import step08_bcg as s8                      # noqa: E402

OUT = ROOT / "reports" / "dima_compare"


def dima_qrs(sid, seg):
    p = glob.glob(str(ROOT / "data_dima" / seg / sid / "*.set"))[0]
    m = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
    return np.array([float(e.latency) - 1 for e in np.atleast_1d(m["event"]) if str(e.type) == "qrs"])


def my_qrs(sid, seg):
    cache = OUT / f"my_qrs_{sid}_{seg}.npy"
    if cache.exists():
        return np.load(cache)
    raw = mne.io.read_raw_fif(ROOT / "data" / sid / "derivatives" / "02_resampled250" / seg / f"{seg}_250hz.fif", preload=True)
    names = list(raw.ch_names)
    data = raw.get_data() * 1e6
    work = ROOT / "analysis_dima" / "_tmp" / f"qrs_{sid}_{seg}"
    q, _ = s8.run_fmrib_bcg_matlab(data, names, names.index("ECG"), float(raw.info["sfreq"]), [], work)
    shutil.rmtree(work, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(cache, q)
    return q


def agreement(dq, mq, sf, tol_s=0.05, rng=400):
    tol = tol_s * sf
    best = None
    for s in range(-rng, rng + 1):
        d = np.min(np.abs(mq[None, :] - (dq + s)[:, None]), axis=1)
        rec = np.mean(d <= tol)
        if best is None or rec > best[1]:
            best = (s, rec)
    s = best[0]
    d_rec = np.min(np.abs(mq[None, :] - (dq + s)[:, None]), axis=1)
    d_prec = np.min(np.abs((dq + s)[None, :] - mq[:, None]), axis=1)
    return s, float(np.mean(d_rec <= tol)), float(np.mean(d_prec <= tol))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", nargs="+", default=["drone", "lasertag"])
    ap.add_argument("--sid", nargs="+", default=["1916", "1917", "1918", "1924", "1925", "1926"])
    a = ap.parse_args()
    rows = []
    for seg in a.seg:
        for sid in a.sid:
            try:
                dq, mq, sf = dima_qrs(sid, seg), my_qrs(sid, seg), 250.0
                s, rec, prec = agreement(dq, mq, sf)
                hr = lambda q: 60 / np.median(np.diff(q) / sf)
                rows.append(dict(sid=sid, seg=seg, n_dima=len(dq), n_mine=len(mq), hr_dima=hr(dq), hr_mine=hr(mq),
                                 shift=s, recall_dima_beats_found=rec, precision_my_beats_real=prec))
                print(rows[-1], flush=True)
            except Exception as e:  # noqa
                print(sid, seg, "FAILED", e, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "qrs_agreement.csv", index=False)
