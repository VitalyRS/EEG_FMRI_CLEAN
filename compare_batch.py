"""
Batch spectral comparison: our cleaned FIF vs reference .set for many segments.
Emits one JSON (/tmp/batch_diff.json) with per-segment PSD curves + band ratios,
consumed by the combined HTML report builder.

Usage: python compare_batch.py 1916:ec 1916:eo ... (subject:segment tokens)
"""
import sys, json
from pathlib import Path
import numpy as np
import mne
from scipy.signal import welch

BANDS = {
    "delta 1-4": (1.0, 4.0), "theta 4-8": (4.0, 8.0), "alpha 8-13": (8.0, 13.0),
    "beta 13-30": (13.0, 30.0), "low-gamma 30-45": (30.0, 45.0),
    "hi 45-80": (45.0, 80.0), "above-80": (80.0, 124.0),
}
DATA = Path("data")
CMP = Path("files_compare")


def load(path):
    p = str(path)
    r = (mne.io.read_raw_eeglab(p, preload=True, verbose="ERROR") if p.endswith(".set")
         else mne.io.read_raw_fif(p, preload=True, verbose="ERROR"))
    r.pick("eeg")
    return r


def psd_mean(raw):
    sf = raw.info["sfreq"]
    d = raw.get_data()
    n = int(min(4 * sf, d.shape[-1]))
    f, p = welch(d, sf, nperseg=n, axis=-1)
    return f, p.mean(axis=0)


def bp(f, p, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(np.trapz(p[m], f[m])) if m.any() else 0.0


def cutoff(f, p):
    """
    Estimate the low-pass edge: highest frequency whose PSD is still >=1% of the
    in-band maximum, where the maximum is taken over 10-70 Hz. Anchoring the max
    to 10-70 Hz (not the global max) avoids the artifact where a huge delta/theta
    peak makes the 1% threshold land at ~20 Hz even though content reaches 80 Hz.
    """
    ref_mask = (f >= 10) & (f <= 70)
    if not ref_mask.any():
        return None
    thr = float(p[ref_mask].max()) * 0.01
    a = f[(p >= thr) & (f >= 10) & (f <= 120)]
    return float(a.max()) if a.size else None


def dec(f, p, npts=400):
    m = f <= 120
    f, p = f[m], p[m]
    idx = np.linspace(0, len(f) - 1, min(npts, len(f))).astype(int)
    return f[idx].tolist(), p[idx].tolist()


def our_fif(subj, seg):
    return DATA / subj / "derivatives" / "05_ica" / seg / f"{seg}_ica_clean.fif"


def ref_set(subj, seg):
    cands = sorted((CMP / subj / seg).glob("*.set"))
    return cands[0] if cands else None


def main():
    tokens = sys.argv[1:]
    out = {"segments": []}
    for tok in tokens:
        subj, seg = tok.split(":")
        fif = our_fif(subj, seg)
        ref = ref_set(subj, seg)
        if not fif.exists() or ref is None:
            print(f"[skip] {tok}: fif={fif.exists()} ref={ref is not None}")
            continue
        o, r = load(fif), load(ref)
        common = [c for c in o.ch_names if c in r.ch_names]
        o.pick(common); r.pick(common)
        fo, po = psd_mean(o); fr, pr = psd_mean(r)
        bands = []
        for name, (lo, hi) in BANDS.items():
            bo, br = bp(fo, po, lo, hi), bp(fr, pr, lo, hi)
            bands.append({"name": name, "ratio": (bo / br if br > 0 else None)})
        fo2, po2 = dec(fo, po); fr2, pr2 = dec(fr, pr)
        muscle = "_ica_musle" in ref.name  # ec/eo have extra ICA muscle removal
        out["segments"].append({
            "subj": subj, "seg": seg, "nch": len(common),
            "dur": float(o.times[-1]),
            "cut_ours": cutoff(fo, po), "cut_ref": cutoff(fr, pr),
            "ref_muscle": muscle,
            "bands": bands,
            "f_o": fo2, "p_o": po2, "f_r": fr2, "p_r": pr2,
        })
        print(f"[ok] {tok}: cut ours={out['segments'][-1]['cut_ours']:.1f} "
              f"ref={out['segments'][-1]['cut_ref']:.1f} muscle={muscle}")
    json.dump(out, open("/tmp/batch_diff.json", "w"))
    print(f"Wrote /tmp/batch_diff.json ({len(out['segments'])} segments)")


if __name__ == "__main__":
    main()
