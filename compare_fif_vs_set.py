"""
Compare our cleaned FIF against the reference EEGLAB .set for one segment.
Plots overlaid Welch PSDs (mean over common EEG channels) and prints a table of
band-power ratios so we can see how close our output now sits to the reference.

Usage:
    python compare_fif_vs_set.py <our_fif> <reference_set> [out_png]
"""
import sys
from pathlib import Path
import numpy as np
import mne
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BANDS = {
    "delta 1-4": (1.0, 4.0),
    "theta 4-8": (4.0, 8.0),
    "alpha 8-13": (8.0, 13.0),
    "beta 13-30": (13.0, 30.0),
    "low-gamma 30-45": (30.0, 45.0),
    "hi 45-80": (45.0, 80.0),
    "above-80": (80.0, 124.0),
}


def load(path):
    path = str(path)
    if path.endswith(".set"):
        r = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    else:
        r = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    r.pick("eeg")
    return r


def psd_mean(raw):
    sf = raw.info["sfreq"]
    data = raw.get_data()  # V
    nperseg = int(min(4 * sf, data.shape[-1]))
    f, psd = welch(data, sf, nperseg=nperseg, axis=-1)
    return f, psd.mean(axis=0), sf


def band_power(f, psd, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(np.trapz(psd[m], f[m])) if m.any() else 0.0


def main():
    our_fif = Path(sys.argv[1])
    ref_set = Path(sys.argv[2])
    out_png = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("fif_vs_set_psd.png")

    ours = load(our_fif)
    ref = load(ref_set)

    common = [c for c in ours.ch_names if c in ref.ch_names]
    print(f"Our FIF : {len(ours.ch_names)} ch @ {ours.info['sfreq']}Hz, {ours.times[-1]:.1f}s")
    print(f"Ref .set: {len(ref.ch_names)} ch @ {ref.info['sfreq']}Hz, {ref.times[-1]:.1f}s")
    print(f"Common EEG channels: {len(common)}")

    ours.pick(common)
    ref.pick(common)

    f_o, p_o, sf_o = psd_mean(ours)
    f_r, p_r, sf_r = psd_mean(ref)

    print("\n=== Band power (mean PSD integrated), ratio ours/ref ===")
    print(f"{'band':<18}{'ours':>12}{'ref':>12}{'ours/ref':>10}")
    for name, (lo, hi) in BANDS.items():
        bo = band_power(f_o, p_o, lo, hi)
        br = band_power(f_r, p_r, lo, hi)
        ratio = bo / br if br > 0 else float("nan")
        print(f"{name:<18}{bo:>12.3e}{br:>12.3e}{ratio:>10.2f}")

    # Spectral cutoff estimate (where PSD falls to 1% of its in-band max)
    def cutoff(f, p):
        band = (f >= 1) & (f <= 120)
        pmax = p[band].max()
        thr = pmax * 0.01
        above = f[(p >= thr) & (f >= 1)]
        return above.max() if above.size else np.nan
    print(f"\nEstimated LP cutoff  ours={cutoff(f_o,p_o):.1f}Hz  ref={cutoff(f_r,p_r):.1f}Hz")

    plt.figure(figsize=(11, 6))
    plt.semilogy(f_o, p_o, label="ours (FIF)", lw=1.3)
    plt.semilogy(f_r, p_r, label="reference (.set)", lw=1.3, alpha=0.8)
    for lo, hi in [(8, 13)]:
        plt.axvspan(lo, hi, color="green", alpha=0.08)
    plt.axvline(80, color="red", ls="--", lw=1, label="80 Hz (ref LP)")
    plt.axvline(45, color="orange", ls=":", lw=1, label="45 Hz (old LP)")
    plt.xlim(0, 120)
    plt.xlabel("Hz")
    plt.ylabel("PSD (V^2/Hz)")
    plt.title(f"FIF vs reference .set - {our_fif.parent.parent.name}")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f"\nSaved plot: {out_png}")


if __name__ == "__main__":
    main()
