"""
STEP 02: High-Precision Slice Marker Phase Detection (Zero-Duplication)
=======================================================================
Detects the exact gradient burst phase by reading only the needed segment interval
from the continuous raw recording. Saves slice metadata (slice_detection.json)
and a verification plot without creating duplicate raw files.
"""
from pathlib import Path
import json
import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .config import DEFAULT_SEGMENT_DIR, DEFAULT_RAW_VHDR, DEFAULT_TR_SEC, DEFAULT_SLICES_PER_VOLUME
except ImportError:
    from config import DEFAULT_SEGMENT_DIR, DEFAULT_RAW_VHDR, DEFAULT_TR_SEC, DEFAULT_SLICES_PER_VOLUME


def run_detect_slices(segment_dir: Path = DEFAULT_SEGMENT_DIR,
                      tr_sec: float = DEFAULT_TR_SEC,
                      slices_per_volume: int = DEFAULT_SLICES_PER_VOLUME):
    segment_dir = Path(segment_dir).resolve()
    print("=" * 70)
    print(f"[STEP 02] Detecting slice phase for segment: {segment_dir.name}")
    print("=" * 70)

    # 1. Load segment info
    info_json = segment_dir / "segment_info.json"
    if info_json.exists():
        with open(info_json, "r", encoding="utf-8") as f:
            seg_info = json.load(f)
        t_start = seg_info["t_start_sec"]
        t_stop = seg_info["t_stop_sec"]
        raw_vhdr = Path(seg_info.get("raw_vhdr", DEFAULT_RAW_VHDR)).resolve()
    else:
        print("  [WARNING] segment_info.json not found. Using full raw VHDR defaults.")
        t_start = 0.0
        t_stop = None
        raw_vhdr = Path(DEFAULT_RAW_VHDR).resolve()

    if not raw_vhdr.exists():
        raise FileNotFoundError(f"Raw BrainVision header not found: {raw_vhdr}")

    print(f"  Reading segment interval [{t_start:.2f}s .. {t_stop if t_stop else 'END'}] from: {raw_vhdr.name}")
    raw = mne.io.read_raw_brainvision(raw_vhdr, preload=False, verbose=False)
    sfreq = float(raw.info["sfreq"])
    t_max = raw.times[-1] if t_stop is None else min(raw.times[-1], t_stop)

    nominal_slice_samples = int(round((tr_sec / slices_per_volume) * sfreq))
    nominal_volume_samples = nominal_slice_samples * slices_per_volume

    print(f"  Sampling frequency: {sfreq} Hz")
    print(f"  Slice interval: {nominal_slice_samples} samples ({(nominal_slice_samples/sfreq)*1000:.1f} ms)")
    print(f"  Volume interval: {nominal_volume_samples} samples ({tr_sec:.3f} s)")

    det_chs = ["Fz", "Cz", "Pz", "Fp1", "Fp2"]
    avail_chs = [ch for ch in det_chs if ch in raw.ch_names]
    if not avail_chs:
        avail_chs = raw.ch_names[:5]

    # Crop only the detection channels in memory
    raw_crop = raw.copy().crop(tmin=t_start, tmax=t_max).pick(avail_chs)
    data = raw_crop.get_data(units="uV")

    # Compute gradient profile
    grad_profile = np.zeros(data.shape[1])
    for ch_idx in range(len(avail_chs)):
        grad_profile += np.abs(np.diff(data[ch_idx], prepend=data[ch_idx, 0]))

    # Cross-correlation with nominal comb template to find absolute phase
    comb_len = min(len(grad_profile), nominal_volume_samples * 2)
    scores = np.zeros(nominal_slice_samples)
    for offset in range(nominal_slice_samples):
        indices = np.arange(offset, comb_len, nominal_slice_samples)
        scores[offset] = np.mean(grad_profile[indices])

    best_phase = int(np.argmax(scores))
    print(f"  Optimal phase offset: {best_phase} samples ({(best_phase/sfreq)*1000:.2f} ms)")

    # Save detection metadata
    detection_info = {
        "best_phase_samples": int(best_phase),
        "nominal_slice_samples": int(nominal_slice_samples),
        "nominal_volume_samples": int(nominal_volume_samples),
        "sfreq": float(sfreq),
        "tr_sec": float(tr_sec),
        "slices_per_volume": int(slices_per_volume),
        "t_start_sec": float(t_start),
        "t_stop_sec": float(t_max),
        "raw_vhdr": str(raw_vhdr)
    }

    out_json = segment_dir / "slice_detection.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(detection_info, f, indent=2)

    # Plot verification grid
    fig, ax = plt.subplots(figsize=(10, 4))
    t_axis = np.arange(nominal_slice_samples) / sfreq * 1000.0
    ax.plot(t_axis, scores, color="#0984e3", lw=1.5, label="Comb alignment score")
    ax.axvline(t_axis[best_phase], color="#d63031", ls="--", lw=2, label=f"Best Phase: {best_phase} smp ({t_axis[best_phase]:.1f} ms)")
    ax.set_title(f"Slice Phase Alignment: {segment_dir.name}", fontweight="bold")
    ax.set_xlabel("Offset within slice period (ms)")
    ax.set_ylabel("Mean Gradient Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_png = segment_dir / "slice_phase_check.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

    print(f"[STEP 02] Done. Saved metadata: {out_json.name} and plot: {out_png.name}")
    return detection_info


if __name__ == "__main__":
    run_detect_slices()
