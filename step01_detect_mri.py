"""
STEP 01: Detect MRI Scanning Sessions in Raw Continuous EEG (Zero-Duplication)
==============================================================================
Analyzes RMS envelope across control channels (Fp1, Fp2, Fz, Cz, Pz)
and detects time intervals [t_start, t_stop] for each MRI session.
Saves lightweight metadata (segment_info.json) without saving heavy intermediate .fif files.
"""
from pathlib import Path
import json
import numpy as np
import mne

try:
    from .config import DEFAULT_RAW_VHDR, DEFAULT_EXPERIMENT, PROJECT_ROOT
except ImportError:
    from config import DEFAULT_RAW_VHDR, DEFAULT_EXPERIMENT, PROJECT_ROOT


DETECTION_CHANNELS = ["Fp1", "Fp2", "Fz", "Cz", "Pz"]
WINDOW_SEC = 0.20
MIN_SEGMENT_SEC = 25.0
MERGE_GAP_SEC = 1.0
AUTO_THRESHOLD = False
MANUAL_THRESHOLD = 300.0


def detect_mri_sessions(vhdr_path: Path = DEFAULT_RAW_VHDR, output_dir: Path = None):
    vhdr_path = Path(vhdr_path).resolve()
    if output_dir is None:
        output_dir = PROJECT_ROOT / DEFAULT_EXPERIMENT
    segments_dir = Path(output_dir) / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"[STEP 01] Reading continuous EEG (metadata only): {vhdr_path}")
    print("=" * 70)

    raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose=False)
    sfreq = float(raw.info["sfreq"])
    total_dur = float(raw.times[-1])
    print(f"Sampling frequency: {sfreq} Hz | Channels: {len(raw.ch_names)} | Total Duration: {total_dur:.2f} s ({total_dur/60:.1f} min)")

    avail_chs = [ch for ch in DETECTION_CHANNELS if ch in raw.ch_names]
    if not avail_chs:
        avail_chs = raw.ch_names[:5]

    print(f"Using channels for detection: {avail_chs}")
    det_raw = raw.copy().pick(avail_chs)
    data = det_raw.get_data(units="uV")

    window_samples = int(WINDOW_SEC * sfreq)
    n_windows = data.shape[1] // window_samples

    rms_values = np.zeros(n_windows)
    for i in range(n_windows):
        start = i * window_samples
        stop = start + window_samples
        channel_rms = np.sqrt(np.mean(data[:, start:stop] ** 2, axis=1))
        rms_values[i] = np.median(channel_rms)

    smooth_window = 5
    rms_smooth = np.copy(rms_values)
    for i in range(len(rms_values)):
        left = max(0, i - smooth_window)
        right = min(len(rms_values), i + smooth_window + 1)
        rms_smooth[i] = np.median(rms_values[left:right])

    threshold = MANUAL_THRESHOLD if not AUTO_THRESHOLD else np.percentile(rms_smooth, 20) * 2.0
    print(f"Detection threshold: {threshold:.1f} uV")

    is_mri = rms_smooth > threshold
    segments = []
    in_seg = False
    seg_start = 0

    for i in range(len(is_mri)):
        if is_mri[i] and not in_seg:
            in_seg = True
            seg_start = i * window_samples / sfreq
        elif not is_mri[i] and in_seg:
            in_seg = False
            seg_stop = i * window_samples / sfreq
            if (seg_stop - seg_start) >= MIN_SEGMENT_SEC:
                segments.append((seg_start, seg_stop))

    if in_seg:
        seg_stop = len(is_mri) * window_samples / sfreq
        if (seg_stop - seg_start) >= MIN_SEGMENT_SEC:
            segments.append((seg_start, seg_stop))

    # Merge nearby segments
    merged = []
    for s_start, s_stop in segments:
        if not merged:
            merged.append([s_start, s_stop])
        else:
            prev_start, prev_stop = merged[-1]
            if s_start - prev_stop <= MERGE_GAP_SEC:
                merged[-1][1] = s_stop
            else:
                merged.append([s_start, s_stop])

    print(f"\nFound {len(merged)} MRI segments:")
    dur_lines = []
    for idx, (t0, t1) in enumerate(merged, 1):
        seg_folder = segments_dir / f"segment{idx}"
        seg_folder.mkdir(exist_ok=True)
        dur = t1 - t0
        line = f"Segment {idx}: [{t0:.2f} s - {t1:.2f} s] (Duration: {dur:.1f} s / {dur/60:.1f} min)"
        print(f"  {line}")
        dur_lines.append(line)

        # Save lightweight metadata (NO heavy .fif)
        info_path = seg_folder / "segment_info.json"
        seg_info = {
            "segment_idx": idx,
            "t_start_sec": float(t0),
            "t_stop_sec": float(t1),
            "duration_sec": float(dur),
            "raw_vhdr": str(vhdr_path.resolve()),
            "sfreq": sfreq
        }
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(seg_info, f, indent=2)

    # Save summary duration text
    with open(segments_dir / "segments_duration.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(dur_lines) + "\n")

    print(f"\n[STEP 01] Done. Metadata saved for {len(merged)} segments (zero data duplication).")
    return merged


if __name__ == "__main__":
    detect_mri_sessions()
