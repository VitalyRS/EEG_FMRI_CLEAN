"""
STEP 03: Trim Dummy Scans & Generate Slice Trigger Text File (Zero-Duplication)
================================================================================
Calculates sample-exact working boundaries matching SPM motion parameters (rp_*.txt)
and exports a lightweight text marker file (slice_triggers.txt) and segment_work_info.json.
Eliminates duplicate BrainVision .eeg/.vhdr files on disk.
"""
from pathlib import Path
import json
import numpy as np

try:
    from .config import DEFAULT_SEGMENT_DIR, DEFAULT_TR_SEC, DEFAULT_SLICES_PER_VOLUME, DEFAULT_DUMMY_VOLUMES
except ImportError:
    from config import DEFAULT_SEGMENT_DIR, DEFAULT_TR_SEC, DEFAULT_SLICES_PER_VOLUME, DEFAULT_DUMMY_VOLUMES


def trim_dummy_scans(segment_dir: Path = DEFAULT_SEGMENT_DIR,
                     dummy_volumes: int = DEFAULT_DUMMY_VOLUMES,
                     tr_sec: float = DEFAULT_TR_SEC,
                     slices_per_volume: int = DEFAULT_SLICES_PER_VOLUME):
    segment_dir = Path(segment_dir).resolve()
    print("=" * 70)
    print(f"[STEP 03] Generating slice triggers for: {segment_dir.name}")
    print("=" * 70)

    # 1. Load slice detection info
    slice_json = segment_dir / "slice_detection.json"
    if not slice_json.exists():
        raise FileNotFoundError(f"Missing {slice_json}. Run step02 first!")

    with open(slice_json, "r", encoding="utf-8") as f:
        det = json.load(f)

    sfreq = float(det["sfreq"])
    t_start = float(det["t_start_sec"])
    best_phase = int(det["best_phase_samples"])
    nominal_slice_samples = int(det["nominal_slice_samples"])
    nominal_volume_samples = int(det["nominal_volume_samples"])
    raw_vhdr = Path(det["raw_vhdr"]).resolve()

    # 2. Check SPM motion file
    rp_files = list(segment_dir.rglob("rp_*.txt"))
    if not rp_files:
        print("  [WARNING] No rp_*.txt file found in segment directory. Using default 200 volumes.")
        n_work_volumes = 200
        rp_path = None
    else:
        rp_path = rp_files[0].resolve()
        rp_data = np.loadtxt(rp_path)
        n_work_volumes = len(rp_data)
        print(f"  Found SPM motion file: {rp_path.name} ({n_work_volumes} work volumes)")

    total_slices = n_work_volumes * slices_per_volume
    work_duration_sec = n_work_volumes * tr_sec

    # Start of work interval in continuous recording (1-based sample index for MATLAB)
    raw_start_sample_0idx = int(round(t_start * sfreq)) + best_phase + dummy_volumes * nominal_volume_samples
    raw_stop_sample_0idx  = raw_start_sample_0idx + n_work_volumes * nominal_volume_samples - 1

    sample_start_1based = raw_start_sample_0idx + 1
    sample_stop_1based  = raw_stop_sample_0idx + 1

    t_work_start_sec = raw_start_sample_0idx / sfreq
    t_work_stop_sec  = (raw_stop_sample_0idx + 1) / sfreq

    print(f"  Dummy volumes trimmed: {dummy_volumes} ({dummy_volumes * tr_sec:.1f} s)")
    print(f"  Work interval: [{t_work_start_sec:.3f}s .. {t_work_stop_sec:.3f}s] ({work_duration_sec:.1f} s)")
    print(f"  Raw continuous sample range (1-based): [{sample_start_1based} .. {sample_stop_1based}]")
    print(f"  Total slices: {total_slices} ({n_work_volumes} volumes x {slices_per_volume} slices)")

    # 3. Generate slice trigger sample indices (1-based for MATLAB relative to the cropped segment)
    slice_triggers = [1 + i * nominal_slice_samples for i in range(total_slices)]

    # 4. Save slice_triggers.txt
    triggers_txt = segment_dir / "slice_triggers.txt"
    np.savetxt(triggers_txt, slice_triggers, fmt="%d")
    print(f"  Exported slice triggers: {triggers_txt.name} ({len(slice_triggers)} markers, size: {triggers_txt.stat().st_size / 1024:.1f} KB)")

    # 5. Save segment_work_info.json
    work_info = {
        "t_work_start_sec": float(t_work_start_sec),
        "t_work_stop_sec": float(t_work_stop_sec),
        "sample_start_raw": int(sample_start_1based),
        "sample_stop_raw": int(sample_stop_1based),
        "n_work_volumes": int(n_work_volumes),
        "slices_per_volume": int(slices_per_volume),
        "nominal_slice_samples": int(nominal_slice_samples),
        "nominal_volume_samples": int(nominal_volume_samples),
        "total_slices": int(total_slices),
        "sfreq": float(sfreq),
        "raw_vhdr": str(raw_vhdr),
        "rp_file": str(rp_path) if rp_path else None
    }

    out_json = segment_dir / "segment_work_info.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(work_info, f, indent=2)

    print(f"[STEP 03] Done. Saved metadata: {out_json.name} (zero data duplication).")
    return triggers_txt


if __name__ == "__main__":
    trim_dummy_scans()
