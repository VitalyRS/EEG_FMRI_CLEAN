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
    from .config import (DEFAULT_SEGMENT_DIR, DEFAULT_TR_SEC, DEFAULT_SLICES_PER_VOLUME,
                         DEFAULT_DUMMY_VOLUMES, RP_DIR)
except ImportError:
    from config import (DEFAULT_SEGMENT_DIR, DEFAULT_TR_SEC, DEFAULT_SLICES_PER_VOLUME,
                        DEFAULT_DUMMY_VOLUMES, RP_DIR)


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

    # 2. Locate the SPM motion file for THIS segment.
    # Priority 1: rp_file / rp_path in segment_info.json (written by step00 / step01 / pipeline_multi).
    # Priority 2: FMRI_ROOT/<subject_id>FMRI/rp_*_<suffix>_*.txt
    # Priority 3: subject-level data/<subject>/raw/rp_spm/rp_<segmentNN>.txt
    seg_info_path = segment_dir / "segment_info.json"
    rp_path = None
    subject_id = segment_dir.parent.parent.name if segment_dir.parent.name == "segments" else segment_dir.parent.name
    suffix = None

    if seg_info_path.exists():
        with open(seg_info_path, "r", encoding="utf-8") as _f:
            _seg_info = json.load(_f)
        suffix = _seg_info.get("session_suffix")
        _rp_str = _seg_info.get("rp_path") or _seg_info.get("rp_file")
        if _rp_str:
            _rp_candidate = Path(_rp_str)
            if _rp_candidate.exists():
                rp_path = _rp_candidate.resolve()
                print(f"  rp file from segment_info.json: {rp_path}")

    # Fallback to FMRI_ROOT/<subject_id>FMRI/
    if rp_path is None and suffix:
        from .config import FMRI_ROOT
        fmri_dir = FMRI_ROOT / f"{subject_id}FMRI"
        if fmri_dir.exists():
            for f in sorted(fmri_dir.glob(f"rp_*{suffix}*.txt")):
                rp_path = f.resolve()
                print(f"  rp file found in {fmri_dir}: {rp_path}")
                break

    if rp_path is None:
        rp_candidates = [
            RP_DIR / f"rp_{segment_dir.name}.txt",
            RP_DIR / f"{segment_dir.name}.txt",
        ]
        rp_path = next((p for p in rp_candidates if p.exists()), None)

    if rp_path is None:
        raise FileNotFoundError(
            f"\n[ERROR] SPM motion file (rp_*.txt) was NOT found for segment '{segment_dir.name}' (subject '{subject_id}')!\n"
            f"Expected location:\n"
            f"  - /media/vitaly/48DEA853CCEBFBF0/0DATDA_2026_eeg_fnri/fmri/fmri/{subject_id}FMRI/rp_*_{suffix or '*'}_*.txt\n"
            f"  - or in {seg_info_path} under 'rp_path'/'rp_file'\n"
            f"Execution stopped. Please verify that the fMRI SPM rp file exists."
        )

    rp_path = rp_path.resolve()
    rp_data = np.loadtxt(rp_path)
    n_work_volumes = len(rp_data)
    print(f"  Found SPM motion file: {rp_path} ({n_work_volumes} work volumes)")

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
        "rp_file": str(rp_path) if rp_path else None,
        "rp_path": str(rp_path) if rp_path else None
    }

    out_json = segment_dir / "segment_work_info.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(work_info, f, indent=2)

    print(f"[STEP 03] Done. Saved metadata: {out_json.name} (zero data duplication).")
    return triggers_txt


if __name__ == "__main__":
    import argparse
    try:
        from .config import DATA_ROOT
    except ImportError:
        from config import DATA_ROOT

    parser = argparse.ArgumentParser(description="STEP 03: Trim dummy scans & generate slice triggers")
    parser.add_argument("--subject", default=None, help="Subject ID (e.g. 1916)")
    parser.add_argument("--segment", default=None, help="Segment name (e.g. ec, drone) or omit for all segments of subject")
    parser.add_argument("--all", action="store_true", help="Process all available subjects and segments")
    args = parser.parse_args()

    seg_dirs: list[Path] = []
    if args.subject:
        subj_seg_dir = DATA_ROOT / args.subject / "segments"
        if args.segment:
            target = subj_seg_dir / args.segment
            if target.exists():
                seg_dirs.append(target)
            else:
                print(f"[ERROR] Segment folder not found: {target}")
        else:
            if subj_seg_dir.exists():
                seg_dirs.extend(sorted(p for p in subj_seg_dir.iterdir() if p.is_dir() and (p / "slice_detection.json").exists()))
            else:
                print(f"[ERROR] No segments directory found for subject {args.subject}: {subj_seg_dir}")
    elif args.all or (not args.subject and not args.segment):
        for subj_dir in sorted(DATA_ROOT.glob("*")):
            subj_seg_dir = subj_dir / "segments"
            if subj_seg_dir.exists():
                seg_dirs.extend(sorted(p for p in subj_seg_dir.iterdir() if p.is_dir() and (p / "slice_detection.json").exists()))

    if not seg_dirs:
        print("[ERROR] No segments found with slice_detection.json. Run step02_detect_slices.py first!")
    else:
        for sdir in seg_dirs:
            trim_dummy_scans(sdir)
