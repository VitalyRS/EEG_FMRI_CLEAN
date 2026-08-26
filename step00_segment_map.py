"""
STEP 00: Segment Mapping — Build segment_info.json for each EEG segment
========================================================================
This step replaces step01 (detect_mri_sessions) in the multi-subject pipeline.
Instead of detecting MRI sessions from the raw EEG signal amplitude,
it uses:
  - BrainVision .vmrk marker files (ec/eo annotations from the experimenter)
  - SPM motion parameter files (rp_*.txt) for precise task-segment durations

Output per segment (compatible with existing step02–step12):
  <processed_root>/<subject_id>/segments/<seg_name>/segment_info.json

segment_info.json fields (same schema as step01):
  segment_idx     — integer index within the subject (1-based)
  segment_name    — human-readable name: "ec", "eo", "drone", "lasertag", "video",
                    "ec_out", "eo_out"
  t_start_sec     — start time in the continuous .vhdr recording (seconds)
  t_stop_sec      — stop  time in the continuous .vhdr recording (seconds)
  duration_sec    — t_stop - t_start
  raw_vhdr        — absolute path to the .vhdr file (inside or outside)
  sfreq           — sampling frequency (Hz)
  rp_file         — absolute path to the SPM rp_*.txt file, or null
  needs_bergen    — true if MRI artifact is present (inside recording with rp)
  session_suffix  — fMRI session number: "301"/"401"/…, or "" for outside segs
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    from .segment_mapper import (
        discover_subjects, map_segments_for_subject,
        EEG_ORIGINAL_ROOT, FMRI_ROOT, PROCESSED_ROOT,
        SubjectInfo, SegmentInfo,
    )
except ImportError:
    from segment_mapper import (
        discover_subjects, map_segments_for_subject,
        EEG_ORIGINAL_ROOT, FMRI_ROOT, PROCESSED_ROOT,
        SubjectInfo, SegmentInfo,
    )


def _write_segment_info(seg: SegmentInfo, seg_idx: int) -> Path:
    """Create the segment working directory and write segment_info.json."""
    seg.segment_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "segment_idx":     seg_idx,
        "segment_name":    seg.name,
        "t_start_sec":     float(seg.t_start_sec),
        "t_stop_sec":      float(seg.t_stop_sec),
        "duration_sec":    float(seg.duration_sec),
        "raw_vhdr":        str(seg.vhdr_path.resolve()),
        "sfreq":           float(seg.sfreq),
        "rp_file":         str(seg.rp_path.resolve()) if seg.rp_path else None,
        "rp_path":         str(seg.rp_path.resolve()) if seg.rp_path else None,
        "needs_bergen":    seg.needs_bergen,
        "session_suffix":  seg.session_suffix,
    }
    out = seg.segment_dir / "segment_info.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    return out


def run_step00(subjects: list[SubjectInfo] | None = None,
               subject_filter: str | None = None) -> dict[str, list[Path]]:
    """
    Create segment_info.json for every segment of every subject.

    Parameters
    ----------
    subjects : pre-built subject list (from discover_subjects); if None, auto-discovers.
    subject_filter : if given, only process this subject ID.

    Returns
    -------
    dict mapping subject_id → list of segment_info.json paths created/updated.
    """
    if subjects is None:
        subjects = discover_subjects()

    if subject_filter:
        subjects = [s for s in subjects if s.subject_id == subject_filter]

    results: dict[str, list[Path]] = {}

    for subj in subjects:
        sid = subj.subject_id
        print("=" * 70)
        print(f"[STEP 00] Subject {sid}: {len(subj.segments)} segments")
        print("=" * 70)

        written: list[Path] = []
        for idx, seg in enumerate(subj.segments, start=1):
            out_path = _write_segment_info(seg, idx)
            print(f"  [{idx:02d}] {seg.name:12s} "
                  f"[{seg.t_start_sec:.1f}s – {seg.t_stop_sec:.1f}s] "
                  f"({seg.duration_sec:.0f}s)  "
                  f"Bergen={'YES' if seg.needs_bergen else 'no ':3}  "
                  f"→ {out_path}")
            written.append(out_path)

        results[sid] = written

    total = sum(len(v) for v in results.values())
    print(f"\n[STEP 00] Done. Created/updated {total} segment_info.json files.")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Step 00: Build segment_info.json for all subjects/segments"
    )
    parser.add_argument("--subject", default=None,
                        help="Limit to a single subject ID (e.g. 1916)")
    args = parser.parse_args()
    run_step00(subject_filter=args.subject)
