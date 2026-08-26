"""
pipeline_multi.py — Horizontal Multi-Subject EEG-fMRI Cleaning Pipeline
========================================================================
Processes ALL subjects × ALL segments step-by-step in a "horizontal" order:
  Step 00 for every subject/segment  →  Step 02 for every …  →  etc.

This means all subjects complete Step N before any subject starts Step N+1,
which is ideal for batch review at each stage.

Pipeline steps
--------------
00  Segment mapping  (vmrk + rp → segment_info.json)
02  Slice detection  (gradient burst phase)
03  Trim dummy scans / generate slice_triggers.txt
04  Optuna Bergen AAS parameter search   [only if needs_bergen]
05  Bergen AAS full cleaning             [only if needs_bergen]
06  Spectral analysis                    [only if needs_bergen]
07  Bergen HTML report                   [only if needs_bergen]
08  BCG artifact removal (OBS)
10  Optuna ICA parameter search
11  Apply ICA with optimized parameters
12  Summary report

Usage
-----
# Dry-run: show what would be processed without running anything
python pipeline_multi.py --dry-run

# Run all steps for all subjects
python pipeline_multi.py

# Run only a specific step for all subjects
python pipeline_multi.py --step 02

# Run all steps for one subject only
python pipeline_multi.py --subject 1916

# Force full recompute (wipe outputs for all subjects/steps)
python pipeline_multi.py --recalc

# Skip Bergen or ICA globally
python pipeline_multi.py --skip-bergen
python pipeline_multi.py --skip-ica

# Combine: just Bergen steps for one subject
python pipeline_multi.py --subject 1916 --step 04
python pipeline_multi.py --subject 1916 --step 05

Notes
-----
- On restart, the pipeline automatically re-discovers any new EEG subjects
  (new XXXXEEG folders) and includes them from Step 00.
- Each step is skipped if its output marker already exists (incremental).
- Use --recalc to force a full recompute from scratch.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Optional

# ---- Import segment mapper and steps --------------------------------------
try:
    from .segment_mapper import (
        discover_subjects, SubjectInfo, SegmentInfo,
        EEG_ORIGINAL_ROOT, FMRI_ROOT, PROCESSED_ROOT,
    )
    from .step00_segment_map import run_step00
    from .step02_detect_slices import run_detect_slices
    from .step03_trim_dummy import trim_dummy_scans
    from .step04_optuna_tune import run_optuna_tuning
    from .step05_bergen_clean import clean_full_dataset
    from .step06_spectra_analysis import compute_spectra
    from .step07_html_report import generate_html_report
    from .step08_bcg import run_bcg_pipeline
    from .step10_optuna_ica import run_optuna_ica
    from .step11_ica_final import apply_optimized_ica
    from .step12_summary_report import generate_summary_report
    from .config import DEFAULT_TR_SEC, DEFAULT_SLICES_PER_VOLUME, DEFAULT_N_TRIALS
except ImportError:
    from segment_mapper import (
        discover_subjects, SubjectInfo, SegmentInfo,
        EEG_ORIGINAL_ROOT, FMRI_ROOT, PROCESSED_ROOT,
    )
    from step00_segment_map import run_step00
    from step02_detect_slices import run_detect_slices
    from step03_trim_dummy import trim_dummy_scans
    from step04_optuna_tune import run_optuna_tuning
    from step05_bergen_clean import clean_full_dataset
    from step06_spectra_analysis import compute_spectra
    from step07_html_report import generate_html_report
    from step08_bcg import run_bcg_pipeline
    from step10_optuna_ica import run_optuna_ica
    from step11_ica_final import apply_optimized_ica
    from step12_summary_report import generate_summary_report
    from config import DEFAULT_TR_SEC, DEFAULT_SLICES_PER_VOLUME, DEFAULT_N_TRIALS


# ---------------------------------------------------------------------------
# Skip/done markers
# ---------------------------------------------------------------------------

def _is_done(markers) -> bool:
    """True if ALL marker paths exist on disk. Supports Path or (dir, glob) tuples."""
    if not isinstance(markers, (list, tuple)) or (
        len(markers) == 2
        and isinstance(markers[0], Path)
        and isinstance(markers[1], str)
    ):
        markers = [markers]
    for m in markers:
        if isinstance(m, tuple):
            parent, pat = m
            if not (parent.exists() and any(parent.glob(pat))):
                return False
        else:
            if not Path(m).exists():
                return False
    return True


def _markers_for(step: str, seg: SegmentInfo) -> object:
    """Return the output marker(s) for a given step and segment."""
    d = seg.segment_dir
    deriv = seg.segment_dir.parent.parent / "derivatives"  # segments/../derivatives
    seg_name = seg.name

    return {
        "00": d / "segment_info.json",
        "02": d / "slice_detection.json",
        "03": d / "segment_work_info.json",
        "04": d / "optuna_best_params.json",
        "05": (d, "*bergen*.set"),
        "06": d / "step03_spectra_data.npz",
        "07": d / f"{seg_name}_cleaning_report.html",
        "08": deriv / "03_bcg" / seg_name / f"{seg_name}_bcg_clean.fif",
        "10": d / "ica_optuna_best.json",
        "11": deriv / "05_ica" / seg_name / f"{seg_name}_ica_clean.fif",
        "12": d / f"{seg_name}_summary_report.html",
    }.get(step)


# ---------------------------------------------------------------------------
# Recalc: wipe outputs for a segment
# ---------------------------------------------------------------------------

def _recalc_segment(seg: SegmentInfo) -> None:
    """Delete all computed artifacts for this segment (preserve segment_info.json)."""
    d = seg.segment_dir
    deriv = d.parent.parent / "derivatives"
    seg_name = seg.name

    print(f"  [--recalc] Wiping outputs for {seg.subject_id}/{seg_name}")

    # Remove derivative subdirs
    for stage in ["01_bergen", "02_resamp", "03_bcg", "04_channels", "05_ica", "06_final"]:
        tgt = deriv / stage / seg_name
        if tgt.exists():
            shutil.rmtree(tgt, ignore_errors=True)
            print(f"    - removed {tgt}")

    # Remove computed files in segment dir (keep segment_info.json)
    patterns = [
        "slice_detection.json", "segment_work_info.json",
        "slice_triggers.txt", "slice_phase_check.png",
        "*.db", "*optuna*", "trial*", "*_best_params.json",
        "*.set", "*.fdt", "temp_multich_crop.mat", "run_full_optuna_clean.m",
        "*.npz", "*.png", "*.csv", "*_report.html",
        "ica_optuna_best.json", "ica_final",
    ]
    seen: set[Path] = set()
    for pat in patterns:
        for item in d.glob(pat):
            if item in seen or item.name == "segment_info.json":
                continue
            seen.add(item)
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

def _run_step(step: str, seg: SegmentInfo,
              n_trials: int = DEFAULT_N_TRIALS,
              skip_bergen: bool = False,
              skip_ica: bool = False,
              skip_ica_optuna: bool = False,
              dry_run: bool = False) -> bool:
    """
    Execute one pipeline step for one segment.
    Returns True on success, False on skip or error.
    """
    label = f"{seg.subject_id}/{seg.name}"

    # --- Bergen steps: only for segments with MRI artifact + rp file --------
    bergen_steps = {"04", "05", "06", "07"}
    if step in bergen_steps:
        if not seg.needs_bergen:
            print(f"  [SKIP] {label} step {step}: outside recording, no Bergen needed")
            return True
        if seg.rp_path is None:
            print(f"  [SKIP] {label} step {step}: no rp file → Bergen skipped")
            return True
        if skip_bergen:
            print(f"  [SKIP] {label} step {step}: --skip-bergen")
            return True

    # --- ICA steps -----------------------------------------------------------
    if step in {"10", "11"} and skip_ica:
        print(f"  [SKIP] {label} step {step}: --skip-ica")
        return True
    if step == "10" and skip_ica_optuna:
        print(f"  [SKIP] {label} step {step}: --skip-ica-optuna")
        return True

    # --- Already done? -------------------------------------------------------
    marker = _markers_for(step, seg)
    if marker is not None and _is_done(marker):
        print(f"  [DONE] {label} step {step}: already computed, skipping "
              "(use --recalc to force)")
        return True

    # --- Dry run -------------------------------------------------------------
    if dry_run:
        print(f"  [DRY-RUN] would run step {step} for {label}")
        return True

    # --- Execute -------------------------------------------------------------
    try:
        d = seg.segment_dir
        if step == "02":
            run_detect_slices(d, tr_sec=DEFAULT_TR_SEC,
                              slices_per_volume=DEFAULT_SLICES_PER_VOLUME)
        elif step == "03":
            trim_dummy_scans(d, tr_sec=DEFAULT_TR_SEC,
                             slices_per_volume=DEFAULT_SLICES_PER_VOLUME)
        elif step == "04":
            run_optuna_tuning(d, n_trials=n_trials)
        elif step == "05":
            clean_full_dataset(d)
        elif step == "06":
            compute_spectra(d)
        elif step == "07":
            generate_html_report(d)
        elif step == "08":
            run_bcg_pipeline(d)
        elif step == "10":
            run_optuna_ica(d, n_trials=n_trials)
        elif step == "11":
            apply_optimized_ica(d)
        elif step == "12":
            generate_summary_report(d)
        else:
            print(f"  [WARN] Unknown step {step}")
            return False
        return True

    except Exception as exc:
        print(f"\n  [ERROR] step {step} failed for {label}: {exc}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Ordered step list (step 00 handled separately via run_step00)
# ---------------------------------------------------------------------------

STEP_ORDER = ["02", "03", "04", "05", "06", "07", "08", "10", "11", "12"]

STEP_DESCRIPTIONS = {
    "00": "Segment mapping (vmrk → segment_info.json)",
    "02": "Slice phase detection",
    "03": "Trim dummy scans / slice_triggers.txt",
    "04": "Optuna Bergen AAS tuning",
    "05": "Bergen AAS full cleaning",
    "06": "Spectral analysis",
    "07": "Bergen HTML report",
    "08": "BCG artifact removal",
    "10": "Optuna ICA parameter search",
    "11": "Apply optimized ICA",
    "12": "Summary report",
}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Horizontal multi-subject EEG-fMRI pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--subject", default=None,
                        help="Process only this subject ID (e.g. 1916)")
    parser.add_argument("--step", default=None,
                        help="Run only this pipeline step (e.g. 02, 05, 08)")
    parser.add_argument("--trials", type=int, default=DEFAULT_N_TRIALS,
                        help="Number of Optuna trials (default: %(default)s)")
    parser.add_argument("--skip-bergen", action="store_true",
                        help="Skip steps 04-07 (Bergen AAS)")
    parser.add_argument("--skip-ica", action="store_true",
                        help="Skip steps 10-11 (ICA)")
    parser.add_argument("--skip-ica-optuna", action="store_true",
                        help="Skip step 10 (use existing/default ICA params)")
    parser.add_argument("--recalc", action="store_true",
                        help="Wipe computed outputs and recompute from scratch")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without running anything")
    args = parser.parse_args()

    # ---- Banner -------------------------------------------------------------
    print("=" * 72)
    print("  EEG-fMRI MULTI-SUBJECT HORIZONTAL PIPELINE")
    print(f"  EEG root:       {EEG_ORIGINAL_ROOT}")
    print(f"  fMRI root:      {FMRI_ROOT}")
    print(f"  Output root:    {PROCESSED_ROOT}")
    if args.subject:
        print(f"  Subject filter: {args.subject}")
    if args.step:
        print(f"  Step filter:    {args.step}")
    if args.dry_run:
        print("  ** DRY-RUN MODE — no changes will be made **")
    print("=" * 72)

    # ---- Discover subjects --------------------------------------------------
    print("\n[Discovering subjects…]")
    subjects = discover_subjects()
    if args.subject:
        subjects = [s for s in subjects if s.subject_id == args.subject]
    if not subjects:
        print("No subjects found. Check EEG_ORIGINAL_ROOT path.")
        sys.exit(1)

    all_segments = [seg for subj in subjects for seg in subj.segments]
    print(f"Found {len(subjects)} subject(s), {len(all_segments)} total segments.\n")

    # ---- Step 00: segment mapping -------------------------------------------
    if args.step is None or args.step == "00":
        hdr = "Step 00 — " + STEP_DESCRIPTIONS["00"]
        print("\n" + "=" * 72)
        print(f"  {hdr}")
        print("=" * 72)
        if args.dry_run:
            for seg in all_segments:
                print(f"  [DRY-RUN] would write segment_info.json → {seg.segment_dir}")
        else:
            for subj in subjects:
                # Re-discover and re-write even if already done (idempotent)
                from step00_segment_map import run_step00 as _step00
                _step00(subjects=[subj])

        if args.step == "00":
            print("\n[Done. Only step 00 requested — stopping here.]")
            return

    # ---- Steps 02-12: horizontal -------------------------------------------
    steps_to_run = STEP_ORDER if args.step is None else [args.step]
    # Filter out step 00 (already handled above)
    steps_to_run = [s for s in steps_to_run if s != "00"]

    errors: list[str] = []

    for step in steps_to_run:
        hdr = f"Step {step} — {STEP_DESCRIPTIONS.get(step, '?')}"
        print("\n" + "=" * 72)
        print(f"  {hdr}")
        print("=" * 72)

        for seg in all_segments:
            # --recalc: wipe before running
            if args.recalc and not args.dry_run:
                marker = _markers_for(step, seg)
                if marker is not None and _is_done(marker):
                    _recalc_segment(seg)

            label = f"{seg.subject_id}/{seg.name}"
            print(f"\n  → {label}")
            ok = _run_step(
                step, seg,
                n_trials=args.trials,
                skip_bergen=args.skip_bergen,
                skip_ica=args.skip_ica,
                skip_ica_optuna=args.skip_ica_optuna,
                dry_run=args.dry_run,
            )
            if not ok:
                errors.append(f"step {step} / {label}")

    # ---- Final summary ------------------------------------------------------
    print("\n" + "=" * 72)
    if errors:
        print(f"  PIPELINE FINISHED WITH {len(errors)} ERROR(S):")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)
    else:
        print("  ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 72)


if __name__ == "__main__":
    main()
