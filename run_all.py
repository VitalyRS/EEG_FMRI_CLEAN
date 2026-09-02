"""
End-to-End Master Pipeline Runner for Optuna-Bergen Cleaning + ICA
===================================================================
Runs the pipeline sequentially. By default each step is SKIPPED if its output
already exists (incremental); --recalc wipes everything first for a clean rerun.

   1. Detect MRI sessions
   2. Detect slices & generate R128 markers
   3. Trim dummy scans
   4. Optuna Bayesian Optimization (Bergen AAS)
   5. Full 96-channel Bergen AAS cleaning
   6. Compute spectra & check optional EEG21
   7. Generate Bergen HTML & PNG report
   8. BCG artifact removal (OBS)
  10. Optuna ICA parameter optimization
  11. Apply ICA with optimized parameters (sole writer of derivatives/05_ica)
  12. Aggregate before/after summary report for the segment

Note: the former step 09 (preliminary ICA) was removed — it recomputed ICA with
default params into the same files step 11 overwrites, so it was a dead branch.
"""
import argparse
import shutil
from pathlib import Path

try:
    from .config import (DEFAULT_RAW_VHDR, DEFAULT_N_TRIALS,
                         PROJECT_ROOT, DATA_ROOT, DEFAULT_EXPERIMENT)
    from .step01_detect_mri import detect_mri_sessions, discover_all_eeg_vhdrs
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
except ImportError:
    from config import (DEFAULT_RAW_VHDR, DEFAULT_N_TRIALS,
                        PROJECT_ROOT, DATA_ROOT, DEFAULT_EXPERIMENT)
    from step01_detect_mri import detect_mri_sessions, discover_all_eeg_vhdrs
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


def _is_done(markers) -> bool:
    """A step is 'done' if ALL its output markers already exist on disk.

    `markers` is a single Path or a list of Paths. Globs are supported per-marker
    via a (parent_dir, pattern) tuple — matches if at least one file matches.
    """
    if not isinstance(markers, (list, tuple)) or (
        len(markers) == 2 and isinstance(markers[0], Path) and isinstance(markers[1], str)
    ):
        markers = [markers]
    for m in markers:
        if isinstance(m, tuple):  # (parent_dir, glob_pattern)
            parent, pat = m
            if not (parent.exists() and any(parent.glob(pat))):
                return False
        else:
            if not Path(m).exists():
                return False
    return True


def clean_all_derivatives(seg_dir: Path):
    """
    Force a full recompute: delete every computed artifact for this segment while
    preserving raw inputs. Removes:
      - all derivatives/<stage>/<segment>/ subdirs (Bergen, resampled, BCG, channels, ICA, final)
      - all qc/<subject>/<stage>/<segment>* plots
      - Optuna study DBs, best-param JSONs, trial folders, .set files, reports in the segment dir
    Raw recordings (data/<subject>/raw/) and pipeline code are never touched.
    """
    seg = seg_dir.name
    subject = (seg_dir.parent.parent.name if seg_dir.parent.name == "segments"
               else seg_dir.parent.name)
    print("=" * 80)
    print(f"  [--recalc] FORCING FULL RECOMPUTE for segment '{subject}/{seg}'")
    print("=" * 80)

    removed = []

    # 1. derivatives/<stage>/<segment>/
    deriv_root = DATA_ROOT / subject / "derivatives"
    if deriv_root.exists():
        for stage_dir in sorted(deriv_root.iterdir()):
            if not stage_dir.is_dir():
                continue
            target = stage_dir / seg
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
                removed.append(str(target.relative_to(PROJECT_ROOT)))

    # 2. qc/<subject>/<stage>/<segment>* (plots may be files, not dirs)
    qc_root = PROJECT_ROOT / "qc" / subject
    if qc_root.exists():
        for stage_dir in sorted(qc_root.iterdir()):
            if not stage_dir.is_dir():
                continue
            for item in stage_dir.glob(f"{seg}*"):
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
                removed.append(str(item.relative_to(PROJECT_ROOT)))

    # 3. Segment working dir: every COMPUTED artifact (raw inputs are elsewhere,
    #    under data/<subject>/raw/, and are never touched).
    if seg_dir.exists():
        patterns = [
            # step01-03 computed metadata (these are the skip-markers for 01-03)
            "segment_info.json", "slice_detection.json", "segment_work_info.json",
            "slice_triggers.txt", "slice_phase_check.png",
            # step04 Optuna Bergen: study DB, trials, best params
            "*.db", "*optuna*", "trial*", "*_best_params.json",
            # step05 Bergen dataset + its MATLAB scratch
            "*.set", "*.fdt", "temp_multich_crop.mat", "run_full_optuna_clean.m",
            # step06/07 spectra + Bergen report
            "*.npz", "*.png", "*.csv", "summary_*.csv", "*_report.html",
            # step10/11 ICA scratch + params
            "ica_optuna_best.json", "ica_final",
        ]
        seen = set()
        for pat in patterns:
            for item in seg_dir.glob(pat):
                if item in seen:
                    continue
                seen.add(item)
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
                removed.append(str(item.relative_to(PROJECT_ROOT)))

    if removed:
        print(f"  Deleted {len(removed)} artifact path(s):")
        for r in sorted(removed):
            print(f"    - {r}")
    else:
        print("  Nothing to delete (already clean).")
    print("  Raw inputs and pipeline code preserved.")
    print("=" * 80 + "\n")


# Canonical processing order for named segments. Segments not listed here are
# appended afterwards in alphabetical order, so unknown names still get run.
SEGMENT_ORDER = ["eo", "ec", "drone", "lasertag", "video"]


def _order_segments(dirs):
    """Sort segment dirs by SEGMENT_ORDER, unknown names last (alphabetical)."""
    def key(d):
        try:
            return (0, SEGMENT_ORDER.index(d.name))
        except ValueError:
            return (1, d.name)
    return sorted(dirs, key=key)


def _discover_segments(args) -> list:
    """Resolve the list of segment dirs to process from CLI flags.

    Priority: --segment-dir (explicit single) > --subject (all segments of one
    subject) > --all (every segment of every subject under data/). With no
    selection flag we fall back to the default subject (config.DEFAULT_EXPERIMENT)
    and process all of its segments. A valid segment dir is one that contains
    (or will contain) segment_info.json — an immediate child of a
    data/<subject>/segments/ folder. Named segments are processed in
    SEGMENT_ORDER (eo, ec, drone, lasertag, video).
    """
    # Explicit single segment always wins.
    if args.segment_dir is not None:
        return [args.segment_dir.resolve()]

    # --all across every subject, --subject for one, else the default subject.
    if args.all:
        subjects = [d.name for d in sorted(DATA_ROOT.glob("*")) if d.is_dir()]
    elif args.subject:
        subjects = [args.subject]
    else:
        subjects = [DEFAULT_EXPERIMENT]

    seg_dirs = []
    for sid in subjects:
        seg_root = DATA_ROOT / sid / "segments"
        if not seg_root.exists():
            continue
        subject_dirs = [d.resolve() for d in seg_root.iterdir() if d.is_dir()]
        seg_dirs.extend(_order_segments(subject_dirs))
    return seg_dirs


def run_segment(seg_dir: Path, args):
    """Run the full 01->12 pipeline for a single segment directory."""
    print("=" * 80)
    print("  OPTUNA-BERGEN EEG-fMRI PIPELINE + BCG + ICA (END-TO-END)")
    print(f"  Target Segment: {seg_dir}")
    print("=" * 80)

    if args.recalc:
        clean_all_derivatives(seg_dir)

    # ---- Output markers used to skip already-computed steps -----------------
    # By default a step is skipped if its marker(s) already exist (incremental).
    # --recalc deletes everything above, so all markers are gone -> full rerun.
    # Subject is derived from the segment path (data/<subject>/segments/<seg>),
    # so this works for any subject, not just DEFAULT_EXPERIMENT.
    seg = seg_dir.name
    subject = (seg_dir.parent.parent.name if seg_dir.parent.name == "segments"
               else seg_dir.parent.name)
    deriv = DATA_ROOT / subject / "derivatives"
    M = {
        "01": seg_dir / "segment_info.json",
        "02": seg_dir / "slice_detection.json",
        "03": seg_dir / "segment_work_info.json",
        "04": seg_dir / "optuna_best_params.json",
        "05": (seg_dir, "*bergen*.set"),
        "06": seg_dir / "step03_spectra_data.npz",
        "07": seg_dir / f"{seg}_cleaning_report.html",
        "08": deriv / "03_bcg" / seg / f"{seg}_bcg_clean.fif",
        # step10 optimizes ICA params; step11 applies them and is the ONLY writer
        # of 05_ica (step09 was removed from the pipeline as a dead branch).
        "10": seg_dir / "ica_optuna_best.json",
        "11": deriv / "05_ica" / seg / f"{seg}_ica_clean.fif",
    }

    def _skip(step, reason="already computed"):
        print(f"[SKIP] Step {step}: {reason} (use --recalc to force).")

    # Step 01: detect MRI sessions & write segment_info.json for every session
    # of THIS subject. detect_mri_sessions writes all segments at once, so the
    # marker for any one segment implies step01 already ran for the subject.
    if args.skip_detect_mri:
        print("[INFO] Step 01 skipped by flag.")
    elif _is_done(M["01"]):
        _skip("01")
    else:
        # Resolve the raw vhdr for THIS subject (not the global default).
        vhdr = None
        if args.vhdr_raw and args.vhdr_raw.exists() and subject in args.vhdr_raw.parts:
            vhdr = args.vhdr_raw  # explicit override matching this subject
        else:
            vhdr = discover_all_eeg_vhdrs().get(subject)
        if vhdr is None:
            print(f"[INFO] Step 01 skipped: no raw VHDR found for subject {subject}.")
        else:
            detect_mri_sessions(vhdr_path=vhdr, subject_id=subject)

    # Step 02
    if _is_done(M["02"]):
        _skip("02")
    else:
        run_detect_slices(seg_dir)

    # Step 03: trim dummy scans. dummy_volumes=None -> auto-detect from the time
    # alignment between the detected segment span and the fMRI task (rp volumes),
    # so the trimmed EEG window matches the task. A per-run --dummy N forces it.
    if _is_done(M["03"]):
        _skip("03")
    else:
        trim_dummy_scans(seg_dir, dummy_volumes=args.dummy)

    # Step 04
    if args.skip_optuna:
        print("[INFO] Step 04 (Optuna Bergen) skipped by flag, using existing/default params.")
    elif _is_done(M["04"]):
        _skip("04")
    else:
        run_optuna_tuning(seg_dir, n_trials=args.trials)

    # Step 05
    if _is_done(M["05"]):
        _skip("05", "Bergen .set already present")
    else:
        clean_full_dataset(seg_dir)

    # Step 06
    if _is_done(M["06"]):
        _skip("06")
    else:
        compute_spectra(seg_dir)

    # Step 07
    if _is_done(M["07"]):
        _skip("07")
        html_path = M["07"]
    else:
        html_path = generate_html_report(seg_dir)
    print(f"\n[REPORT] Bergen HTML: {Path(html_path).resolve()}")

    # Step 08: BCG artifact removal
    if args.skip_bcg:
        print("[INFO] Step 08 (BCG) skipped by flag.")
    else:
        print("\n" + "=" * 80)
        print("  STARTING BCG ARTIFACT REMOVAL")
        print("=" * 80)
        if _is_done(M["08"]):
            _skip("08", "BCG-clean .fif already present")
        else:
            run_bcg_pipeline(seg_dir)
        print("[INFO] Step 08 (BCG) complete.")

    # Step 10-11: ICA optimization and application
    if args.skip_ica:
        print("[INFO] Steps 10-11 (ICA) skipped by flag.")
    else:
        print("\n" + "=" * 80)
        print("  STARTING ICA PIPELINE")
        print("=" * 80)
        if args.skip_ica_optuna:
            print("[INFO] Step 10 (Optuna ICA) skipped by flag, using existing/default params.")
        elif _is_done(M["10"]):
            _skip("10", "ICA Optuna params already present")
        else:
            run_optuna_ica(seg_dir, n_trials=args.trials)

        if _is_done(M["11"]):
            _skip("11", "final ICA already applied")
        else:
            apply_optimized_ica(seg_dir)
        print("[INFO] Steps 10-11 (ICA) complete.")

    # Step 12: aggregate before/after summary report for the whole segment
    print("\n" + "=" * 80)
    print("  GENERATING SEGMENT SUMMARY REPORT (before/after each stage)")
    print("=" * 80)
    summary_html = None
    try:
        # On --recalc every upstream artifact was recomputed, so the summary must
        # be rebuilt too (it lives outside seg_dir and is not wiped by --recalc).
        summary_html = generate_summary_report(seg_dir, force=args.recalc)
    except Exception as ex:
        print(f"[WARN] Step 12 summary report failed: {ex}")

    print("\n" + "=" * 80)
    print(f"  ALL PIPELINE STEPS FINISHED SUCCESSFULLY! ({subject}/{seg})")
    print(f"  Bergen Report: {Path(html_path).resolve()}")
    if not args.skip_ica:
        ica_report = deriv / "05_ica" / seg / f"{seg}_ica_report.html"
        if ica_report.exists():
            print(f"  ICA Report:    {ica_report.resolve()}")
    if summary_html:
        print(f"  SUMMARY:       {summary_html.resolve()}")
    print("=" * 80)
    return summary_html


def main():
    parser = argparse.ArgumentParser(description="End-to-End Optuna-Bergen EEG-fMRI Pipeline + ICA")
    parser.add_argument("--segment-dir", type=Path, default=None,
                        help="Path to a single segment folder. If omitted, all segments of "
                             "--subject (or the default subject) are processed in order "
                             "(eo, ec, drone, lasertag, video).")
    parser.add_argument("--subject", default=None, help="Process every segment of this subject (e.g. 1916)")
    parser.add_argument("--all", action="store_true", help="Process every segment of every subject under data/")
    parser.add_argument("--vhdr-raw", type=Path, default=DEFAULT_RAW_VHDR, help="Path to continuous raw .vhdr (Step 01, single-segment mode only)")
    parser.add_argument("--trials", type=int, default=DEFAULT_N_TRIALS, help="Number of Optuna trials (Bergen & ICA)")
    parser.add_argument("--dummy", type=int, default=None,
                        help="Force an exact dummy-volume count for Step 03 (e.g. 13). "
                             "Omit to auto-detect per segment from time alignment. "
                             "Only valid for a single segment, not --all/--subject.")
    parser.add_argument("--skip-detect-mri", action="store_true", help="Skip Step 01 (session detection)")
    parser.add_argument("--skip-optuna", action="store_true", help="Skip Step 04 (use existing/default Bergen params)")
    parser.add_argument("--skip-bcg", action="store_true", help="Skip Step 08 (BCG removal)")
    parser.add_argument("--skip-ica-optuna", action="store_true", help="Skip Step 10 (use existing/default ICA params)")
    parser.add_argument("--skip-ica", action="store_true", help="Skip Steps 10-11 (ICA entirely)")
    parser.add_argument("--recalc", action="store_true",
                        help="Delete ALL computed artifacts for each segment first, forcing a full recompute (raw inputs are preserved)")
    args = parser.parse_args()

    if args.dummy is not None and (args.all or args.subject):
        parser.error("--dummy forces one dummy count and cannot be combined with "
                     "--all/--subject (each segment needs its own). Run a single "
                     "--segment-dir, or omit --dummy to auto-detect per segment.")

    seg_dirs = _discover_segments(args)
    if not seg_dirs:
        print("[ERROR] No segment directories found. Check --subject/--all or run step01 first.")
        return

    multi = len(seg_dirs) > 1
    if multi:
        print("#" * 80)
        print(f"  BATCH RUN: {len(seg_dirs)} segment(s)")
        for s in seg_dirs:
            print(f"    - {s.relative_to(DATA_ROOT) if DATA_ROOT in s.parents else s}")
        print("#" * 80 + "\n")

    ok, failed = [], []
    for seg_dir in seg_dirs:
        try:
            run_segment(seg_dir, args)
            ok.append(seg_dir)
        except Exception as ex:
            failed.append((seg_dir, ex))
            print("\n" + "!" * 80)
            print(f"  [FAILED] Segment {seg_dir} raised: {ex}")
            print("  Continuing with the next segment." if multi else "")
            print("!" * 80 + "\n")
            if not multi:
                raise

    if multi:
        print("\n" + "#" * 80)
        print(f"  BATCH DONE — {len(ok)} ok, {len(failed)} failed")
        for s in ok:
            print(f"    [OK]     {s.name}")
        for s, ex in failed:
            print(f"    [FAILED] {s.name}: {ex}")
        print("#" * 80)


if __name__ == "__main__":
    main()
