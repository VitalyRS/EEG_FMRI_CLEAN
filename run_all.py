"""
End-to-End Master Pipeline Runner for Optuna-Bergen Cleaning + ICA
===================================================================
Runs all steps 01-11 sequentially or individually:
  1. Detect MRI sessions
  2. Detect slices & generate R128 markers
  3. Trim dummy scans
  4. Optuna Bayesian Optimization (Bergen AAS)
  5. Full 96-channel Bergen AAS cleaning
  6. Compute spectra & check optional EEG21
  7. Generate HTML & PNG reports (Bergen only)
  8. BCG artifact removal (Optuna)
  9. Apply BCG cleaning
  10. Optuna ICA parameter optimization
  11. Apply ICA with optimized parameters
"""
import argparse
from pathlib import Path

try:
    from .config import DEFAULT_SEGMENT_DIR, DEFAULT_RAW_VHDR, DEFAULT_N_TRIALS
    from .step01_detect_mri import detect_mri_sessions
    from .step02_detect_slices import run_detect_slices
    from .step03_trim_dummy import trim_dummy_scans
    from .step04_optuna_tune import run_optuna_tuning
    from .step05_bergen_clean import clean_full_dataset
    from .step06_spectra_analysis import compute_spectra
    from .step07_html_report import generate_html_report
    from .step08_bcg_optuna import run_bcg_optuna
    from .step09_ica import run_ica_cleaning
    from .step10_optuna_ica import run_optuna_ica
    from .step11_ica_final import apply_optimized_ica
except ImportError:
    from config import DEFAULT_SEGMENT_DIR, DEFAULT_RAW_VHDR, DEFAULT_N_TRIALS
    from step01_detect_mri import detect_mri_sessions
    from step02_detect_slices import run_detect_slices
    from step03_trim_dummy import trim_dummy_scans
    from step04_optuna_tune import run_optuna_tuning
    from step05_bergen_clean import clean_full_dataset
    from step06_spectra_analysis import compute_spectra
    from step07_html_report import generate_html_report
    from step08_bcg_optuna import run_bcg_optuna
    from step09_ica import run_ica_cleaning
    from step10_optuna_ica import run_optuna_ica
    from step11_ica_final import apply_optimized_ica


def main():
    parser = argparse.ArgumentParser(description="End-to-End Optuna-Bergen EEG-fMRI Pipeline + ICA")
    parser.add_argument("--segment-dir", type=Path, default=DEFAULT_SEGMENT_DIR, help="Path to segment folder")
    parser.add_argument("--vhdr-raw", type=Path, default=DEFAULT_RAW_VHDR, help="Path to continuous raw .vhdr")
    parser.add_argument("--trials", type=int, default=DEFAULT_N_TRIALS, help="Number of Optuna trials (Bergen & ICA)")
    parser.add_argument("--skip-detect-mri", action="store_true", help="Skip Step 01 (session detection)")
    parser.add_argument("--skip-optuna", action="store_true", help="Skip Step 04 (use existing/default Bergen params)")
    parser.add_argument("--skip-bcg", action="store_true", help="Skip Steps 08-09 (BCG removal)")
    parser.add_argument("--skip-ica-optuna", action="store_true", help="Skip Step 10 (use existing/default ICA params)")
    parser.add_argument("--skip-ica", action="store_true", help="Skip Steps 10-11 (ICA entirely)")
    args = parser.parse_args()

    seg_dir = args.segment_dir.resolve()
    print("=" * 80)
    print("  OPTUNA-BERGEN EEG-fMRI PIPELINE + BCG + ICA (END-TO-END)")
    print(f"  Target Segment: {seg_dir}")
    print("=" * 80)

    # Step 01
    if not args.skip_detect_mri and args.vhdr_raw.exists():
        detect_mri_sessions(args.vhdr_raw)
    else:
        print("[INFO] Step 01 skipped or raw VHDR not found.")

    # Step 02
    run_detect_slices(seg_dir)

    # Step 03
    trim_dummy_scans(seg_dir)

    # Step 04
    if not args.skip_optuna:
        run_optuna_tuning(seg_dir, n_trials=args.trials)
    else:
        print("[INFO] Step 04 (Optuna Bergen) skipped, using existing or default parameters.")

    # Step 05
    clean_full_dataset(seg_dir)

    # Step 06
    compute_spectra(seg_dir)

    # Step 07
    html_path = generate_html_report(seg_dir)
    print(f"\n[REPORT] Bergen HTML: {html_path.resolve()}")

    # Step 08-09: BCG removal
    if not args.skip_bcg:
        print("\n" + "=" * 80)
        print("  STARTING BCG ARTIFACT REMOVAL")
        print("=" * 80)
        run_bcg_optuna(seg_dir, n_trials=args.trials)
        run_ica_cleaning(seg_dir)
        print("[INFO] Steps 08-09 (BCG) complete.")
    else:
        print("[INFO] Steps 08-09 (BCG) skipped.")

    # Step 10-11: ICA optimization and application
    if not args.skip_ica:
        print("\n" + "=" * 80)
        print("  STARTING ICA PIPELINE")
        print("=" * 80)
        if not args.skip_ica_optuna:
            run_optuna_ica(seg_dir, n_trials=args.trials)
        else:
            print("[INFO] Step 10 (Optuna ICA) skipped, using existing or default parameters.")

        apply_optimized_ica(seg_dir)
        print("[INFO] Steps 10-11 (ICA) complete.")
    else:
        print("[INFO] Steps 10-11 (ICA) skipped.")

    print("\n" + "=" * 80)
    print("  ALL PIPELINE STEPS FINISHED SUCCESSFULLY!")
    print(f"  Bergen Report: {html_path.resolve()}")
    if not args.skip_ica:
        seg = seg_dir.name
        ica_report = seg_dir.parent.parent / "data" / "1916" / "derivatives" / "05_ica" / seg / f"{seg}_ica_report.html"
        if ica_report.exists():
            print(f"  ICA Report:    {ica_report.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
