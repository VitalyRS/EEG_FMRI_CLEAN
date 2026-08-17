"""
End-to-End Master Pipeline Runner for Optuna-Bergen Cleaning
============================================================
Runs all steps 01-07 sequentially or individually:
  1. Detect MRI sessions
  2. Detect slices & generate R128 markers
  3. Trim dummy scans
  4. Optuna Bayesian Optimization
  5. Full 96-channel Bergen AAS cleaning
  6. Compute spectra & check optional EEG21
  7. Generate HTML & PNG reports
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
except ImportError:
    from config import DEFAULT_SEGMENT_DIR, DEFAULT_RAW_VHDR, DEFAULT_N_TRIALS
    from step01_detect_mri import detect_mri_sessions
    from step02_detect_slices import run_detect_slices
    from step03_trim_dummy import trim_dummy_scans
    from step04_optuna_tune import run_optuna_tuning
    from step05_bergen_clean import clean_full_dataset
    from step06_spectra_analysis import compute_spectra
    from step07_html_report import generate_html_report


def main():
    parser = argparse.ArgumentParser(description="End-to-End Optuna-Bergen EEG-fMRI Pipeline")
    parser.add_argument("--segment-dir", type=Path, default=DEFAULT_SEGMENT_DIR, help="Path to segment folder")
    parser.add_argument("--vhdr-raw", type=Path, default=DEFAULT_RAW_VHDR, help="Path to continuous raw .vhdr")
    parser.add_argument("--trials", type=int, default=DEFAULT_N_TRIALS, help="Number of Optuna trials")
    parser.add_argument("--skip-detect-mri", action="store_true", help="Skip Step 01 (session detection)")
    parser.add_argument("--skip-optuna", action="store_true", help="Skip Step 04 (use existing/default params)")
    args = parser.parse_args()

    seg_dir = args.segment_dir.resolve()
    print("=" * 75)
    print("  OPTUNA-BERGEN EEG-fMRI PIPELINE (END-TO-END)")
    print(f"  Target Segment: {seg_dir}")
    print("=" * 75)

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
        print("[INFO] Step 04 (Optuna) skipped, using existing or default parameters.")

    # Step 05
    clean_full_dataset(seg_dir)

    # Step 06
    compute_spectra(seg_dir)

    # Step 07
    html_path = generate_html_report(seg_dir)

    print("\n" + "=" * 75)
    print("  ALL PIPELINE STEPS FINISHED SUCCESSFULLY!")
    print(f"  Final HTML Report: {html_path.resolve()}")
    print("=" * 75)


if __name__ == "__main__":
    main()
