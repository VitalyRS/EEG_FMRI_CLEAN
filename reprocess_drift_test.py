#!/usr/bin/env python3
"""
Reprocess one segment with updated FILTER_HP = 1.0 Hz to test drift removal.
Tests on the first segment from batch_timeseries.json.
"""
import json
from pathlib import Path
import sys

# Get first segment from timeseries
ts_path = Path('/tmp/batch_timeseries.json')
if not ts_path.exists():
    print("ERROR: /tmp/batch_timeseries.json not found")
    print("Run: python3 extract_timeseries.py first")
    sys.exit(1)

ts = json.load(open(ts_path))
if not ts['segments']:
    print("ERROR: No segments in timeseries.json")
    sys.exit(1)

seg_info = ts['segments'][0]
subj = seg_info['subj']
seg = seg_info['seg']

print(f"Reprocessing {subj}/{seg} with FILTER_HP=1.0 Hz...")
print(f"Pipeline: step08 (BCG) → step10 (Optuna+ICA) → step11 (final ICA)")
print()

# Import pipeline functions
from step08_bcg import run_bcg_pipeline
from step10_optuna_ica import run_optuna_ica
from step11_ica_final import apply_optimized_ica

segment_dir = Path('data') / subj / 'segments' / seg

print(f"=== STEP 08: BCG (with HP 1.0 Hz) ===")
run_bcg_pipeline(segment_dir, force=True)

print(f"\n=== STEP 10: Optuna ICA ===")
run_optuna_ica(segment_dir, force=True)

print(f"\n=== STEP 11: Final ICA ===")
apply_optimized_ica(segment_dir, force=True)

print(f"\n✅ Reprocessing complete: {subj}/{seg}")
print(f"\nNow re-extract timeseries and regenerate report:")
print(f"  python3 extract_timeseries.py")
print(f"  python3 generate_report.py")
print(f"  xdg-open /tmp/batch_report.html")
