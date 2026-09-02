"""
Batch: rerun step08 (BCG, new 1.0-80 Hz band) + step10 (Optuna + Phase-ASR) +
step11 (final ICA) for 1916 & 1925, segments ec/eo/drone/lasertag, force=True.
Prints a clear per-segment marker so a monitor can track progress.

HP raised 0.5 -> 1.0 Hz to remove slow baseline drift, so ALL segments must be
reprocessed (nothing is skipped: prior runs used the old 0.5 Hz filter).
"""
import sys, time, traceback
from pathlib import Path
from step08_bcg import run_bcg_pipeline
from step10_optuna_ica import run_optuna_ica
from step11_ica_final import apply_optimized_ica

SUBJECTS = ["1916", "1925"]
SEGMENTS = ["ec", "eo", "drone", "lasertag"]
SKIP = set()  # reprocess ALL: prior 1916/drone run used the old 0.5 Hz filter

root = Path("data")
ok, failed = [], []
for subj in SUBJECTS:
    for seg in SEGMENTS:
        if (subj, seg) in SKIP:
            print(f"[BATCH] SKIP {subj}/{seg} (already done)", flush=True)
            continue
        sd = (root / subj / "segments" / seg).resolve()
        tag = f"{subj}/{seg}"
        t0 = time.time()
        print(f"[BATCH] >>> START {tag}", flush=True)
        try:
            run_bcg_pipeline(sd, force=True)
            print(f"[BATCH] {tag} step08 OK", flush=True)
            run_optuna_ica(sd, force=True)
            print(f"[BATCH] {tag} step10 OK", flush=True)
            apply_optimized_ica(sd, force=True)
            print(f"[BATCH] {tag} step11 OK", flush=True)
            dt = time.time() - t0
            print(f"[BATCH] <<< DONE {tag} ({dt/60:.1f} min)", flush=True)
            ok.append(tag)
        except Exception as e:
            print(f"[BATCH] !!! FAILED {tag}: {e}", flush=True)
            traceback.print_exc()
            failed.append(tag)

print(f"[BATCH] ===== COMPLETE. ok={ok} failed={failed} =====", flush=True)
