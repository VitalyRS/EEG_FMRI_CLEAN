# CLAUDE.md

Guidance for working in this repository. Read this before making changes.

## What this project is

A modular EEG-fMRI cleaning pipeline (steps 01–12) that removes MRI gradient
artifact (Bergen AAS), ballistocardiogram / BCG (fMRIB OBS), powerline (50 Hz
notch), does soft in-place ASR burst reconstruction, and full-rank Picard ICA
with ICLabel rejection — while preserving brain rhythms (alpha 8–13 Hz).

Stage **P01** (`cleanP01.py`, steps 01–07): raw 5000 Hz → Bergen AAS → resample
to 250 Hz. Stage **P02** (`cleanP02.py`, steps 08/10/11): band-pass + notch +
BCG → ICA param search → final ICA/ICLabel cleaning. See `README.md` for the
full architecture and run commands.

## Ground rules (important)

- **Language: reply in Russian.** The user works in Russian; match it.
- **MATLAB:** do NOT modify the standard EEGLAB libraries or plugins (fMRIB,
  BrainBeats, ICLabel, Bergen).  **New Python helper scripts are allowed** and are the
  preferred way to add processing.
- **Verify before claiming.** This is a research pipeline; numbers matter. After
  a change, re-run the relevant segment and recompute the comparison against the
  reference (`analysis_dima/compare_dima.py`) before stating an effect. Do not
  report predicted improvements as measured ones.
- **Watch for stale caches.** `analysis_dima/qrs_agreement.py` caches detections
  in `reports/dima_compare/my_qrs_<sid>_<seg>.npy` and returns early if the file
  exists. Delete the cache after changing the detector. `compare_dima.py` reads
  the latest `*_ica_clean.fif`; make sure the segment was actually reprocessed
  (check mtime / `cleanP02_state.json`) before trusting a CSV.

## Layout

- `config.py` — absolute paths (`PROJECT_ROOT`, `DATA_ROOT`, `MATLAB_BIN`,
  `EEGLAB_DIR`, plugin dirs). Overridable via env vars.
- `config/cleanP02.yaml` — all P02 cleaning parameters. The orchestrator injects
  them into the step modules; per-segment / per-subject overrides live under
  `overrides:`. Any key is overridable on the CLI with `--set a.b.c=value`.
- `config/ica_meta_model.json` — the trained logistic meta-model (weights read
  at runtime by step11, no sklearn needed).
- `step08_bcg.py` / `step10_optuna_ica.py` / `step11_ica_final.py` — the P02 work.
- `analysis_dima/` — benchmarking against the reference ("Dima") in `data_dima/`:
  `compare_dima.py` (r, RMSE, PSD, cardiac RMS), `qrs_agreement.py`,
  `train_meta_classifier.py`, etc.
- `data/<sid>/derivatives/{01..05}/<seg>/` — pipeline outputs. Final clean data:
  `05_ica/<seg>/<seg>_ica_clean.fif` (+ `.set`, metrics, html).

## Running P02

```bash
python cleanP02.py --subjects 1916 --segments drone            # one segment
python cleanP02.py --all --force ica_final                     # re-run ICA only (BCG unchanged)
python cleanP02.py --subjects 1916 --set ica_final.rejection_mode=thresholds
python analysis_dima/compare_dima.py --seg drone --sid 1916    # grade vs reference
```

A single segment is ~11–85 min (MATLAB OBS + ICA). Prefer testing one segment
before a cohort run. The orchestrator re-runs a step automatically if its config
section changed (`run.rerun_on_config_change`); downstream steps cascade.

## Key metrics (what "good" means)

Grade cleaning against the reference with `compare_dima.py`, not by eye:
- `ica_cardiac_rms_uV` — residual BCG; lower is better. Reference ≈ 0.5–1.9 µV.
- `ratio_ica_<band>` — my/reference band power; ~1.0 is the target. High
  delta/theta ratio = under-subtracted BCG (its fundamental ~1 Hz sits in delta).
- `r_ica_vs_dima_mean` — spatial-temporal correlation (target ≥ 0.40).

step08 also writes `03_bcg/<seg>/qrs_qc.json` (`rr_cv`, `on_rpeak`,
`template_sharpness`) — a reference-free self-check of QRS detection quality.
`on_rpeak` ≥ 0.80 and `template_sharpness` > 1.0 mean the QRS snap worked.

## Current state & known issues (Oct 2026)

See the "Текущее состояние vs эталон" section in `README.md` for the full
picture. In short:

- **Cardiac residual is still ~2.5–6.7× above the reference.** The gap is in
  **OBS (step08)**, not ICA: OBS suppresses ~73% vs the reference's ~94% from the
  same input. QRS detection and IC count already match the reference.
- **QRS detection is fixed** (snap to R-peak + missed-beat interpolation +
  self-check in step08). Manual peak re-labeling is NOT needed — verified our
  peaks sit on the R-wave as well as the reference's.
- **meta_model vs thresholds give nearly identical results** on the current
  cohort; the meta-model is weak (LOSO AUC ≈ 0.62, trained on surrogate labels).
  Cardiac rejection leans on QRS-synchrony, not the model.
- **IC rejection is capped** at `max(5, int(max_rej_fraction·n))` = 11. Cardiac
  ICs now get a priority score so they survive the cap (fixed in step11).
- **ECG-guided detector rarely fires** — residual BCG after OBS is a smeared
  ballistocardiographic trace, poorly matched by linear correlation to raw ECG
  (max corr ≈ 0.25 vs 0.25 threshold). QRS-locking is the reliable channel.

