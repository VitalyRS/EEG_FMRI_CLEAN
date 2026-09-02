# 🧠 Modular EEG-fMRI Cleaning Pipeline: Step 01 to Step 12

**High-precision modular software pipeline for concurrent EEG-fMRI gradient artifact removal (Bergen AAS), BCG (ballistocardiogram) OBS cleaning, and ICA with ICLabel auto-classification and quantitative alpha rhythm preservation control.**

---

## 📋 Overview of Pipeline Architecture

The pipeline processes raw high-frequency EEG data recorded inside an MRI scanner (5000 Hz) through sequential, automated, and memory-safe cleaning stages:

1. **Stage 1: MRI Gradient Artifact Removal (Steps 01–07)**
   - Automatic fMRI session boundary detection
   - Sub-sample slice phase offset identification
   - Dummy scan trimming & SPM motion (`rp_*.txt`) alignment
   - Bayesian hyperparameter tuning (Optuna TPE) for Bergen AAS
   - Full 96-channel Bergen Average Artifact Subtraction (AAS)
   - Spectral PSD analysis & Bergen HTML QC reporting
2. **Stage 2: Ballistocardiogram (BCG) Artifact Removal (Step 08)**
   - Resampling from 5000 Hz to 250 Hz (RAM footprint reduced to ~28 MB per segment)
   - MATLAB fMRIB QRS peak detection
   - Optimal Principal Component (OBS) selection with alpha-rhythm preservation
3. **Stage 3: ICA & Channel Artifact Cleaning (Steps 10–12)**
   - Fast Two-Phase optimization of `clean_rawdata` & `ICLabel` rejection thresholds (Step 10)
   - Full-length Extended Infomax ICA, artifact IC rejection, bad channel spherical interpolation, average re-referencing (Step 11)
   - Multi-stage interactive summary report generation (Step 12)

> **Note on step numbering.** The pipeline runs `01 → … → 08 → 10 → 11 → 12`. There is **no step 09**: the former `step09_ica.py` computed a *preliminary* ICA with default parameters into the exact same files Step 11 later overwrites, so it was a dead branch and has been **removed**. Step 11 is now the sole writer of `derivatives/05_ica/`.

---

## 🎛️ One-Command Orchestrator (`run_all.py`)

`run_all.py` runs the **entire chain 01 → 12** for you. You do not have to call the
per-step scripts by hand (those still exist for debugging a single stage).

```bash
python run_all.py                                     # every segment of the default subject (1916)
python run_all.py --segment-dir data/1916/segments/ec # one explicit segment
python run_all.py --subject 1916                      # every segment of subject 1916
python run_all.py --all                               # EVERY segment of EVERY subject under data/
python run_all.py --all --recalc                      # full clean recompute of everything
```

**Subject/segment resolution.** With no `--segment-dir`, the run processes **all
segments** of the target subject (`--subject`, or the default subject
`config.DEFAULT_EXPERIMENT` when neither `--subject` nor `--all` is given). The
subject is derived from the path (`data/<subject>/segments/<seg>`), so
`derivatives/` are always written under the correct subject — nothing is hard-coded
to `1916`. In `--all` / `--subject` mode a failure on one segment is logged and the
batch **continues**; a final `N ok / M failed` summary is printed at the end.

**Segment processing order.** Named segments run in a fixed canonical order —
`eo → ec → drone → lasertag → video` (`SEGMENT_ORDER` in `run_all.py`). Any segment
whose name is not in that list is processed afterwards in alphabetical order, so
unknown names are never skipped.

### Incremental execution (skip-if-computed)

By **default** each step is *skipped if its output already exists*, and the pipeline
moves on to the next step. This means you can safely re-run `run_all.py` after an
interruption — completed steps are not recomputed. Each step is guarded by a unique
**done-marker** file:

| Step | Done-marker (relative to the segment / its derivatives) |
|---|---|
| 01 | `segment_info.json` |
| 02 | `slice_detection.json` |
| 03 | `segment_work_info.json` |
| 04 | `optuna_best_params.json` |
| 05 | `*bergen*.set` (any Bergen `.set` present) |
| 06 | `step03_spectra_data.npz` |
| 07 | `<seg>_cleaning_report.html` |
| 08 | `derivatives/03_bcg/<seg>/<seg>_bcg_clean.fif` |
| 10 | `ica_optuna_best.json` |
| 11 | `derivatives/05_ica/<seg>/<seg>_ica_clean.fif` |

Step 12 (summary report) is always regenerated — it is cheap and only aggregates
existing artifacts.

### `--recalc`: the file-cleaning pipeline

`--recalc` forces a **full recompute from scratch**. Before running any step, it
calls `clean_all_derivatives(seg_dir)`, which **deletes every computed artifact** for
the segment so all done-markers disappear. It removes, in order:

1. **`data/<subject>/derivatives/<stage>/<seg>/`** — the whole per-segment subfolder
   under every stage (`01_bergen`, `02_resampled250`, `03_bcg`, `04_channels`,
   `05_ica`, …).
2. **`qc/<subject>/<stage>/<seg>*`** — QC plots (files or folders).
3. **Computed files inside the segment working dir** matching these patterns:

   | Pattern(s) | What they are |
   |---|---|
   | `segment_info.json`, `slice_detection.json`, `segment_work_info.json` | Step 01–03 metadata (their done-markers) |
   | `slice_triggers.txt`, `slice_phase_check.png` | Step 02–03 outputs |
   | `*.db`, `*optuna*`, `trial*`, `*_best_params.json` | Step 04 Optuna study, trials, best params |
   | `*.set`, `*.fdt`, `temp_multich_crop.mat`, `run_full_optuna_clean.m` | Step 05 Bergen dataset + MATLAB scratch |
   | `*.npz`, `*.png`, `*.csv`, `summary_*.csv`, `*_report.html` | Step 06–07 spectra + Bergen report |
   | `ica_optuna_best.json`, `ica_final` | Step 10–11 ICA params + scratch dir |

> **What `--recalc` NEVER touches:** the raw inputs. Continuous recordings live at
> `data/<subject>/raw/eeg96/<subject>.vhdr` and SPM motion files at
> `data/<subject>/raw/rp_spm/` (or on the external fMRI drive) — these are **outside**
> `seg_dir` and are never deleted. Pipeline source code is never touched either.

A deletion report (`Deleted N artifact path(s): …`) is printed so you can see exactly
what was wiped.

**Other useful flags:** `--skip-optuna` (Step 04), `--skip-bcg` (Step 08),
`--skip-ica-optuna` (Step 10), `--skip-ica` (Steps 10–11), `--skip-detect-mri`
(Step 01), `--trials N` (Optuna trials for both Bergen and ICA).

---

## 📂 Data Directory & Derivatives Layout

```
EEG_FMRI_CLEAN/
├── data/
│   └── <subject_id>/
│       ├── segments/
│       │   └── <segment_name>/             # e.g., ec, eo, drone, lasertag, video
│       │       ├── segment_info.json       # Step 01: Boundary times & SPM rp path
│       │       ├── slice_detection.json    # Step 02: Best slice trigger phase
│       │       ├── segment_work_info.json  # Step 03: Aligned bounds after dummy scans
│       │       ├── slice_triggers.txt      # Step 03: 1-based trigger indices for MATLAB
│       │       ├── optuna_best_params.json # Step 04: Winning Bergen AAS parameters
│       │       ├── summary_alpha_quality.csv # Step 06: Spectral metrics table
│       │       ├── <segment>_cleaning_report.html # Step 07: Bergen QC report
│       │       └── optuna_ica_best_params.json # Step 10: Winning ICA & clean_rawdata parameters
│       │
│       └── derivatives/
│           ├── 01_bergen/<seg>/            # Step 05: <seg>_bergen_optuna_*.set (5000 Hz)
│           ├── 02_resampled250/<seg>/      # Step 08: <seg>_250hz.fif (250 Hz)
│           ├── 03_bcg/<seg>/               # Step 08: <seg>_bcg_clean.fif & metrics.json
│           ├── 04_channels/<seg>/          # Step 11: removed_channels.json
│           └── 05_ica/<seg>/               # Step 11: <seg>_ica_clean.fif & metrics.json
│
└── reports/
    └── <subject_id>/
        └── <seg>/
            └── report_<subject_id>_<seg>.html  # Step 12: Complete Multi-Stage HTML Report
```

---

## 🚀 Step-by-Step Guide (Steps 01 to 12)

All steps are parameterized and can be executed for an entire subject (`--subject <ID>`), a specific segment (`--subject <ID> --segment <NAME>`), or all subjects/segments (`--all`).

### [STEP 01] MRI Session Detection (`step01_detect_mri.py`)
- **Goal**: Automatically scans continuous raw EEG (`.vhdr`, 5000 Hz) using sliding RMS envelope across key control channels (`Fp1, Fp2, Fz, Cz, Pz`) to detect fMRI gradient onsets and offsets.
- **SPM Motion Rule**: Automatically matches each fMRI run with its corresponding SPM motion file (`rp_*.txt`) located in `/media/vitaly/48DEA853CCEBFBF0/0DATDA_2026_eeg_fnri/fmri/fmri/<subject>FMRI/`. If missing, the pipeline halts with an error.
- **Output**: `data/<subject>/segments/<segment>/segment_info.json`
```bash
python step01_detect_mri.py --subject 1916
# Dry run preview without modifying files:
python step01_detect_mri.py --subject 1916 --dry-run
```

---

### [STEP 02] Sub-Sample Slice Phase Detection (`step02_detect_slices.py`)
- **Goal**: Detects exact sub-sample phase of gradient pulses by cross-correlating the absolute gradient profile with a comb template at the nominal slice period (100 ms = 500 samples @ 5000 Hz).
- **Output**: `slice_detection.json` and verification plot `slice_phase_check.png`.
```bash
python step02_detect_slices.py --subject 1916
python step02_detect_slices.py --subject 1916 --segment ec
```

---

### [STEP 03] Dummy Scan Trimming & SPM Alignment (`step03_trim_dummy.py`)
- **Goal**: Trims the initial scanner stabilization volumes (*dummy scans*), aligns the EEG duration precisely with the `rp_*.txt` volume count, and exports uniform synthetic slice triggers maintaining the Kronecker product invariant.
- **Dummy count is auto-detected from time alignment.** The number of dummy volumes is **not** a fixed constant — it is inferred so the trimmed EEG window matches the fMRI task in time:

  `n_dummy = round( (t_stop − t_start) / TR − n_work_volumes )`

  where the detected span comes from Step 01/02 and `n_work_volumes` is the row count of the SPM `rp_*.txt`. It is usually 12 (12 × 2.5 s = 30 s) but legitimately becomes 13 (or another value) on runs with a longer stabilization period. If the alignment is ambiguous (the estimate lands near a half-volume boundary, `frac > 0.35`), Step 03 falls back to `DEFAULT_DUMMY_VOLUMES` from `config.py` and prints a warning to check that segment manually.
- **Manual override**: pass `--dummy N` to force an exact count (e.g. `--dummy 13`). Step 03 still prints a warning if a forced value disagrees with the time alignment by more than half a volume.
- **Output**: `slice_triggers.txt` (1-based indices for MATLAB) and `segment_work_info.json` (now also records `dummy_auto`, `dummy_est`, and `tr_sec` for transparency).
```bash
python step03_trim_dummy.py --subject 1916              # auto-detect per segment
python step03_trim_dummy.py --subject 1916 --segment ec # auto-detect one segment
python step03_trim_dummy.py --subject 1916 --segment ec --dummy 13  # force 13
```

> **Why this matters:** the dummy count sets *where the EEG work window starts*. Trim
> too few and the window includes stabilization volumes; too many and it eats into
> real task data. The auto-detection keeps the EEG window locked to the fMRI task in
> time — which is the actual correctness criterion, not the specific number 12 vs 13.

---

### [STEP 04] Bayesian Bergen Hyperparameter Optimization (`step04_optuna_tune.py`)
- **Goal**: Performs Bayesian multi-objective hyperparameter optimization (Optuna TPE) on target channel `Oz` (~15 MB in RAM) to find optimal Bergen AAS settings:
  - `shift`: marker alignment phase (−6 to +6 samples)
  - `win_k`: volume averaging window (3 to 14 volumes)
  - `motion_thresh`: Framewise Displacement threshold (0.20 to 1.50 mm)
- **Output**: `optuna_best_params.json`, `optuna_result.png`, SQLite study `optuna_study.db`.
```bash
python step04_optuna_tune.py --subject 1916 --trials 20
python step04_optuna_tune.py --subject 1916 --segment ec --trials 25
```

---

### [STEP 05] Full Multi-Channel Bergen AAS Cleaning (`step05_bergen_clean.py`)
- **Goal**: Cleans all 96 EEG channels in MATLAB using the vectorized Bergen kernel `bergen_fast_correction.m`.
- **Memory Safety**: Uses `float32` arrays to save 50% RAM.
- **Output**: `data/<subject>/derivatives/01_bergen/<seg>/<seg>_bergen_optuna_*.set` (~935 MB).
```bash
python step05_bergen_clean.py --subject 1916
python step05_bergen_clean.py --subject 1916 --segment ec
```

---

### [STEP 06] Spectral Analysis & Alpha Quality Metrics (`step06_spectra_analysis.py`)
- **Goal**: Calculates Welch PSD (0.5–40 Hz) across occipital/parietal channels (`O1, Oz, O2, Pz, P3, P4, Fz, Cz`), computes gradient harmonic suppression (20, 30, 40, 50, 60 Hz), alpha prominence, and alpha preservation.
- **Output**: `summary_alpha_quality.csv`, `metrics.csv`, `alpha_quality_check.png`, `step03_spectra.png`.
```bash
python step06_spectra_analysis.py --subject 1916
python step06_spectra_analysis.py --subject 1916 --segment ec
```

---

### [STEP 07] Bergen Interactive HTML Report (`step07_html_report.py`)
- **Goal**: Creates a self-contained interactive HTML QC report for the Bergen gradient cleaning stage.
- **Output**: `data/<subject>/segments/<seg>/<seg>_cleaning_report.html`.
```bash
python step07_html_report.py --subject 1916
python step07_html_report.py --subject 1916 --segment ec
```

---

### [STEP 08] Ballistocardiogram (BCG) OBS Removal (`step08_bcg.py`)
- **Goal**:
  1. Downsamples Bergen-cleaned EEG from 5000 Hz $\to$ 250 Hz (data array drops from 950 MB $\to$ 28 MB).
  2. Applies a **1.0–80 Hz band-pass** after resampling (`FILTER_HP = 1.0`, `FILTER_LP = 80.0`). The 1.0 Hz high-pass removes slow baseline drift (0.5–1 Hz respiration/movement) that otherwise causes visible baseline wander in the cleaned time-domain traces; the 80 Hz low-pass matches the reference `LPF_80` pipeline. *(The high-pass was raised from 0.5 → 1.0 Hz — 0.5 Hz was too gentle and left visible drift.)*
  3. Detects QRS cardiac peaks using MATLAB fMRIB plugin (`fmrib_pas`).
  4. Sweeps Optimal Basis Set (`npc = 1..8`), evaluates cardiac suppression (0.7–4.0 Hz) vs. alpha retention (8–13 Hz), and selects optimal `npc` (or accepts manual `--npc`).
- **Output**:
  - `data/<subject>/derivatives/02_resampled250/<seg>/<seg>_250hz.fif`
  - `data/<subject>/derivatives/03_bcg/<seg>/<seg>_bcg_clean.fif`
  - `data/<subject>/derivatives/03_bcg/<seg>/<seg>_bcg_metrics.json`
```bash
python step08_bcg.py --subject 1916
python step08_bcg.py --subject 1916 --segment ec
# Force specific number of PCA components (e.g., npc=1):
python step08_bcg.py --subject 1916 --segment ec --npc 1
```

---

### [STEP 10] Fast Two-Phase ICA Optimization (`step10_optuna_ica.py`)
- **Goal**: High-speed parameter optimization on a 60-second crop using a two-phase strategy:
  - **Phase 1 (Grid Search)**: sweeps EEGLAB `clean_rawdata` parameters (`ChannelCriterion=0.7..0.85`, `FlatlineCriterion=5.0`, `LineNoiseCriterion=4.0..8.0`) without computing ICA.
  - **Phase 2 (Single ICA)**: computes Extended Infomax ICA and ICLabel once on the best cleaned channel set.
  - **Phase 3 (Python Sweep)**: sweeps ICLabel artifact rejection thresholds (`0.65..0.85`) instantly in RAM.
- **Output**: `optuna_ica_best_params.json` and `optuna_ica_result.png`.
```bash
python step10_optuna_ica.py --subject 1916
python step10_optuna_ica.py --subject 1916 --segment ec
```

---

### [STEP 11] Full-Length ICA & Bad Channel Interpolation (`step11_ica_final.py`)
- **Goal**: Applies optimal parameters to the complete segment duration at 250 Hz:
  1. Applies `clean_rawdata` to remove bad/flatline channels.
  2. Fits Extended Infomax ICA (with rank-deficient PCA reduction if needed).
  3. Classifies components with ICLabel and removes artifact ICs (Eye, Muscle, Heart, Line Noise, Channel Noise).
  4. Spherical spline interpolation of removed channels back to original 95-channel layout.
  5. Average Reference re-referencing (`pop_reref`).
- **Output**:
  - `data/<subject>/derivatives/04_channels/<seg>/removed_channels.json`
  - `data/<subject>/derivatives/05_ica/<seg>/<seg>_ica_clean.fif` (Final Cleaned EEG)
  - `data/<subject>/derivatives/05_ica/<seg>/<seg>_ica_metrics.json`
  - `data/<subject>/segments/<seg>/qc/<seg>_ica_report.html`
```bash
python step11_ica_final.py --subject 1916
python step11_ica_final.py --subject 1916 --segment ec
```

---

### [STEP 12] Complete Multi-Stage Summary Report (`step12_summary_report.py`)
- **Goal**: Compiles an end-to-end HTML dashboard comparing Raw EEG $\to$ Bergen AAS $\to$ BCG OBS $\to$ Final ICA, so every operation and its parameters can be inspected *before vs. after*:
  - Per-stage **before/after PSD overlays** on occipital/parietal channels
  - **Parameter tables** for each operation (Bergen: dummy volumes trimmed, work volumes, `shift`, `win_k`, `motion_thresh`; BCG: `npc`; ICA: `clean_rawdata` + ICLabel thresholds)
  - Alpha preservation and artifact suppression **metrics tables** with an OK/WARN verdict per stage
  - Rejection summary (channels removed, ICs pruned)
- **Output**: `reports/<subject>/<seg>/report_<subject>_<seg>.html`

> **Data source note.** The "dummy volumes trimmed" figure is read from
> `segment_work_info.json` (written by Step 03). Segments processed before this field
> was added show `—` until Step 03 is re-run (e.g. via `run_all.py --recalc`).
```bash
python step12_summary_report.py --subject 1916
python step12_summary_report.py --subject 1916 --segment ec
```

---

## ⚡ Batch Processing

**Recommended — let `run_all.py` drive the whole chain:**

```bash
python run_all.py --subject 1916            # all segments of subject 1916 (01 → 12)
python run_all.py --subject 1916 --recalc   # same, but wipe old artifacts first
python run_all.py --all                      # every segment of every subject
```

`run_all.py` handles ordering, skip-if-computed, per-subject path resolution, and
`--recalc` cleaning automatically (see the orchestrator section above).

**Manual — call each stage yourself** (useful for debugging a single step):

```bash
# 1. Gradient Artifact Removal (Steps 01-07)
python step01_detect_mri.py --subject 1916
python step02_detect_slices.py --subject 1916
python step03_trim_dummy.py --subject 1916
python step04_optuna_tune.py --subject 1916
python step05_bergen_clean.py --subject 1916
python step06_spectra_analysis.py --subject 1916
python step07_html_report.py --subject 1916

# 2. BCG & ICA Cleaning (Steps 08, 10-12 — there is no step 09)
python step08_bcg.py --subject 1916
python step10_optuna_ica.py --subject 1916
python step11_ica_final.py --subject 1916
python step12_summary_report.py --subject 1916
```

---

## 🛡️ RAM & Memory Optimization Rules

- **Steps 01–05 (5000 Hz Raw)**: Arrays are loaded as `np.float32` (~900 MB for 96 channels). MATLAB scripts allocate memory temporarily and call `clear` and `gc.collect()` immediately after processing.
- **Steps 08–12 (250 Hz Resampled)**: Downsampled datasets are ~28 MB per segment. Processing uses < 1.8 GB RAM during ICA, fully preventing Linux OOM kills even on 16 GB machines.
- **Kronecker Structure**: Slice templates are averaged across volume multiples ($W = W_{vol} \otimes I_{25}$), protecting the 10 Hz alpha rhythm from template subtraction contamination.

---

## 📊 Quality Targets Reference

| Metric | Target | Warning / Over-cleaning Flag |
|---|---|---|
| **Bergen MRI Suppression (20–60 Hz)** | $\ge 99.5\%$ | $< 99.0\%$ |
| **Bergen Alpha Retention** | $\ge 85\%$ | $< 70\%$ |
| **BCG Cardiac Suppression** | $\ge 20\%$ | $< 10\%$ (if BCG present) |
| **ICA Rejected Components** | $20\% - 40\%$ | $> 60\%$ (over-cleaning) |
| **ICA Variance Drop** | $\le 30\%$ | $> 50\%$ |
| **ICA Alpha Retention** | $\ge 70\%$ | $< 50\%$ |
| **Removed Bad Channels** | $\le 10\%$ ($\le 9$ ch) | $> 15$ channels |
