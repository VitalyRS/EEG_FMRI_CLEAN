# 🧠 Modular EEG-fMRI Cleaning Pipeline: Step 01 to Step 12

**High-precision modular software pipeline for concurrent EEG-fMRI gradient artifact removal (Bergen AAS), BCG (ballistocardiogram) OBS cleaning, and ICA with ICLabel auto-classification and quantitative alpha rhythm preservation control.**

---

## 📋 Overview of Pipeline Architecture

The pipeline processes raw high-frequency EEG data recorded inside an MRI scanner (5000 Hz) through sequential, automated, and memory-safe cleaning stages:

1. **Stage 1: MRI Gradient Artifact Removal (Steps 01–07)**
   - Automatic fMRI session boundary detection with short-burst noise filtering
   - Robust positional matching of SPM motion files (`rp_*.txt`) sorted by series number
   - Sub-sample slice phase offset identification (comb-filter template cross-correlation)
   - Adaptive dummy scan trimming & SPM motion volume alignment
   - Bayesian hyperparameter tuning (Optuna TPE) for Bergen AAS on occipital target (`Oz`)
   - Full 96-channel Bergen Average Artifact Subtraction (AAS) in MATLAB (`float32` memory-safe)
   - Spectral Welch PSD analysis & Bergen interactive HTML QC reporting
2. **Stage 2: Ballistocardiogram (BCG) Artifact Removal (Step 08)**
   - Resampling from 5000 Hz to 250 Hz (reducing RAM footprint to ~28 MB per segment)
   - 1.0–80 Hz band-pass filtering to eliminate slow baseline respiration/movement drift
   - MATLAB fMRIB QRS cardiac peak detection (`fmrib_pas`)
   - Optimal Basis Set (OBS) sweep with cardiac suppression (0.7–4.0 Hz) vs. alpha retention (8–13 Hz) control
3. **Stage 3: ICA & Channel Artifact Cleaning (Steps 10–12)**
   - Fast Two-Phase optimization of `clean_rawdata` & `ICLabel` rejection thresholds (Step 10)
   - Full-length Extended Infomax ICA, artifact IC rejection (Eye, Muscle, Heart, Line Noise, Channel Noise), spherical spline channel interpolation, and average re-referencing (Step 11)
   - Multi-stage interactive summary report generation (Step 12)

> **Note on step numbering.** The pipeline executes `01 → … → 08 → 10 → 11 → 12`. There is **no step 09**: the former `step09_ica.py` computed a *preliminary* ICA with default parameters into the exact same files Step 11 later overwrites, making it redundant. Step 11 is the sole writer of `derivatives/05_ica/`.

---

## 🎛️ One-Command Orchestrator (`run_all.py`)

`run_all.py` executes the **entire chain 01 → 12** automatically. You do not need to call per-step scripts individually unless debugging a specific stage.

```bash
# Run all segments for a specific subject (e.g. 1929)
python run_all.py --subject 1929

# Run all segments for a subject, wiping old derivative artifacts first (full clean recompute)
python run_all.py --subject 1929 --recalc --skip-detect-mri

# Run a single specific segment folder with clean recalculation
python run_all.py --segment-dir data/1929/segments/lasertag --recalc

# Run every segment across all subjects discovered under data/
python run_all.py --all

# Full clean recompute of every subject and every segment
python run_all.py --all --recalc
```

### Subject and Segment Resolution
- **Target resolution:** If `--segment-dir` is omitted, the pipeline processes **all segments** for `--subject <ID>` (or `config.DEFAULT_EXPERIMENT` if neither `--subject` nor `--all` is specified).
- **Per-subject derivative paths:** Subject ID is dynamically derived from the directory structure (`data/<subject>/segments/<seg>`), ensuring `derivatives/` and `reports/` are always written to the correct subject tree.
- **Batch resilience:** In `--all` or `--subject` batch mode, if one segment fails, the error is logged and the batch **continues** with the next segment; a final `N ok / M failed` summary table is displayed upon completion.
- **Canonical processing order:** Named segments are processed in chronological acquisition order:
  $$\mathbf{ec \to eo \to drone \to lasertag \to video}$$
  (`SEGMENT_ORDER` in `run_all.py`). Any unexpected or custom segment names are processed alphabetically afterwards so no folder is overlooked.

### Incremental Execution (Skip-If-Computed)

By default, each step is **skipped if its output marker already exists** on disk. This guarantees safe resumption after any interruption without recomputing completed work.

| Step | Operation | Done-Marker Path (relative to segment or derivatives) |
|:---:|---|---|
| **01** | Session Detection | `segment_info.json` |
| **02** | Slice Phase Offset | `slice_detection.json` |
| **03** | Dummy Trimming | `segment_work_info.json` |
| **04** | Optuna Bergen AAS | `optuna_best_params.json` |
| **05** | Full Bergen AAS | `*bergen*.set` (any Bergen `.set` dataset) |
| **06** | Spectral Analysis | `step03_spectra_data.npz` |
| **07** | Bergen HTML QC | `<seg>_cleaning_report.html` |
| **08** | BCG OBS Filter | `derivatives/03_bcg/<seg>/<seg>_bcg_clean.fif` |
| **10** | Optuna ICA Params | `ica_optuna_best.json` |
| **11** | Final ICA & Interpolation | `derivatives/05_ica/<seg>/<seg>_ica_clean.fif` |

*Step 12 (Multi-Stage Summary Report) is always regenerated as an aggregate dashboard.*

### `--recalc`: Safe Artifact Cleaning

`--recalc` forces a complete recompute of Steps 02–12 for the selected segment(s). It invokes `clean_all_derivatives(seg_dir)`, removing:
1. **`data/<subject>/derivatives/<stage>/<seg>/`** — all downstream derivatives (`01_bergen`, `02_resampled250`, `03_bcg`, `04_channels`, `05_ica`).
2. **`qc/<subject>/<stage>/<seg>*`** — all intermediate stage QC figures and plots.
3. **Computed files in the segment working directory:**
   - `slice_detection.json`, `segment_work_info.json`, `slice_triggers.txt`, `slice_phase_check.png` (Steps 02–03)
   - `*.db`, `*optuna*`, `trial*`, `*_best_params.json` (Step 04 Optuna tuning)
   - `*.set`, `*.fdt`, `temp_multich_crop.mat`, `run_full_optuna_clean.m` (Step 05 Bergen output & MATLAB scratch)
   - `*.npz`, `*.png`, `*.csv`, `summary_*.csv`, `*_report.html` (Steps 06–07 Spectra & Bergen reports)
   - `ica_optuna_best.json`, `ica_final` (Steps 10–11 ICA tuning & scratch)

> [!IMPORTANT]
> **Preservation of Raw Inputs & Step 01 Metadata:**
> `--recalc` **never** deletes raw EEG recordings (`.vhdr`/`.eeg`/`.vmrk`), SPM motion files (`rp_*.txt`), or Step 01 `segment_info.json`. This prevents redundant re-reading of continuous 4 GB multi-channel raw EEG files across every segment loop.

**Useful Orchestrator Flags:**
- `--skip-detect-mri`: Skip Step 01 (assumes `segment_info.json` already exists).
- `--skip-optuna`: Skip Step 04 (use default/existing Bergen AAS hyperparameters).
- `--skip-bcg`: Skip Step 08 (BCG artifact removal).
- `--skip-ica-optuna`: Skip Step 10 (use default/existing ICA hyperparameters).
- `--skip-ica`: Skip Steps 10–11 completely.
- `--trials N`: Override the number of Optuna trials for Bergen and ICA searches (default: 20).

---

## 📂 Data Directory & Derivatives Layout

```
EEG_FMRI_CLEAN/
├── data/
│   └── <subject_id>/                       # e.g., 1927, 1928, 1929, 1930, 1932
│       ├── segments/
│       │   ├── segments_duration.txt       # Step 01: Summary of detected session timings
│       │   └── <segment_name>/             # ec, eo, drone, lasertag, video
│       │       ├── segment_info.json       # Step 01: Boundary times, sfreq, SPM rp path
│       │       ├── slice_detection.json    # Step 02: Best slice trigger phase
│       │       ├── segment_work_info.json  # Step 03: Aligned bounds after dummy scans
│       │       ├── slice_triggers.txt      # Step 03: 1-based trigger sample indices for MATLAB
│       │       ├── optuna_best_params.json # Step 04: Winning Bergen AAS parameters
│       │       ├── summary_alpha_quality.csv # Step 06: Spectral metrics table
│       │       ├── <segment>_cleaning_report.html # Step 07: Bergen QC report
│       │       └── ica_optuna_best.json    # Step 10: Winning ICA & clean_rawdata parameters
│       │
│       └── derivatives/
│           ├── 01_bergen/<seg>/            # Step 05: <seg>_bergen_optuna_*.set (5000 Hz)
│           ├── 02_resampled250/<seg>/      # Step 08: <seg>_250hz.fif (250 Hz)
│           ├── 03_bcg/<seg>/               # Step 08: <seg>_bcg_clean.fif & metrics.json
│           ├── 04_channels/<seg>/          # Step 11: removed_channels.json
│           └── 05_ica/<seg>/               # Step 11: <seg>_ica_clean.fif & metrics.json
│
└── reports/
    ├── step01_segment_detection/
    │   ├── all_subjects_report.html        # Step 01: Multi-subject alignment dashboard
    │   └── <subject_id>_report.html        # Step 01: Per-subject RMS envelope & session table
    └── <subject_id>/
        └── <seg>/
            └── report_<subject_id>_<seg>.html  # Step 12: Complete Multi-Stage HTML Report
```

---

## 🚀 Detailed Step-by-Step Guide (Steps 01 to 12)

All steps accept `--subject <ID>`, `--subject <ID> --segment <NAME>`, or `--all`.

### [STEP 01] MRI Session Detection & Alignment (`step01_detect_mri.py`)
- **Objective**: Scans continuous raw EEG (`.vhdr`, 5000 Hz) using a sliding RMS envelope across control channels (`Fp1, Fp2, Fz, Cz, Pz`) to detect fMRI gradient RF burst onsets and offsets.
- **Short-Segment Filtering (`MIN_NAMED_SEC = 250.0s`)**:
  Segments shorter than 250 seconds (such as calibration scans or aborted gradient bursts) are automatically skipped (`SKIP (dur < 250s)`), preventing spurious bursts from stealing named session slots.
- **Positional SPM Motion Matching**:
  Discovers all `rp_*.txt` files in the subject's fMRI folder (`.../<subject>FMRI/`), sorts them in ascending numerical series order (e.g. series `301, 401, 501, 701, 801`), and maps them directly to the 5 valid sessions:
  $$\text{1st} \to \text{ec}, \quad \text{2nd} \to \text{eo}, \quad \text{3rd} \to \text{drone}, \quad \text{4th} \to \text{lasertag}, \quad \text{5th} \to \text{video}$$
- **Interactive Reports**:
  - `reports/step01_segment_detection/<subject>_report.html` (interactive RMS envelope plot + segment table)
  - `reports/step01_segment_detection/all_subjects_report.html` (global multi-subject status dashboard)
```bash
python step01_detect_mri.py --subject 1929
```

---

### [STEP 02] Sub-Sample Slice Phase Detection (`step02_detect_slices.py`)
- **Objective**: Detects the exact sub-sample phase of gradient pulses by cross-correlating the absolute gradient profile with a comb template at the nominal slice period ($100\text{ ms} = 500\text{ samples}$ @ 5000 Hz).
- **Outputs**: `slice_detection.json` and verification figure `slice_phase_check.png`.
```bash
python step02_detect_slices.py --subject 1929
python step02_detect_slices.py --subject 1929 --segment ec
```

---

### [STEP 03] Dummy Scan Trimming & SPM Alignment (`step03_trim_dummy.py`)
- **Objective**: Trims initial scanner stabilization volumes (*dummy scans*), synchronizes EEG start time with the `rp_*.txt` volume count, and exports uniform synthetic slice triggers maintaining the Kronecker product invariant.
- **Dynamic Dummy Auto-Detection**:
  $$n_{\text{dummy}} = \text{round}\left( \frac{t_{\text{stop}} - t_{\text{start}}}{\text{TR}} - n_{\text{work\_volumes}} \right)$$
  Where $n_{\text{work\_volumes}}$ is the row count of `rp_*.txt`. Usually 12 ($12 \times 2.5\text{ s} = 30\text{ s}$), but automatically adapts to 13 (or other durations) if stabilization was prolonged.
- **Manual override**: `--dummy N` forces an exact volume count (e.g. `--dummy 12`).
- **Outputs**: `slice_triggers.txt` (1-based indices for MATLAB) and `segment_work_info.json`.
```bash
python step03_trim_dummy.py --subject 1929              # auto-detect per segment
python step03_trim_dummy.py --subject 1929 --segment ec # auto-detect one segment
python step03_trim_dummy.py --subject 1929 --segment ec --dummy 12  # force 12
```

---

### [STEP 04] Bayesian Bergen Hyperparameter Optimization (`step04_optuna_tune.py`)
- **Objective**: Multi-objective Bayesian hyperparameter optimization (Optuna TPE) executed on target channel `Oz` (~15 MB in RAM) to optimize Bergen AAS parameters:
  - `shift`: marker alignment phase ($-6$ to $+6$ samples)
  - `win_k`: volume averaging window size (3 to 14 volumes)
  - `motion_thresh`: Framewise Displacement threshold (0.20 to 1.50 mm)
- **Outputs**: `optuna_best_params.json`, `optuna_result.png`, SQLite study `optuna_study.db`.
```bash
python step04_optuna_tune.py --subject 1929 --trials 20
python step04_optuna_tune.py --subject 1929 --segment ec --trials 25
```

---

### [STEP 05] Full Multi-Channel Bergen AAS Cleaning (`step05_bergen_clean.py`)
- **Objective**: Cleans all 96 EEG channels in MATLAB using the vectorized Bergen kernel `bergen_fast_correction.m`.
- **Memory Safety**: Employs `float32` arrays to cut peak memory usage by 50%.
- **Output**: `data/<subject>/derivatives/01_bergen/<seg>/<seg>_bergen_optuna_*.set` (~935 MB).
```bash
python step05_bergen_clean.py --subject 1929
python step05_bergen_clean.py --subject 1929 --segment ec
```

---

### [STEP 06] Spectral Analysis & Alpha Quality Metrics (`step06_spectra_analysis.py`)
- **Objective**: Computes Welch PSD (0.5–40 Hz) across occipital/parietal channels (`O1, Oz, O2, Pz, P3, P4, Fz, Cz`), evaluating gradient harmonic suppression (20, 30, 40, 50, 60 Hz), alpha prominence, and alpha preservation ratio.
- **Outputs**: `summary_alpha_quality.csv`, `metrics.csv`, `alpha_quality_check.png`, `step03_spectra.png`.
```bash
python step06_spectra_analysis.py --subject 1929
python step06_spectra_analysis.py --subject 1929 --segment ec
```

---

### [STEP 07] Bergen Interactive HTML Report (`step07_html_report.py`)
- **Objective**: Generates an interactive standalone HTML QC report detailing Bergen gradient cleaning performance.
- **Output**: `data/<subject>/segments/<seg>/<seg>_cleaning_report.html`.
```bash
python step07_html_report.py --subject 1929
python step07_html_report.py --subject 1929 --segment ec
```

---

### [STEP 08] Ballistocardiogram (BCG) OBS Removal (`step08_bcg.py`)
- **Objective**:
  1. Downsamples Bergen-cleaned EEG from 5000 Hz $\to$ 250 Hz (reducing array size from 950 MB $\to$ 28 MB).
  2. Applies a **1.0–80 Hz band-pass filter** (`FILTER_HP = 1.0`, `FILTER_LP = 80.0`). The 1.0 Hz high-pass effectively suppresses slow baseline wander (respiration/drift).
  3. Detects QRS cardiac peaks with the MATLAB fMRIB toolbox (`fmrib_pas`).
  4. Sweeps Optimal Basis Set (`npc = 1..8`), balancing cardiac suppression (0.7–4.0 Hz) with alpha retention (8–13 Hz), selecting the optimal `npc`.
- **Outputs**:
  - `data/<subject>/derivatives/02_resampled250/<seg>/<seg>_250hz.fif`
  - `data/<subject>/derivatives/03_bcg/<seg>/<seg>_bcg_clean.fif`
  - `data/<subject>/derivatives/03_bcg/<seg>/<seg>_bcg_metrics.json`
```bash
python step08_bcg.py --subject 1929
python step08_bcg.py --subject 1929 --segment ec
# Force specific number of PCA components (e.g. npc=1):
python step08_bcg.py --subject 1929 --segment ec --npc 1
```

---

### [STEP 10] Fast Two-Phase ICA Optimization (`step10_optuna_ica.py`)
- **Objective**: High-speed parameter optimization on a 60-second crop using a two-phase search:
  - **Phase 1 (Grid Search)**: Sweeps EEGLAB `clean_rawdata` parameters (`ChannelCriterion=0.7..0.85`, `FlatlineCriterion=5.0`, `LineNoiseCriterion=4.0..8.0`) without running full ICA.
  - **Phase 2 (Single ICA)**: Computes Extended Infomax ICA and ICLabel once on the best cleaned channel set.
  - **Phase 3 (Python Sweep)**: Sweeps ICLabel artifact rejection probability thresholds (`0.65..0.85`) instantly in RAM.
- **Outputs**: `ica_optuna_best.json` and `optuna_ica_result.png`.
```bash
python step10_optuna_ica.py --subject 1929
python step10_optuna_ica.py --subject 1929 --segment ec
```

---

### [STEP 11] Full-Length ICA & Bad Channel Interpolation (`step11_ica_final.py`)
- **Objective**: Applies the optimized parameters to the full continuous 250 Hz segment:
  1. Applies `clean_rawdata` to prune bad / flatline channels.
  2. Fits Extended Infomax ICA (with rank-deficient PCA reduction when channels are removed).
  3. Classifies components using ICLabel and removes artifact ICs (Eye, Muscle, Heart, Line Noise, Channel Noise).
  4. Performs spherical spline interpolation of removed channels back to the full 95 EEG channel layout.
  5. Applies Average Reference re-referencing (`pop_reref`).
- **Outputs**:
  - `data/<subject>/derivatives/04_channels/<seg>/removed_channels.json`
  - `data/<subject>/derivatives/05_ica/<seg>/<seg>_ica_clean.fif` (Final Cleaned EEG)
  - `data/<subject>/derivatives/05_ica/<seg>/<seg>_ica_metrics.json`
  - `data/<subject>/segments/<seg>/qc/<seg>_ica_report.html`
```bash
python step11_ica_final.py --subject 1929
python step11_ica_final.py --subject 1929 --segment ec
```

---

### [STEP 12] Complete Multi-Stage Summary Report (`step12_summary_report.py`)
- **Objective**: Builds an end-to-end interactive dashboard comparing **Raw EEG $\to$ Bergen AAS $\to$ BCG OBS $\to$ Final ICA**:
  - Per-stage **before/after PSD overlays** across occipital and parietal channels
  - **Operation parameter tables** (Bergen: dummy trimmed, work volumes, `shift`, `win_k`, `motion_thresh`; BCG: `npc`; ICA: `clean_rawdata` + ICLabel thresholds)
  - Quantitative alpha preservation and artifact suppression **metrics tables** with clear OK/WARN verdicts
  - Channel and IC rejection summaries
- **Output**: `reports/<subject>/<seg>/report_<subject>_<seg>.html`
```bash
python step12_summary_report.py --subject 1929
python step12_summary_report.py --subject 1929 --segment ec
```

---

## 🛡️ Memory & Performance Guidelines

- **High-Frequency Stage (Steps 01–05 @ 5000 Hz)**: Raw 96-channel arrays are loaded strictly as `np.float32` (~900 MB). MATLAB processes matrices in place, executing `clear` and explicit garbage collection immediately.
- **Resampled Stage (Steps 08–12 @ 250 Hz)**: Downsampled datasets occupy ~28 MB per segment. Extended Infomax ICA and ICLabel require < 1.8 GB RAM, eliminating out-of-memory (OOM) risks even on memory-constrained systems.
- **Kronecker Slice Invariance**: Bergen slice templates are computed as volume multiples ($W = W_{vol} \otimes I_{25}$), protecting true 10 Hz physiological alpha oscillations from artifact subtraction distortion.

---

## 📊 Quality Targets Reference

| Metric | Target Value | Warning / Over-Cleaning Threshold |
|---|:---:|:---:|
| **Bergen MRI Gradient Suppression (20–60 Hz)** | $\ge 99.5\%$ | $< 99.0\%$ |
| **Bergen Alpha Rhythm Retention** | $\ge 85.0\%$ | $< 70.0\%$ |
| **BCG Cardiac Suppression (0.7–4.0 Hz)** | $\ge 20.0\%$ | $< 10.0\%$ (when BCG present) |
| **ICA Rejected Independent Components** | $20\% - 40\%$ | $> 60\%$ (over-cleaning alert) |
| **ICA Total Variance Drop** | $\le 30.0\%$ | $> 50.0\%$ |
| **ICA Alpha Rhythm Retention** | $\ge 70.0\%$ | $< 50.0\%$ |
| **Interpolated Bad Channels** | $\le 10\%$ ($\le 9$ ch) | $> 15$ channels |
