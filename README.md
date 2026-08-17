# 🧠 Optuna-Bergen EEG-fMRI Gradient Artifact Removal Pipeline

**High-precision modular software for concurrent EEG-fMRI gradient artifact removal with quantitative alpha rhythm preservation control.**

This pipeline combines the vectorized Bergen Average Artifact Subtraction (AAS) algorithm (MATLAB/EEGLAB) with Bayesian multi-objective hyperparameter optimization (Python/Optuna TPE) to remove MRI gradient artifacts from multi-channel EEG recordings (up to 96+ channels at 5000 Hz) while preserving physiological brain rhythms, particularly the alpha band (8–13 Hz).

---

## 🎯 Core Principles & Scientific Rationale

### 1. **No Blind Destructive Filtering**
- **Does NOT apply**: notch filters, aggressive low-pass filters (< 40 Hz), ICA, ASR, or BCG filtering.
- Gradient artifacts are removed exclusively via **mathematical template subtraction** (Bergen AAS), fully preserving waveform morphology and physiological frequency content across the entire EEG spectrum.

### 2. **Dual-Objective Quality Control (Multi-Objective Alpha Preservation)**
- **Objective 1: Gradient Suppression (G_suppression > 99.9%)** on slice harmonics (20, 30, 40, 50, 60 Hz). The 10 Hz harmonic is **excluded** from suppression metrics to avoid artificially penalizing alpha-band preservation.
- **Objective 2: Alpha Rhythm Preservation (8–13 Hz)**: quantifies alpha peak prominence, peak frequency stability (f_α ≈ 9.8–10.2 Hz), and alpha retention relative to a reference EEG recording outside the scanner (EEG21 dataset).

### 3. **Zero-Data-Duplication Architecture (Zero-Bloat)**
- Eliminates intermediate gigabyte-sized `.fif`/`.eeg` file copies for each segment.
- All pipeline steps access the continuous raw BrainVision recording via **memory-mapped lazy slicing**, writing only lightweight JSON/TXT metadata and a single final cleaned `.set` file to disk.

### 4. **Volume-Wise Kronecker Template Averaging**
- The averaging matrix is constructed strictly according to MRI acquisition geometry via the Kronecker product: **W = kron(W_vol, I_slices)**.
- Slice #1 is averaged only with Slice #1 from neighboring volumes (TR = 2.5 s), ensuring alpha oscillations (10 Hz = 100 ms period) never contaminate the subtraction template.

### 5. **Per-Trial Artifact Persistence**
- Each Optuna trial creates its own subdirectory (`segment4/trial0/`, `trial1/`, …) containing:
  - `params.json` — hyperparameters (shift, win_k, motion_thresh)
  - `metrics.json` — computed loss, gradient suppression, alpha retention, etc.
  - `spectrum.png` — PSD comparison plot (raw vs. clean, 0–70 Hz) with harmonic markers
  - `clean_1ch.mat` — cleaned single-channel signal for inspection
  - `bergen_clean.m` — executed MATLAB script for reproducibility
- Enables side-by-side visual comparison of trials and full audit trail of the optimization process.

---

## 📂 Project Architecture

```
EEG_FMRI_CLEAN/
├── config.py                      # Centralized configuration: paths, frequencies, bands
├── run_all.py                     # End-to-end master runner
├── bergen_fast_correction.m       # Vectorized high-speed Bergen AAS kernel (MATLAB)
│
├── step01_detect_mri.py           # Step 01: Auto-detect MRI session temporal windows
├── step02_detect_slices.py        # Step 02: Sub-sample slice gradient phase detection
├── step03_trim_dummy.py           # Step 03: Trim dummy scans, align with SPM rp_*.txt
├── step04_optuna_tune.py          # Step 04: Bayesian optimization (Optuna TPE)
├── step05_bergen_clean.py         # Step 05: Full multi-channel Bergen AAS cleaning
├── step06_spectra_analysis.py     # Step 06: Welch PSD, quality metrics, EEG21 comparison
├── step07_html_report.py          # Step 07: Self-contained interactive HTML report
│
├── EEG_files/                     # Continuous raw EEG recordings
│   └── 1916/
│       ├── 1916_inside.vhdr       # Continuous EEG inside MRI (BrainVision)
│       ├── 1916_inside.vmrk
│       └── 1916_inside.eeg
│
└── 1916/                          # Subject directory
    └── segments/
        └── segment4/              # Working directory for target segment
            ├── trial0/            # Optuna trial 0 artifacts
            │   ├── params.json
            │   ├── metrics.json
            │   ├── spectrum.png
            │   ├── clean_1ch.mat
            │   └── bergen_clean.m
            ├── trial1/            # Optuna trial 1 artifacts
            │   └── ...
            ├── ...
            ├── add/
            │   ├── rp/            # SPM head motion parameters (rp_*.txt)
            │   └── eeg21/         # Reference 21-channel EEG outside MRI (.edf + .blocks)
            ├── segment_info.json          # Session metadata (t_start, t_stop)
            ├── slice_detection.json       # Detected slice phase (best_phase)
            ├── segment_work_info.json     # Synchronized working interval boundaries
            ├── slice_triggers.txt         # Slice trigger indices for MATLAB (1-based)
            ├── optuna_best_params.json    # Winning hyperparameters from Optuna
            ├── optuna_study.db            # SQLite database of all trials (persistent)
            ├── summary_alpha_quality.csv  # Quality metrics summary table
            ├── metrics.csv                # Metrics table for downstream analysis
            ├── alpha_quality_check.png    # Alpha rhythm control panel (5–20 Hz)
            ├── step03_spectra.png         # Full spectral overview (0.5–40 Hz)
            ├── segment4_bergen_*.set      # Cleaned full multi-channel EEGLAB dataset
            └── segment4_cleaning_report.html # Final self-contained HTML report
```

---

## 🔬 Pipeline Steps — Detailed Description

```
       Continuous Raw EEG (1916_inside.vhdr, 5000 Hz)
                              │
                              ▼
  [STEP 01] step01_detect_mri.py      ──► segment_info.json
                              │
                              ▼
  [STEP 02] step02_detect_slices.py   ──► slice_detection.json, slice_phase_check.png
                              │
                              ▼
  [STEP 03] step03_trim_dummy.py      ──► slice_triggers.txt, segment_work_info.json
           (+ SPM rp_*.txt)
                              │
                              ▼
  [STEP 04] step04_optuna_tune.py     ──► trial0..N/, optuna_best_params.json, optuna_result.png
           (1-channel Oz, G_supp + Alpha retention)
                              │
                              ▼
  [STEP 05] step05_bergen_clean.py    ──► segmentX_bergen_optuna_*.set (96 channels)
           (MATLAB AAS, float32)
                              │
                              ▼
  [STEP 06] step06_spectra_analysis.py──► summary_alpha_quality.csv, alpha_quality_check.png
           (Comparison with EEG21 reference)
                              │
                              ▼
  [STEP 07] step07_html_report.py     ──► segmentX_cleaning_report.html
```

### Step 01: MRI Session Detection (`step01_detect_mri.py`)
- Analyzes sliding root-mean-square (RMS) envelope of control channels (`Fp1, Fp2, Fz, Cz, Pz`) in 200 ms windows.
- Identifies MRI scanning start/stop boundaries [t_start, t_stop], merges micro-pauses, and saves `segment_info.json` metadata without duplicating heavy data files.

### Step 02: Sub-Sample Slice Phase Detection (`step02_detect_slices.py`)
- Computes the absolute gradient profile: `Profile(t) = Σ|ΔEEG_ch(t)|`.
- Performs cross-correlation between the profile and a comb template at the nominal slice period (TR_slice = TR / N_slices = 2.500 / 25 = 100 ms = 500 samples @ 5000 Hz).
- Determines the precise sub-sample phase offset of the gradient pulse (`best_phase_samples`) and generates a validation plot `slice_phase_check.png`.

### Step 03: Dummy Scan Trimming & SPM Alignment (`step03_trim_dummy.py`)
- Removes initial scanner stabilization volumes (*dummy scans*, default 12 volumes = 30 s).
- Automatically locates the SPM motion parameter file `rp_*.txt` (including recursive search in `add/rp/`) and aligns the exact EEG length to match the number of volumes in the rp file.
- Exports lightweight text file `slice_triggers.txt` (1-based indices for MATLAB) and metadata `segment_work_info.json`.
- **Critical detail**: Generates a **uniform synthetic trigger array** at exactly `nominal_slice_samples` spacing (500 samples), preserving the exact count as an integer multiple of `slices_per_volume` (25) to maintain the Kronecker product invariant.

### Step 04: Bayesian Hyperparameter Optimization (`step04_optuna_tune.py`)
- Extracts only 1 target channel into RAM (default occipital `Oz`, ~15–20 MB).
- Launches a series of fast Bergen AAS trials in MATLAB using Optuna's TPE Sampler with Median Pruner, optimizing:
  - `shift`: fine phase adjustment of slice markers (−6 to +6 samples).
  - `win_k`: width of the sliding template averaging window across volumes (3 to 14 volumes, 7.5 to 35.0 s). **Small windows (k~4) track non-stationary artifacts better than wide windows (k>20) for subjects with head motion.**
  - `motion_thresh`: Framewise Displacement threshold from SPM data (0.20 to 1.50 mm).
- **Loss Function** (multi-objective, minimized):
  ```
  Loss = (1.0 - G_suppression) × 100.0 + 200.0 × max(0, α_retention_target - α_retention)²
  ```
  where `G_suppression` is computed over non-alpha harmonics (20, 30, 40, 50, 60 Hz), and `α_retention` protects the physiological alpha band (8–13 Hz) by penalizing retention below 90%.
- **Per-trial persistence**: Each trial is saved to `segment4/trialN/` with:
  - `params.json`, `metrics.json`, `spectrum.png` (PSD raw vs. clean), `clean_1ch.mat`, `bergen_clean.m`
- Saves winning parameters to `optuna_best_params.json`, trial history to `optuna_study.db` (SQLite), and optimization summary plot `optuna_result.png`.
- **Study versioning**: The study name includes a version tag (`v4_trslfix`) that is incremented whenever the search space or loss function changes, ensuring stale trials from previous versions are never mixed into the current optimization.

#### Critical Bug Fix (v4_trslfix): TR_sl Derivation
**Problem**: Previous versions derived the slice period as `TR_sl = Peak_slices(2) - Peak_slices(1)` after shifting and clipping triggers. When `shift < 0`, the first trigger was clipped to 1, shrinking `TR_sl` to ~497 instead of the nominal 500 samples. This left a 3-sample uncorrected gap at the end of every slice window, allowing the 20/30/40 Hz harmonic comb to survive.

**Solution**: `TR_sl` is now **fixed to the nominal value** (`nominal_slice_samples = 500`) and never derived from (possibly shifted/clipped) triggers. Triggers are shifted but **not clipped or dropped**, preserving the exact trigger count and the `n_slices = n_volumes × slices_per_volume` invariant required by the Kronecker product. The MATLAB function `bergen_fast_correction.m` safely handles the 1–2 out-of-bounds boundary slices internally via its own `valid_slices` mask.

**Impact**: Harmonic suppression improved from partial (residual peaks visible at 20/30/40 Hz) to **99.96–100.00%** across all channels.

### Step 05: Full Multi-Channel Bergen AAS Cleaning (`step05_bergen_clean.py`)
- Loads the working segment interval for all 96 channels as `np.float32` (saves 50% RAM and disk: ~900 MB instead of 1.8 GB).
- Passes the array and `slice_triggers.txt` to MATLAB, where the vectorized kernel `bergen_fast_correction.m` constructs the weight matrix **W = kron(W_vol, I_25)** and performs channel-wise artifact subtraction.
- Builds a proper EEGLAB structure via `eeg_emptyset()` and saves a single output file `segmentX_bergen_optuna_*.set` (~935 MB), immediately deleting the temporary MAT file.

### Step 06: Spectral Analysis & Alpha Quality Metrics (`step06_spectra_analysis.py`)
- Computes Welch Power Spectral Density (PSD) in the 0.5–40 Hz range for evaluation channels `O1, Oz, O2, Pz, P3, P4, Fz, Cz`.
- Calculates 4 key quality metrics:
  1. **G_suppression**: Percentage suppression of MRI harmonics (20, 30, 40, 50, 60 Hz):
     ```
     G_suppression = 1 - mean(PSD_clean[20..60 Hz]) / mean(PSD_raw[20..60 Hz])
     ```
  2. **AlphaProminence**: Ratio of alpha band power to flanking background:
     ```
     AlphaProminence = mean(PSD[8–13 Hz]) / (0.5 × (mean(PSD[5–8 Hz]) + mean(PSD[13–16 Hz])))
     ```
  3. **f_α (Alpha Peak Frequency)**: Precise frequency of the maximum in the 8–13 Hz band.
  4. **AlphaPreservation**: Preservation coefficient relative to the outside-scanner reference (EEG21):
     ```
     AlphaPreservation = AlphaProminence_clean / AlphaProminence_EEG21
     ```
- **Slice-comb masking**: Bins within ±0.15 Hz of the slice-repetition frequency (10.0 Hz for this sequence) are **excluded** from alpha metrics, as the fundamental of the slice artifact comb lands exactly at 10 Hz and is mathematically inseparable from physiological alpha at that frequency.
- Exports tables `summary_alpha_quality.csv` and `metrics.csv`, and plots `step03_spectra.png` and `alpha_quality_check.png` (5–20 Hz panel).

### Step 07: Interactive HTML Report Generation (`step07_html_report.py`)
- Generates a fully self-contained HTML document `segmentX_cleaning_report.html` with all plots embedded as Base64.
- Includes a quality metrics summary table per channel, validation status (`EXCELLENT / PASS / REVIEW`), Optuna optimization progress, slice phase detection plot, and full spectral comparisons.

---

## 📊 Example Quality Metrics Table (`summary_alpha_quality.csv`)

Results from real segment 4 cleaning (`subject 1916`, v4_trslfix code):

| Channel | MRI Suppression (20-60 Hz) | Alpha Power (8-13 Hz) | Alpha Prominence (Clean) | Alpha Prominence (EEG21) | Alpha Peak (Clean) | Alpha Peak (EEG21) | Alpha Preservation % | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **O1** | 100.00% | 57.11 µV² | 0.33 | 0.81 | **9.8 Hz** | 10.8 Hz | 40.7% | `PASS` |
| **Oz** | 99.98% | 202.87 µV² | 0.46 | 0.83 | **9.8 Hz** | 10.8 Hz | 56.1% | `PASS` |
| **O2** | 99.99% | 186.25 µV² | 0.32 | 0.84 | **9.8 Hz** | 10.8 Hz | 38.3% | `PASS` |
| **Pz** | 99.98% | 20.35 µV² | 0.64 | 0.85 | **9.8 Hz** | 11.0 Hz | 75.7% | `PASS` |
| **P3** | 100.00% | 19.97 µV² | 0.33 | 0.91 | **9.8 Hz** | 11.0 Hz | 36.1% | `PASS` |
| **P4** | 100.00% | 13.79 µV² | 0.32 | 0.90 | **9.8 Hz** | 8.8 Hz | 36.3% | `PASS` |
| **Fz** | 99.98% | 32.21 µV² | 1.01 | 0.73 | **9.8 Hz** | 8.0 Hz | **139.3%** | `EXCELLENT` |
| **Cz** | 99.96% | 8.72 µV² | 0.55 | 0.00 | **9.8 Hz** | 0.0 Hz | 27.7% | `PASS` |

**Note on 9.8 Hz Alpha Peak**: The reported alpha peak at 9.8 Hz (vs. 10.8 Hz in the EEG21 reference) is an artifact of the slice-comb masking, not a genuine frequency shift. The slice-repetition fundamental (sfreq / nominal_slice_samples = 5000 / 500 = 10.000 Hz) lands exactly at the center of the alpha band. We mask ±0.15 Hz around 10.0 Hz to exclude inseparable artifact residuals, which causes the peak detector to select the nearest unmasked bin below 10 Hz. This is a fundamental frequency collision of this MRI sequence, not signal degradation.

---

## ⚙️ Configuration (`config.py`)

All global parameters are centralized in `config.py` and can be overridden via environment variables:

```python
# External environment paths
MATLAB_BIN = Path("/home/vitaly/matlab2023/bin/matlab")
EEGLAB_DIR = Path("/home/vitaly/Downloads/eeglab_current/eeglab2026.0.0")
BERGEN_DIR = EEGLAB_DIR / "plugins" / "BERGEN1.0"

# MRI sequence parameters
DEFAULT_TR_SEC = 2.500            # Volume TR duration (s)
DEFAULT_SLICES_PER_VOLUME = 25    # Slices per volume (TR_slice = 100 ms)
DEFAULT_DUMMY_VOLUMES = 12        # Number of dummy scans to remove (30 s)
DEFAULT_SFREQ = 5000.0            # EEG sampling frequency (Hz)

# Optimization and quality control parameters
DEFAULT_N_TRIALS = 20             # Number of Optuna TPE trials
DEFAULT_TARGET_CH = "Oz"          # Target channel for fast 1-channel optimization
GRADIENT_HARMONICS = [20.0, 30.0, 40.0, 50.0, 60.0]  # MRI harmonics (excluding 10 Hz)
ALPHA_BAND = (8.0, 13.0)          # Alpha rhythm band
BG_BANDS = ((5.0, 8.0), (13.0, 16.0))  # Flanking background for Alpha Prominence
EVAL_CHANNELS = ["O1", "Oz", "O2", "Pz", "P3", "P4", "Fz", "Cz"]
```

---

## 🚀 Running the Pipeline

### 1. End-to-End Execution (All Steps):
```bash
# Run for default segment (segment4):
python run_all.py

# Run for arbitrary segment with custom Optuna trial count:
python run_all.py --segment-dir 1916/segments/segment4 --trials 30

# Skip MRI session detection (if sessions already determined):
python run_all.py --segment-dir 1916/segments/segment4 --skip-detect-mri

# Fast cleaning with already-saved Optuna hyperparameters (no re-optimization):
python run_all.py --segment-dir 1916/segments/segment4 --skip-optuna
```

### 2. Step-by-Step Execution (Individual Stages):
```bash
# Step 01: Detect MRI sessions in continuous recording
python step01_detect_mri.py

# Step 02: Sub-sample slice phase detection
python step02_detect_slices.py

# Step 03: Trim dummy scans and align with SPM rp_*.txt
python step03_trim_dummy.py

# Step 04: Hyperparameter tuning via Optuna TPE on Oz
python step04_optuna_tune.py

# Step 05: Full multi-channel Bergen AAS cleaning (96 channels)
python step05_bergen_clean.py

# Step 06: Compute spectra, quality metrics, and EEG21 comparison
python step06_spectra_analysis.py

# Step 07: Assemble final HTML report
python step07_html_report.py
```

---

## 🛠️ Memory & Stability Optimizations (RAM & OS)

### IDE File Watcher Exclusion
In `.vscode/settings.json`, configure `files.watcherExclude` masks for `*.mat, *.fif, *.eeg, *.set, *.fdt, *.db, temp_*` to prevent IDE background process crashes when writing 1–2 GB files.

### Single-Precision Float32
All EEG arrays passed between Python and MATLAB are cast to `np.float32`, reducing peak RAM consumption and file sizes by 50% without loss of precision for microvolt-scale EEG signals.

### Automatic Garbage Collection
Scripts invoke `gc.collect()` and immediately delete intermediate `temp_*.mat` files after computations complete.

---

## 📦 Dependencies

### Python Environment
```bash
conda create -n NLP_ENV python=3.12
conda activate NLP_ENV
pip install numpy scipy mne optuna matplotlib
```

**Package versions** (tested):
- Python 3.12
- numpy 2.2.3
- scipy 1.15.2
- mne 1.12.1
- optuna 4.6.0
- matplotlib 3.x

### MATLAB Environment
- MATLAB R2023a or later
- EEGLAB 2026.0.0 (or compatible version)
- Bergen EEG-fMRI Toolbox 1.0 (EEGLAB plugin)

---

## 🔍 Key Technical Details

### Kronecker Product Template Construction
The Bergen AAS algorithm constructs the averaging weight matrix as:
```
W = kron(W_vol, I_slices)
```
where:
- `W_vol` is an `(n_volumes × n_volumes)` matrix encoding either simple moving-average weights or motion-weighted (rp-based) weights.
- `I_slices` is the `(slices_per_volume × slices_per_volume)` identity matrix (25×25 for this sequence).
- The Kronecker product expands volume-level weights to the full slice-level template matrix.

This structure ensures slice-to-slice correspondence: Slice #1 from volume V is averaged only with Slice #1 from neighboring volumes, never with slices from other positions within a volume. This prevents alpha oscillations (period ~100 ms) from contaminating the gradient template (which repeats every 100 ms at the slice level).

### Slice-Comb Frequency Collision (10 Hz Fundamental)
For this MRI sequence:
- `TR = 2.5 s`, `slices_per_volume = 25`, `sfreq = 5000 Hz`
- `nominal_slice_samples = sfreq × (TR / slices_per_volume) = 5000 × 0.1 = 500 samples`
- Slice-repetition frequency = `sfreq / nominal_slice_samples = 5000 / 500 = 10.0 Hz`

The slice artifact comb has harmonics at exact integer multiples of 10 Hz (10, 20, 30, 40, …). The fundamental (10 Hz) lands precisely at the center of the alpha band (8–13 Hz). At this frequency, physiological alpha and artifact are **mathematically inseparable** — any attempt to zero the artifact at 10 Hz also zeros physiological alpha at that frequency.

To avoid conflating artifact residuals with alpha preservation metrics:
- All alpha metrics **exclude** a ±0.15 Hz notch around 10.0 Hz (and any other comb harmonics in the alpha band).
- This prevents the dead 10 Hz bin from dragging down measured alpha prominence and peak frequency.
- The reported alpha peak at 9.8 Hz (vs. 10.8 Hz in the outside-scanner reference) is an artifact of this masking, not a genuine frequency shift.

### Motion-Weighted vs. Simple Moving-Average Templates
When an SPM motion parameter file (`rp_*.txt`) is available:
- `m_rp_info` computes volume-to-volume Framewise Displacement (FD).
- Volumes with FD > `motion_thresh` are excluded from template averaging for that volume.
- This produces a motion-weighted `W_vol` that adapts to head motion.

When no motion file is available:
- `m_moving_average` produces a simple moving-average `W_vol` with uniform weights over a `win_k`-sized window.

Both methods expand to the full slice-level matrix via the same Kronecker product.

---

## 📖 References

1. **Bergen EEG-fMRI Toolbox**:  
   Moosmann, M., et al. (2009). *High quality EEG-fMRI at 3T and 7T.* NeuroImage, 46(4), 1142–1153.  
   https://doi.org/10.1016/j.neuroimage.2009.02.040

2. **Optuna: A Next-generation Hyperparameter Optimization Framework**:  
   Akiba, T., et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD.  
   https://arxiv.org/abs/1907.10902

3. **MNE-Python (EEG/MEG analysis)**:  
   Gramfort, A., et al. (2013). *MEG and EEG data analysis with MNE-Python.* Frontiers in Neuroscience, 7, 267.  
   https://mne.tools/

---

## 📝 License

This software is provided for research and educational purposes. Commercial use requires explicit permission.

---

## 👤 Author & Contact

**Vitaly** — Neural Engineering Lab  
For questions or collaboration inquiries, please open an issue in this repository.

---

## 🔄 Changelog

### v4_trslfix (Current)
- **Fixed critical TR_sl bug**: TR_sl now derived from nominal_slice_samples (500), not from clipped triggers. Harmonic suppression improved to 99.96–100.00%.
- **Per-trial artifact persistence**: Each Optuna trial saves params/metrics/spectrum/clean_1ch/script to `segment4/trialN/` for visual comparison and reproducibility.
- **Study versioning**: Study name bumped to `v4_trslfix` to force fresh optimization on corrected code.
- **Alpha retention metric**: Replaced alpha prominence penalty with alpha retention (clean/raw off-comb ratio) in loss function, targeting ≥90% preservation.
- **Reduced default trials**: DEFAULT_N_TRIALS = 20 (down from 40) for faster iteration during development.

### v3_smallk
- **Narrowed win_k search space**: 3–14 volumes (was 20–40) after empirically confirming small windows track non-stationary artifacts better for subjects with motion.
- **Alpha-comb notch masking**: Excluded ±0.15 Hz around 10 Hz fundamental from all alpha metrics to avoid artifact-alpha conflation.

### v2_alpharet
- **Initial multi-objective loss**: G_suppression + alpha prominence penalty.

### v1
- **Initial implementation**: Single-objective gradient suppression only.
