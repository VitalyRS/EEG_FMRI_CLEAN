# 🚀 Quick Start: ICA Pipeline with Optuna Optimization

## Overview

The new ICA pipeline (steps 08-11) adds automatic removal of residual artifacts (BCG, muscle, eye blinks) after Bergen gradient cleaning, with full Bayesian parameter optimization.

## Pipeline Flow

```
Bergen-cleaned .set (step 05)
         ↓
    Downsample to 250 Hz + convert to MNE .fif
         ↓
[Step 08] Optimize BCG removal parameters (Optuna TPE)
         ├─ pre_filt: bandpass before peak detection
         ├─ l_freq, h_freq: final bandpass after removal
         └─ corr_thresh: template correlation threshold
         ↓
[Step 09] Apply BCG removal → .../03_bcg/segmentX_bcg_clean.fif
         ↓
[Step 10] Optimize ICA + clean_rawdata parameters (Optuna TPE, 60s subset)
         ├─ FlatlineCriterion, ChannelCriterion, LineNoiseCriterion
         ├─ BurstCriterion, WindowCriterion
         └─ ICLabel rejection threshold (0.60-0.90)
         ↓
[Step 11] Apply optimized ICA to full data
         ├─ Bad channel interpolation
         ├─ Extended Infomax ICA (25 components)
         ├─ ICLabel classification
         ├─ Artifact component removal
         └─ Save → .../05_ica/segmentX_ica_clean.fif + HTML report
```

## Quick Commands

### 1. Full pipeline (Bergen → BCG → ICA):
```bash
python run_all.py
```

### 2. Bergen only (skip BCG and ICA):
```bash
python run_all.py --skip-bcg --skip-ica
```

### 3. Re-run ICA with more trials (skip Bergen):
```bash
python run_all.py --skip-optuna --trials 30
```

### 4. Manual step-by-step (after Bergen step 05):
```bash
# Optimize BCG parameters (20 trials, ~5-10 min)
python step08_bcg_optuna.py

# Apply BCG removal
python step09_ica.py

# Optimize ICA parameters (20 trials, ~40-60 min — ICA is slow!)
python step10_optuna_ica.py

# Apply ICA to full data (~3-5 min)
python step11_ica_final.py
```

## Output Files

- **BCG outputs**:
  - `segment4/bcg_optuna_best.json` — winning parameters
  - `segment4/bcg_optuna_study.db` — SQLite trial history
  - `segment4/bcg_optuna_result.png` — optimization plot
  - `data/1916/derivatives/03_bcg/segment4/segment4_bcg_clean.fif` — BCG-cleaned data

- **ICA outputs**:
  - `segment4/ica_optuna_best.json` — winning parameters
  - `segment4/ica_optuna_study.db` — SQLite trial history
  - `segment4/ica_optuna_result.png` — optimization plot
  - `data/1916/derivatives/05_ica/segment4/segment4_ica_clean.fif` — final clean data
  - `data/1916/derivatives/05_ica/segment4/segment4_ica_report.html` — component report

## Expected Optimization Time

- **Step 08 (BCG Optuna)**: ~5-10 minutes (20 trials, no ICA)
- **Step 09 (BCG apply)**: ~1-2 minutes
- **Step 10 (ICA Optuna)**: ~40-60 minutes (20 trials × 2-3 min per ICA)
- **Step 11 (ICA apply)**: ~3-5 minutes

**Total**: ~50-80 minutes for full BCG+ICA pipeline with default 20 trials.

## Metrics

### BCG Optimization
- **BCG suppression**: power reduction in 1-5 Hz (cardiac band)
- **Alpha retention**: preservation of 8-13 Hz power (target ≥ 90%)

### ICA Optimization
- **Alpha retention**: 8-13 Hz preservation after ICA (target ≥ 85%)
- **Variance drop**: fraction of total variance removed (target ≤ 15%)
- **Bad channels removed**: by clean_rawdata (penalized quadratically)

## Troubleshooting

### "No module named 'mne.preprocessing.iclabel'"
You need MNE-ICALabel:
```bash
conda activate NLP_ENV
pip install mne-icalabel
```

### "MATLAB not found" during steps 08-11
BCG and ICA steps (08-11) are **pure Python** — no MATLAB needed! Only Bergen steps (01-05) require MATLAB.

### ICA optimization is too slow
- Reduce trials: `--trials 10`
- Or skip optimization and use defaults: `--skip-ica-optuna`
- The 60s subset in step 10 is already optimized for speed; full-data ICA would take hours per trial.

### Want to re-optimize with different settings
Delete the study database to start fresh:
```bash
rm 1916/segments/segment4/ica_optuna_study.db
python step10_optuna_ica.py
```

## Example Best Parameters (segment4)

**BCG (typical)**:
```json
{
  "pre_filt": [0.5, 5.0],
  "l_freq": 0.5,
  "h_freq": 40.0,
  "corr_thresh": 0.75
}
```

**ICA (typical)**:
```json
{
  "flatline_crit": 5,
  "channel_crit": 0.8,
  "line_crit": 4,
  "burst_crit": 25,
  "window_crit": 0.3,
  "iclabel_thresh": 0.75
}
```

## Notes

- **Step 10 uses first 60 seconds only** to keep optimization fast. Final ICA (step 11) applies found parameters to full data.
- **ICLabel threshold**: higher (0.85-0.90) → aggressive cleaning, more variance drop; lower (0.60-0.70) → conservative, preserves more signal.
- **clean_rawdata ASR is disabled** — we optimize only bad-channel detection and line noise removal.
- All intermediate `.fif` files are stored in `data/1916/derivatives/` following BIDS-like structure.
