"""
Configuration module for Optuna-Bergen EEG-fMRI Cleaning Pipeline.
Provides central paths and default experimental parameters with easy override.
"""
from pathlib import Path
import os

# Base paths
PROJECT_ROOT = Path(__file__).parent.resolve()
MATLAB_BIN = Path(os.getenv("MATLAB_BIN", "/home/vitaly/matlab2023/bin/matlab"))
EEGLAB_DIR = Path(os.getenv("EEGLAB_DIR", "/home/vitaly/Downloads/eeglab_current/eeglab2026.0.0"))
BERGEN_DIR = Path(os.getenv("BERGEN_DIR", str(EEGLAB_DIR / "plugins" / "BERGEN1.0")))

# Default subject & segment
DEFAULT_EXPERIMENT = "1916"
DEFAULT_SEGMENT_DIR = PROJECT_ROOT / DEFAULT_EXPERIMENT / "segments" / "segment4"
DEFAULT_RAW_VHDR = PROJECT_ROOT / "EEG_files" / DEFAULT_EXPERIMENT / f"{DEFAULT_EXPERIMENT}_inside.vhdr"

# fMRI sequence default parameters
DEFAULT_TR_SEC = 2.500
DEFAULT_SLICES_PER_VOLUME = 25
DEFAULT_DUMMY_VOLUMES = 12
DEFAULT_SFREQ = 5000.0  # Hz

# Optuna default parameters
DEFAULT_N_TRIALS = 20
DEFAULT_TARGET_CH = "Oz"
GRADIENT_HARMONICS = [20.0, 30.0, 40.0, 50.0, 60.0]  # Exclude 10 Hz to prevent interfering with alpha
ALPHA_BAND = (8.0, 13.0)
BG_BANDS = ((5.0, 8.0), (13.0, 16.0))
EVAL_CHANNELS = ["O1", "Oz", "O2", "Pz", "P3", "P4", "Fz", "Cz"]
MIN_ALPHA_PRESERVATION = 0.80

