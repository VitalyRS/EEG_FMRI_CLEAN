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
# New "data/<subject>" layout (see README):
#   data/1916/raw/eeg96/1916.vhdr           -> continuous raw recording
#   data/1916/raw/rp_spm/rp_segment04.txt   -> SPM motion params (per-segment)
#   data/1916/segments/segment04/           -> per-segment working directory
DEFAULT_EXPERIMENT = "1916"
DATA_ROOT = PROJECT_ROOT / "data"
SUBJECT_DIR = DATA_ROOT / DEFAULT_EXPERIMENT
RAW_DIR = SUBJECT_DIR / "raw"
SEGMENTS_DIR = SUBJECT_DIR / "segments"
RP_DIR = RAW_DIR / "rp_spm"   # subject-level SPM rp_*.txt (one per segment)

DEFAULT_SEGMENT_DIR = SEGMENTS_DIR / "segment04"
DEFAULT_RAW_VHDR = RAW_DIR / "eeg96" / f"{DEFAULT_EXPERIMENT}.vhdr"

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

