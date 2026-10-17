"""
STEP 10: Fast Two-Phase Optimization of ICA Pipeline Parameters
================================================================
The old design re-ran the expensive `runica` (5-10 min) inside EVERY Optuna
trial, even though most trials only varied the ICLabel threshold. That is
wasteful: ICLabel classifications do NOT depend on the threshold -- the
threshold only decides WHICH already-classified ICs to reject, and that
rejection is a cheap matrix projection.

This version splits the work so ICA runs only ONCE:

  Phase 1  (one MATLAB session, ~1 min):
      Grid-search clean_rawdata bad-channel detection params
      (flatline / channel / line criteria). No ICA. Pick the param set that
      removes genuinely-bad channels while preserving good/central ones.

  Phase 2  (one MATLAB session, ~7 min):
      With the Phase-1 winner, run clean_rawdata + runica + ICLabel ONCE.
      Save icaweights / icasphere / icawinv / ICLabel classifications /
      channel-pruned data to a .mat file.

  Phase 3  (pure Python, ~seconds):
      Sweep iclabel_thresh over the SAVED classifications. For each threshold,
      decide rejected ICs, zero them, project back (icawinv @ act), and score.
      Pick the threshold with the best loss.

Total wall-clock: ~8-10 min instead of ~3 hours, with the same search intent.
Output JSON format is unchanged, so step11 / run_all.py need no edits.
Optimization runs on the first OPTUNA_DURATION_SEC of BCG-cleaned data.
"""
from pathlib import Path
import subprocess
import json
import numpy as np
import mne
from scipy.io import savemat, loadmat
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .config import (MATLAB_BIN, EEGLAB_DIR, PROJECT_ROOT,
                         DEFAULT_SEGMENT_DIR, ALPHA_BAND, GRADIENT_HARMONICS,
                         ICLABEL_OTHER_BRAIN_MAX)
except ImportError:
    from config import (MATLAB_BIN, EEGLAB_DIR, PROJECT_ROOT,
                         DEFAULT_SEGMENT_DIR, ALPHA_BAND, GRADIENT_HARMONICS,
                         ICLABEL_OTHER_BRAIN_MAX)

DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_EXPERIMENT = "1916"

# For optimization speed: use only first N seconds (runica is expensive)
OPTUNA_DURATION_SEC = 60.0

# Maximum fraction of channels that can be removed before a clean_rawdata
# candidate is disqualified in Phase 1.
MAX_CHANNEL_REMOVAL_FRACTION = 0.20  # 20%

DIPFIT_DIR = EEGLAB_DIR / "plugins" / "dipfit"
STD_1005 = DIPFIT_DIR / "standard_BEM" / "elec" / "standard_1005.elc"

# ---- Phase 1 search grid for clean_rawdata bad-channel detection ----
# channel_crit is the dominant driver of channel removal; flatline/line rarely
# fire on well-preprocessed data, so keep those small and fixed-ish.
FLATLINE_GRID = [5.0]
CHANNEL_GRID = [0.70, 0.75, 0.80, 0.85]
LINE_GRID = [4.0, 6.0]

# ---- Phase-ASR grid for BurstCriterion (k = SD cutoff of the ASR reconstruction).
# Higher k = gentler (fewer bursts touched). 'off' disables ASR entirely (the old
# behaviour, kept as the conservative anchor). We sweep from gentle -> aggressive
# and let scoring pick the SOFTEST k that still buys real artifact suppression,
# so the neural signal is preserved as much as possible (the user's goal).
BURST_GRID = ["off", 30.0, 20.0, 10.0]

# Above ~40 Hz the reference band (80 Hz LP) still carries residual muscle / MRI
# vibration that ASR is meant to knock down. We use the drop in 40-80 Hz power as
# the "artifact suppression" signal and 8-13 Hz (alpha) retention as the "signal
# preserved" signal when picking k.
HF_ARTIFACT_BAND = (40.0, 80.0)

# Minimum HF-power drop (fraction) an ASR setting must deliver over 'off' before
# we consider it worth the risk to the signal. Below this, ASR is doing nothing
# useful and we keep the gentler / 'off' setting.
ASR_MIN_HF_GAIN = 0.05      # 5% extra 40-80 Hz suppression vs 'off'
# Hard floor on alpha retention: an ASR k that drops alpha below this is
# over-correcting into neural territory and is disqualified.
ASR_MIN_ALPHA_RET = 0.90

# ---- Phase 3 threshold sweep ----
THRESHOLD_GRID = [0.65, 0.70, 0.75, 0.80, 0.85]

# Central / midline channels we never want clean_rawdata to throw away: their
# neighbour-correlation is naturally lower, so an aggressive channel_crit tends
# to (wrongly) flag them. Removing these is a strong signal of over-cleaning.
PROTECTED_CHANNELS = {
    "Fz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz", "AFz",
    "F1", "F2", "FC1", "FC2", "C1", "C2", "CP1", "CP2", "P1", "P2",
}

# ICLabel class order (7 columns): probabilities sum to 1 per IC.
ICLABEL_CLASSES = ["Brain", "Muscle", "Eye", "Heart", "Line", "Chan", "Other"]
ARTIFACT_IDX = [1, 2, 3, 4, 5]   # Muscle, Eye, Heart, Line, Chan (0-based)
OTHER_IDX = 6
BRAIN_IDX = 0

# Rule-2 cutoff: reject an 'Other'-dominant IC only if P(Brain) < this value.
OTHER_BRAIN_MAX = ICLABEL_OTHER_BRAIN_MAX
# Optional cap on the number of ICA components (PCA reduction before runica).
# None = only reduce when the data is rank-deficient (original behaviour).
ICA_PCA_MAX = None
# High-pass filter exclusively for ICA training (Hz).
ICA_HP_HZ = 1.0
# ICLabel backend: "python" (mne-icalabel, recommended) or "matlab"
ICLABEL_BACKEND = "python"



def _pca_cap_ml() -> str:
    """MATLAB line that caps n_pca at ICA_PCA_MAX (empty if no cap)."""
    return f"n_pca = min(n_pca, {int(ICA_PCA_MAX)});" if ICA_PCA_MAX else ""


def _find_bcg_fif(segment_dir: Path) -> Path:
    seg = segment_dir.name
    subject_id = segment_dir.parent.parent.name if segment_dir.parent.name == "segments" else segment_dir.parent.name
    p = DATA_ROOT / subject_id / "derivatives" / "03_bcg" / seg / f"{seg}_bcg_clean.fif"
    if not p.exists():
        raise FileNotFoundError(f"BCG input not found: {p}. Run step08 first.")
    return p


# ---------------------------------------------------------------------------
# Phase 1: grid-search clean_rawdata bad-channel detection (no ICA)
# ---------------------------------------------------------------------------
def run_phase1_channels(work_dir: Path, mat_in: Path) -> list[dict]:
    """
    Run clean_rawdata over the FLATLINE/CHANNEL/LINE grid in ONE MATLAB session.
    Returns a list of {flatline_crit, channel_crit, line_crit, n_removed,
    removed_chans} dicts (one per grid point). ICA is NOT run here.
    """
    out_mat = work_dir / "phase1_channels.mat"
    m_file = work_dir / "phase1_channels.m"

    clean_rawdata_plugin = EEGLAB_DIR / "plugins" / "clean_rawdata"

    flat_ml = "[" + " ".join(f"{v}" for v in FLATLINE_GRID) + "]"
    ch_ml = "[" + " ".join(f"{v}" for v in CHANNEL_GRID) + "]"
    line_ml = "[" + " ".join(f"{v}" for v in LINE_GRID) + "]"

    code = f"""
addpath('{EEGLAB_DIR.resolve()}');
addpath('{clean_rawdata_plugin.resolve()}');
addpath('{DIPFIT_DIR.resolve()}');
eeglab nogui;

load('{mat_in.resolve()}', 'data', 'srate', 'labels');
EEG = eeg_emptyset();
EEG.setname = 'phase1';
EEG.data = double(data);
EEG.srate = double(srate);
EEG.nbchan = size(data, 1);
EEG.pnts = size(data, 2);
EEG.trials = 1;
EEG.xmin = 0;
EEG.xmax = (EEG.pnts - 1) / EEG.srate;

% Build chanlocs from labels
EEG.chanlocs = struct([]);
for i = 1:EEG.nbchan
    if iscell(labels)
        EEG.chanlocs(i).labels = char(labels{{i}});
    else
        EEG.chanlocs(i).labels = deblank(labels(i,:));
    end
end
EEG = eeg_checkset(EEG);
EEG = pop_chanedit(EEG, 'lookup', '{STD_1005.resolve()}');
EEG = eeg_checkset(EEG);
orig_chans = {{EEG.chanlocs.labels}};

flat_grid = {flat_ml};
ch_grid   = {ch_ml};
line_grid = {line_ml};

idx = 0;
res = struct('flatline_crit', {{}}, 'channel_crit', {{}}, 'line_crit', {{}}, ...
             'n_removed', {{}}, 'removed', {{}});
for fi = 1:numel(flat_grid)
  for ci = 1:numel(ch_grid)
    for li = 1:numel(line_grid)
      idx = idx + 1;
      EEG_c = pop_clean_rawdata(EEG, ...
          'FlatlineCriterion', flat_grid(fi), ...
          'ChannelCriterion', ch_grid(ci), ...
          'LineNoiseCriterion', line_grid(li), ...
          'Highpass', 'off', ...
          'BurstCriterion', 'off', ...
          'WindowCriterion', 'off', ...
          'BurstRejection', 'off', ...
          'Distance', 'Euclidian');
      kept = {{EEG_c.chanlocs.labels}};
      removed = setdiff(orig_chans, kept);
      res(idx).flatline_crit = flat_grid(fi);
      res(idx).channel_crit  = ch_grid(ci);
      res(idx).line_crit     = line_grid(li);
      res(idx).n_removed     = numel(removed);
      if isempty(removed)
        res(idx).removed = '';
      else
        res(idx).removed = strjoin(removed, ',');
      end
      fprintf('P1 flat=%g ch=%g line=%g -> removed %d\\n', ...
              flat_grid(fi), ch_grid(ci), line_grid(li), numel(removed));
    end
  end
end
save('{out_mat.resolve()}', 'res', '-v7');
clear EEG EEG_c;
quit force;
"""
    with open(m_file, "w", encoding="ascii") as f:
        f.write(code)

    print("  [Phase 1] Grid-searching clean_rawdata parameters (no ICA)...")
    try:
        res = subprocess.run(
            [str(MATLAB_BIN), "-nodisplay", "-nodesktop", "-nosplash", "-batch", f"run('{m_file.resolve()}')"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900
        )
    except subprocess.TimeoutExpired:
        print("  [Phase 1] MATLAB timed out.")
        m_file.unlink(missing_ok=True)
        return []
    m_file.unlink(missing_ok=True)

    if not out_mat.exists():
        print(f"  [Phase 1] MATLAB failed (rc={res.returncode}):")
        print(res.stdout[-1500:] if res.stdout else "(no output)")
        return []

    m = loadmat(out_mat, squeeze_me=True, struct_as_record=False)
    out_mat.unlink(missing_ok=True)
    raw_res = m["res"]
    if not isinstance(raw_res, np.ndarray):
        raw_res = [raw_res]

    results = []
    for r in np.atleast_1d(raw_res):
        removed_str = str(r.removed) if hasattr(r, "removed") else ""
        removed = [c for c in removed_str.split(",") if c] if removed_str else []
        results.append({
            "flatline_crit": float(r.flatline_crit),
            "channel_crit": float(r.channel_crit),
            "line_crit": float(r.line_crit),
            "n_removed": int(r.n_removed),
            "removed_chans": removed,
        })
    return results


def score_phase1(candidate: dict, n_channels_original: int) -> float:
    """
    Score a clean_rawdata candidate (lower is better). We bias CONSERVATIVE
    (our failure mode was over-removal): prefer removing few channels and never
    protected/central ones. Removing zero is fine -- better under- than
    over-clean at the channel stage.
    """
    n_removed = candidate["n_removed"]
    removed = set(candidate["removed_chans"])
    n_central = len(removed & PROTECTED_CHANNELS)

    # Disqualify configs that remove too many channels outright.
    if n_removed / max(n_channels_original, 1) > MAX_CHANNEL_REMOVAL_FRACTION:
        return 1e9 + n_removed

    # Heavy penalty per protected channel removed; mild penalty per channel;
    # tiny tie-break toward channel_crit ~0.80 for robustness.
    score = (20.0 * n_central
             + 1.0 * n_removed
             + 0.5 * abs(candidate["channel_crit"] - 0.80))
    return score


# ---------------------------------------------------------------------------
# Phase-ASR: sweep BurstCriterion (k) in ONE MATLAB session, return cleaned data
# ---------------------------------------------------------------------------
def run_phase_asr(work_dir: Path, mat_in: Path, best_ch: dict) -> list[dict]:
    """
    With the Phase-1 winning channel params fixed, run clean_rawdata for each k
    in BURST_GRID (BurstRejection='off' => reconstruct-in-place, duration kept)
    inside ONE MATLAB session. Returns [{burst_crit, data (n_ch x n_samp, uV)}].

    ASR is a MATLAB clean_rawdata operation, so unlike the ICLabel threshold it
    cannot be swept in pure Python -- but running all k in a single MATLAB call
    keeps it cheap (channel detection + ASR only, no ICA).
    """
    out_mat = work_dir / "phase_asr.mat"
    m_file = work_dir / "phase_asr.m"
    clean_rawdata_plugin = EEGLAB_DIR / "plugins" / "clean_rawdata"

    # MATLAB cell of burst settings ('off' or numeric).
    burst_items = ["'off'" if (isinstance(b, str) and b == "off") else f"{float(b)}"
                   for b in BURST_GRID]
    burst_ml = "{" + ", ".join(burst_items) + "}"

    code = f"""
addpath('{EEGLAB_DIR.resolve()}');
addpath('{clean_rawdata_plugin.resolve()}');
addpath('{DIPFIT_DIR.resolve()}');
eeglab nogui;

load('{mat_in.resolve()}', 'data', 'srate', 'labels');
EEG = eeg_emptyset();
EEG.setname = 'phase_asr';
EEG.data = double(data);
EEG.srate = double(srate);
EEG.nbchan = size(data, 1);
EEG.pnts = size(data, 2);
EEG.trials = 1;
EEG.xmin = 0;
EEG.xmax = (EEG.pnts - 1) / EEG.srate;
EEG.chanlocs = struct([]);
for i = 1:EEG.nbchan
    if iscell(labels)
        EEG.chanlocs(i).labels = char(labels{{i}});
    else
        EEG.chanlocs(i).labels = deblank(labels(i,:));
    end
end
EEG = eeg_checkset(EEG);
EEG = pop_chanedit(EEG, 'lookup', '{STD_1005.resolve()}');
EEG = eeg_checkset(EEG);

burst_grid = {burst_ml};
res = struct('burst', {{}}, 'data', {{}});
for bi = 1:numel(burst_grid)
    bc = burst_grid{{bi}};
    % Same channel params for every k so the ONLY difference is ASR strength.
    EEG_c = pop_clean_rawdata(EEG, ...
        'FlatlineCriterion', {best_ch['flatline_crit']}, ...
        'ChannelCriterion', {best_ch['channel_crit']}, ...
        'LineNoiseCriterion', {best_ch['line_crit']}, ...
        'Highpass', 'off', ...
        'BurstCriterion', bc, ...
        'WindowCriterion', 'off', ...
        'BurstRejection', 'off', ...
        'Distance', 'Euclidian');
    % Re-project onto the original channel set so every k has the same shape
    % (channel detection can differ slightly; interpolate back for fair PSD).
    if EEG_c.nbchan < EEG.nbchan
        EEG_c = pop_interp(EEG_c, EEG.chanlocs, 'spherical');
    end
    if ischar(bc)
        res(bi).burst = -1;   % sentinel for 'off'
    else
        res(bi).burst = bc;
    end
    res(bi).data = single(EEG_c.data);
end
save('{out_mat.resolve()}', 'res', '-v7');
clear EEG EEG_c;
quit force;
"""
    with open(m_file, "w", encoding="ascii") as f:
        f.write(code)

    print(f"  [Phase-ASR] Sweeping BurstCriterion {BURST_GRID} (one MATLAB session)...")
    out_mat.unlink(missing_ok=True)
    import time
    proc = subprocess.Popen(
        [str(MATLAB_BIN), "-nodisplay", "-nodesktop", "-nosplash", "-batch", f"run('{m_file.resolve()}')"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    start_t = time.time()
    while proc.poll() is None:
        if time.time() - start_t > 1800:
            print("  [Phase-ASR] MATLAB timed out; skipping ASR tuning (keeping 'off').")
            proc.kill()
            break
        if out_mat.exists() and (time.time() - out_mat.stat().st_mtime > 3):
            # File is written and settled; MATLAB may be stuck in futex on exit
            time.sleep(1)
            proc.kill()
            break
        time.sleep(2)
    m_file.unlink(missing_ok=True)

    if not out_mat.exists():
        print("  [Phase-ASR] MATLAB script failed to produce phase_asr.mat; keeping ASR 'off'.")
        return []

    m = loadmat(out_mat, squeeze_me=False, struct_as_record=False)
    out_mat.unlink(missing_ok=True)
    raw_res = np.atleast_1d(m["res"].ravel())

    out = []
    for rr in raw_res:
        b = float(np.asarray(rr.burst).ravel()[0])
        burst = "off" if b < 0 else b
        out.append({
            "burst_crit": burst,
            "data": np.asarray(rr.data, dtype=np.float64) * 1e-6,  # uV -> V
        })
    return out


def score_phase_asr(asr_results: list[dict], sfreq: float) -> dict:
    """
    Pick the SOFTEST BurstCriterion k that still delivers real high-frequency
    artifact suppression without eating alpha. 'off' is the baseline.

    For each k:  hf_drop  = 1 - P_hf(k) / P_hf(off)      (40-80 Hz)  [want high]
                 alpha_ret = P_alpha(k) / P_alpha(off)    (8-13 Hz)   [want ~1]

    Selection: among settings meeting ASR_MIN_ALPHA_RET, choose the one whose
    hf_drop beats 'off' by >= ASR_MIN_HF_GAIN; if several qualify, take the
    gentlest (largest k). If none qualify, keep 'off' (preserve the signal).
    """
    def band_power(data, lo, hi):
        nperseg = int(min(4 * sfreq, data.shape[-1]))
        f, psd = welch(data, sfreq, nperseg=nperseg, axis=-1)
        m = (f >= lo) & (f <= hi)
        return float(np.mean(psd[:, m]))

    base = next((r for r in asr_results if r["burst_crit"] == "off"), None)
    if base is None:
        base = asr_results[0]
    p_hf_off = band_power(base["data"], *HF_ARTIFACT_BAND)
    p_alpha_off = band_power(base["data"], *ALPHA_BAND)

    scored = []
    for r in asr_results:
        p_hf = band_power(r["data"], *HF_ARTIFACT_BAND)
        p_alpha = band_power(r["data"], *ALPHA_BAND)
        hf_drop = 1.0 - p_hf / max(p_hf_off, 1e-20)
        alpha_ret = p_alpha / max(p_alpha_off, 1e-20)
        scored.append({
            "burst_crit": r["burst_crit"],
            "hf_drop": float(hf_drop),
            "alpha_ret": float(alpha_ret),
        })
        tag = "off" if r["burst_crit"] == "off" else f"k={r['burst_crit']:.0f}"
        print(f"    [Phase-ASR] {tag:>6s}: HF-drop={hf_drop:+.1%}  alpha-ret={alpha_ret:.2f}")

    # Numeric candidates that preserve alpha AND actually suppress HF artifact.
    qualifiers = [
        s for s in scored
        if s["burst_crit"] != "off"
        and s["alpha_ret"] >= ASR_MIN_ALPHA_RET
        and s["hf_drop"] >= ASR_MIN_HF_GAIN
    ]
    if qualifiers:
        # Softest = largest k among qualifiers (preserve signal maximally).
        best = max(qualifiers, key=lambda s: s["burst_crit"])
        print(f"    [Phase-ASR] -> chose k={best['burst_crit']:.0f} "
              f"(HF-drop {best['hf_drop']:.1%}, alpha {best['alpha_ret']:.2f})")
    else:
        best = {"burst_crit": "off", "hf_drop": 0.0, "alpha_ret": 1.0}
        print("    [Phase-ASR] -> no k beat 'off' safely; keeping ASR off.")
    return best


# ---------------------------------------------------------------------------
def run_phase2_ica(work_dir: Path, mat_in: Path, best_ch: dict) -> dict | None:
    clean_rawdata_plugin = EEGLAB_DIR / "plugins" / "clean_rawdata"
    iclabel_plugin = EEGLAB_DIR / "plugins" / "ICLabel"

    out_mat = work_dir / "phase2_ica.mat"
    
    # --- 1. MATLAB: clean_rawdata ---
    m_clean = work_dir / "phase2_clean.m"
    code_clean = f"""
addpath('{EEGLAB_DIR.resolve()}');
addpath('{clean_rawdata_plugin.resolve()}');
eeglab nogui;

load('{mat_in.resolve()}', 'data', 'srate', 'labels');
EEG = eeg_emptyset();
EEG.data = double(data);
EEG.srate = double(srate);
EEG.nbchan = size(data, 1);
EEG.pnts = size(data, 2);
EEG.xmin = 0;
EEG.xmax = (EEG.pnts - 1) / EEG.srate;
EEG.chanlocs = struct([]);
for i = 1:EEG.nbchan
    if iscell(labels)
        EEG.chanlocs(i).labels = char(labels{{i}});
    else
        EEG.chanlocs(i).labels = deblank(labels(i,:));
    end
end
EEG = eeg_checkset(EEG);
EEG = pop_chanedit(EEG, 'lookup', '{STD_1005.resolve()}');

EEG_clean = pop_clean_rawdata(EEG, ...
    'FlatlineCriterion', {best_ch['flatline_crit']}, ...
    'ChannelCriterion', {best_ch['channel_crit']}, ...
    'LineNoiseCriterion', {best_ch['line_crit']}, ...
    'Highpass', 'off', 'BurstCriterion', 'off', 'WindowCriterion', 'off', ...
    'BurstRejection', 'off', 'Distance', 'Euclidian');

pop_saveset(EEG_clean, 'filename', 'phase2_clean.set', 'filepath', '{work_dir.resolve().as_posix()}');
quit force;
"""
    m_clean.write_text(code_clean, encoding="ascii")
    print(f"  [Phase 2] Running clean_rawdata in MATLAB (ch={best_ch['channel_crit']}, flat={best_ch['flatline_crit']}, line={best_ch['line_crit']})...")
    res = subprocess.run([str(MATLAB_BIN), "-nodisplay", "-nodesktop", "-nosplash", "-batch", f"run('{m_clean.resolve()}')"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    clean_set = work_dir / "phase2_clean.set"
    if not clean_set.exists():
        print("  [Phase 2] MATLAB clean_rawdata failed.")
        print(res.stdout[-1500:] if res.stdout else "")
        return None

    if ICLABEL_BACKEND == "python":
        import mne
        from mne.preprocessing import ICA
        from mne_icalabel.iclabel import iclabel_label_components
        import scipy.io as sio

        raw_clean = mne.io.read_raw_eeglab(clean_set, preload=True, verbose=False)
        kept_labels = list(raw_clean.ch_names)

        mat_orig = sio.loadmat(mat_in)
        labels_raw = mat_orig["labels"]
        if hasattr(labels_raw, "flatten"):
            orig_chans = [str(x[0]) if len(x) > 0 and isinstance(x, (np.ndarray, list)) else str(x).strip() for x in labels_raw.flatten()]
        else:
            orig_chans = kept_labels
        removed = [c for c in orig_chans if c not in kept_labels]

        raw_for_ica = raw_clean.copy().filter(l_freq=1.0, h_freq=100.0, verbose=False).set_eeg_reference('average', verbose=False)
        data_rank = np.linalg.matrix_rank(raw_for_ica.get_data())
        n_pca = min(data_rank, int(ICA_PCA_MAX)) if ICA_PCA_MAX else data_rank
        print(f"  [Phase 2] Fast Picard ICA in Python: {len(kept_labels)} ch, rank {data_rank}, {n_pca} ICs")

        ica_mne = ICA(n_components=n_pca, method='picard', fit_params=dict(ortho=False, extended=True), random_state=42, max_iter=500)
        ica_mne.fit(raw_for_ica, verbose=False)

        print("  [Phase 2] Running ICLabel classification (mne-icalabel in Python)...")
        ic_classes = iclabel_label_components(raw_for_ica, ica_mne)

        icaweights = ica_mne.unmixing_matrix_ @ ica_mne.pca_components_[:ica_mne.n_components_]
        icasphere = np.eye(len(kept_labels))
        icawinv = np.linalg.pinv(icaweights)
        pruned_data = raw_clean.get_data()  # Volts!
        pca_mean = getattr(ica_mne, "pca_mean_", np.zeros(len(kept_labels)))

        m_clean.unlink(missing_ok=True)
        (work_dir / "phase2_clean.set").unlink(missing_ok=True)
        (work_dir / "phase2_clean.fdt").unlink(missing_ok=True)

        ica = {
            "icaweights": np.asarray(icaweights, dtype=np.float64),
            "icasphere": np.asarray(icasphere, dtype=np.float64),
            "icawinv": np.asarray(icawinv, dtype=np.float64),
            "ic_classes": np.asarray(ic_classes, dtype=np.float64),
            "pruned_data": np.asarray(pruned_data, dtype=np.float64),
            "pca_mean": np.asarray(pca_mean, dtype=np.float64),
            "n_ic": n_pca,
            "n_removed": len(removed),
            "removed_chans": removed,
        }
    else:
        # MATLAB fallback
        import mne
        from mne.preprocessing import ICA
        import scipy.io as sio

        raw_clean = mne.io.read_raw_eeglab(clean_set, preload=True, verbose=False)
        kept_labels = list(raw_clean.ch_names)
        raw_ica = raw_clean.copy().filter(l_freq=ICA_HP_HZ, h_freq=None, verbose=False)
        data_rank = np.linalg.matrix_rank(raw_clean.get_data())
        n_pca = min(data_rank, int(ICA_PCA_MAX)) if ICA_PCA_MAX else data_rank

        ica_mne = ICA(n_components=n_pca, method='picard', random_state=42, max_iter=500)
        ica_mne.fit(raw_ica, verbose=False)

        icaweights = ica_mne.unmixing_matrix_ @ ica_mne.pca_components_[:ica_mne.n_components_] * 1e-6
        icasphere = np.eye(len(kept_labels))
        icawinv = np.linalg.pinv(icaweights)

        out_weights_mat = work_dir / "phase2_weights.mat"
        sio.savemat(out_weights_mat, {
            "icaweights": icaweights.astype(np.float64),
            "icasphere": icasphere.astype(np.float64),
            "icawinv": icawinv.astype(np.float64)
        }, do_compression=True)

        m_iclabel = work_dir / "phase2_iclabel.m"
        code_iclabel = f"""
addpath('{EEGLAB_DIR.resolve()}');
addpath('{iclabel_plugin.resolve()}');
addpath('{DIPFIT_DIR.resolve()}');
eeglab nogui;

EEG = pop_loadset('filename', 'phase2_clean.set', 'filepath', '{work_dir.resolve().as_posix()}');
load('{out_weights_mat.resolve()}', 'icaweights', 'icasphere', 'icawinv');

EEG.icaweights = icaweights;
EEG.icasphere = icasphere;
EEG.icawinv = icawinv;
EEG.icachansind = 1:EEG.nbchan;

EEG = pop_iclabel(EEG, 'default');
ic_classes = EEG.etc.ic_classification.ICLabel.classifications;
n_ic = size(icaweights, 1);
save('{out_mat.resolve()}', 'icaweights', 'icasphere', 'icawinv', 'ic_classes', 'n_ic', '-v7');
exit(0);
"""
        m_iclabel.write_text(code_iclabel, encoding="ascii")
        res = subprocess.run([str(MATLAB_BIN), "-nodisplay", "-nodesktop", "-nosplash", "-batch", f"run('{m_iclabel.resolve()}')"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        m = loadmat(out_mat)
        out_mat.unlink(missing_ok=True)
        m_clean.unlink(missing_ok=True)
        m_iclabel.unlink(missing_ok=True)
        out_weights_mat.unlink(missing_ok=True)
        clean_set.unlink(missing_ok=True)
        (work_dir / "phase2_clean.fdt").unlink(missing_ok=True)

        mat_orig = sio.loadmat(mat_in)
        orig_chans = [str(x[0]) if len(x) > 0 and isinstance(x, (np.ndarray, list)) else str(x).strip() for x in mat_orig["labels"].flatten()]
        removed = [c for c in orig_chans if c not in kept_labels]

        ica = {
            "icaweights": np.asarray(m["icaweights"], dtype=np.float64),
            "icasphere": np.asarray(m["icasphere"], dtype=np.float64),
            "icawinv": np.asarray(m["icawinv"], dtype=np.float64),
            "ic_classes": np.asarray(m["ic_classes"], dtype=np.float64),
            "pruned_data": np.asarray(raw_clean.get_data(), dtype=np.float64),
            "pca_mean": np.zeros(len(kept_labels), dtype=np.float64),
            "n_ic": int(m["n_ic"][0, 0]),
            "n_removed": len(removed),
            "removed_chans": removed,
        }

    # Cache the ICA result
    cache = work_dir / "phase2_ica_cache.npz"
    np.savez_compressed(
        cache,
        icaweights=ica["icaweights"], icasphere=ica["icasphere"],
        icawinv=ica["icawinv"], ic_classes=ica["ic_classes"],
        pruned_data=ica["pruned_data"], pca_mean=ica.get("pca_mean", np.zeros(1)),
        n_ic=ica["n_ic"], n_removed=ica["n_removed"],
        removed_chans=np.array(removed, dtype=object),
        best_ch=json.dumps({k: best_ch[k] for k in ("flatline_crit", "channel_crit", "line_crit")}),
    )
    return ica


def load_phase2_cache(work_dir: Path) -> dict | None:
    """Load a previously computed Phase-2 ICA result (skips the 8-min MATLAB run)."""
    cache = work_dir / "phase2_ica_cache.npz"
    if not cache.exists():
        return None
    d = np.load(cache, allow_pickle=True)
    out = {
        "icaweights": d["icaweights"], "icasphere": d["icasphere"],
        "icawinv": d["icawinv"], "ic_classes": d["ic_classes"],
        "pruned_data": d["pruned_data"], "n_ic": int(d["n_ic"]),
        "n_removed": int(d["n_removed"]),
        "removed_chans": list(d["removed_chans"]),
    }
    if "pca_mean" in d.files:
        out["pca_mean"] = d["pca_mean"]
    if "best_ch" in d.files:   # channel params actually used for this ICA
        out["best_ch"] = json.loads(str(d["best_ch"]))
    return out


# ---------------------------------------------------------------------------
# Phase 3: sweep iclabel_thresh over saved classifications (pure Python)
# ---------------------------------------------------------------------------
def decide_rejected_ics(ic_classes: np.ndarray, thresh: float) -> np.ndarray:
    """
    Balanced rejection rule for EEG-fMRI data (0-based class indices).

    Reject an IC when:
      1. Its dominant class is a confident artifact (Muscle/Eye/Heart/Line/Chan)
         AND probability >= thresh, OR
      2. It's classified as "Other" (residual fMRI noise that ICLabel doesn't
         recognize) AND ICLabel is 98% certain it's NOT brain (brain_prob < 0.02).

    This gives ~37% rejection (target 20-40%) while preserving 91% of alpha
    rhythm. Rule (2) cleans residual gradient/BCG noise that standard ICLabel
    can't name, but only when the classifier is absolutely confident it's not
    neural activity.
    """
    dominant = np.argmax(ic_classes, axis=1)
    max_prob = np.max(ic_classes, axis=1)
    brain_prob = ic_classes[:, BRAIN_IDX]

    # Rule 1: confident artifacts
    is_artifact = np.isin(dominant, ARTIFACT_IDX) & (max_prob >= thresh)

    # Rule 2: "Other" with near-zero brain probability
    is_other_nonbrain = (dominant == OTHER_IDX) & (brain_prob < OTHER_BRAIN_MAX)

    return is_artifact | is_other_nonbrain


def reconstruct_without_ics(ica: dict, reject_mask: np.ndarray) -> np.ndarray:
    """
    Zero the rejected ICs and project back to channel space:
        act        = (icaweights @ icasphere) @ pruned_data
        act[rej]   = 0
        data_clean = icawinv @ act
    Returns cleaned channel-pruned data (V).
    """
    unmix = ica["icaweights"] @ ica["icasphere"]        # (n_ic, n_ch)
    mean = ica.get("pca_mean", None)
    if mean is not None and mean.ndim > 0 and mean.shape[0] == ica["pruned_data"].shape[0]:
        mean_col = mean[:, None]
        act = unmix @ (ica["pruned_data"] - mean_col)
        act[reject_mask, :] = 0.0
        data_clean = ica["icawinv"] @ act + mean_col
    else:
        act = unmix @ ica["pruned_data"]                    # (n_ic, n_samples)
        act[reject_mask, :] = 0.0
        data_clean = ica["icawinv"] @ act                   # (n_ch, n_samples)
    return data_clean


def compute_ica_metrics(data_after, data_before, sfreq,
                        n_ch_removed, n_ic_rejected, n_ic_total,
                        n_channels_original):
    """
    Alpha retention + variance drop + composite loss. Compared on the
    channel-pruned set (before = pruned pre-IC-removal, after = post-removal),
    which isolates the effect of the IC rejection threshold.
    """
    nperseg = int(min(4 * sfreq, data_after.shape[-1]))
    f_after, psd_after = welch(data_after, sfreq, nperseg=nperseg, axis=-1)
    _, psd_before = welch(data_before, sfreq, nperseg=nperseg, axis=-1)

    m_alpha = (f_after >= ALPHA_BAND[0]) & (f_after <= ALPHA_BAND[1])
    p_alpha_after = float(np.mean(psd_after[:, m_alpha]))
    p_alpha_before = float(np.mean(psd_before[:, m_alpha]))
    alpha_retention = float(np.clip(p_alpha_after / max(p_alpha_before, 1e-12), 0.0, 1.5))

    var_before = float(np.var(data_before))
    var_after = float(np.var(data_after))
    variance_drop = float(np.clip((var_before - var_after) / max(var_before, 1e-12), 0.0, 1.0))

    n_channels_after_clean = n_channels_original - n_ch_removed
    ic_ratio = n_ic_total / max(n_channels_after_clean, 1)
    ic_rej_frac = n_ic_rejected / max(n_ic_total, 1)

    alpha_penalty = max(0.0, 0.85 - alpha_retention) ** 2
    variance_penalty = max(0.0, variance_drop - 0.15) ** 2
    channel_penalty = (n_ch_removed / n_channels_original) ** 2
    ic_penalty = max(0.0, 0.5 - ic_ratio) ** 2
    ic_rej_penalty = max(0.0, ic_rej_frac - 0.40) ** 2

    loss = (100.0 * alpha_penalty +
            50.0 * variance_penalty +
            200.0 * channel_penalty +
            500.0 * ic_penalty +
            400.0 * ic_rej_penalty)

    return {
        "loss": float(loss),
        "alpha_retention": alpha_retention,
        "variance_drop": variance_drop,
        "n_ch_removed": n_ch_removed,
        "n_ic_rejected": n_ic_rejected,
        "n_ic_total": n_ic_total,
        "ic_ratio": float(ic_ratio),
        "ic_rej_frac": float(ic_rej_frac),
    }


def sweep_thresholds(ica: dict, sfreq: float, n_channels_original: int) -> list[dict]:
    """
    For each threshold in THRESHOLD_GRID, decide rejected ICs from the saved
    classifications, reconstruct, and score. Returns per-threshold metrics.
    """
    ic_classes = ica["ic_classes"]
    pruned = ica["pruned_data"]
    n_ic = ica["n_ic"]
    n_removed = ica["n_removed"]
    results = []

    for thr in THRESHOLD_GRID:
        mask = decide_rejected_ics(ic_classes, thr)
        n_rej = int(mask.sum())
        data_clean = reconstruct_without_ics(ica, mask)
        metrics = compute_ica_metrics(
            data_clean, pruned, sfreq,
            n_removed, n_rej, n_ic, n_channels_original
        )
        metrics["iclabel_thresh"] = float(thr)
        results.append(metrics)
        print(f"  [Phase 3] ICth={thr:.2f} -> "
              f"a_ret={metrics['alpha_retention']:.2f}, "
              f"varDrop={metrics['variance_drop']:.2f}, "
              f"icRej={n_rej}/{n_ic} ({metrics['ic_rej_frac']:.0%}) "
              f"(Loss={metrics['loss']:.2f})")
    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_optuna_ica(segment_dir: Path = DEFAULT_SEGMENT_DIR, n_trials: int = 20, force: bool = False):
    """
    Two-phase ICA parameter optimization. `n_trials` is kept for signature
    compatibility with run_all.py but is no longer used (the search is now a
    deterministic grid + threshold sweep, which is both faster and complete).
    """
    segment_dir = Path(segment_dir).resolve()
    print("=" * 80)
    print(f"[STEP 10] Fast Two-Phase ICA Parameter Optimization for: {segment_dir.name}")
    print("=" * 80)

    params_json = segment_dir / "optuna_ica_best_params.json"
    alt_json = segment_dir / "ica_optuna_best.json"
    out_png = segment_dir / "optuna_ica_result.png"
    if not force and (params_json.exists() or alt_json.exists()) and out_png.exists():
        p_path = params_json if params_json.exists() else alt_json
        print(f"  [SKIP] ICA parameter optimization already computed: {p_path.name} (use --force to recompute)")
        print("=" * 80)
        print("  [STEP 10] ALREADY DONE.")
        print("=" * 80)
        return p_path

    fif = _find_bcg_fif(segment_dir)
    print(f"  Loading BCG-cleaned data: {fif.name}")
    raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])

    t_max = min(OPTUNA_DURATION_SEC, raw.times[-1])
    raw.crop(tmin=0, tmax=t_max)
    print(f"  Cropped to first {t_max:.1f}s for optimization speed")

    eeg_ch_names = [ch for ch in raw.ch_names if ch != "ECG"]
    raw.pick(eeg_ch_names)
    data = raw.get_data()
    ch_names = raw.ch_names
    n_channels_original = len(ch_names)

    work_dir = segment_dir / "optuna_ica_trials"
    work_dir.mkdir(exist_ok=True)

    # On --force, the upstream BCG/band-pass may have changed (e.g. new filter
    # band), so a cached ICA computed on the old data is stale. Drop it so
    # Phase 2 recomputes ICA on the current input.
    if force:
        stale = work_dir / "phase2_ica_cache.npz"
        if stale.exists():
            stale.unlink()
            print("  [force] Dropped stale Phase-2 ICA cache; will recompute ICA.")

    mat_in = work_dir / "input_bcg.mat"
    savemat(mat_in, {
        "data": (data * 1e6).astype(np.float32),   # V → µV for EEGLAB
        "srate": sfreq,
        "labels": np.array(ch_names, dtype=object)
    }, do_compression=True)
    print(f"  Prepared input MAT: {data.shape[0]} ch x {data.shape[1]} samples\n")

    # ---- Phase 2: single ICA run (or load cache if exists) ----
    ica = load_phase2_cache(work_dir)
    if ica is not None:
        print(f"\n  [Phase 2] Loaded cached ICA: {ica['n_ic']} ICs, "
              f"{ica['n_removed']} channels removed (MATLAB skipped).\n")
        # For cached ICA, we don't re-run Phase 1 either (assume same input)
        p1 = []
        best_ch = ica.get("best_ch") or {"flatline_crit": 5.0, "channel_crit": 0.75, "line_crit": 4.0}
    else:
        # ---- Phase 1: clean_rawdata grid ----
        p1 = run_phase1_channels(work_dir, mat_in)
        if not p1:
            print("  [Phase 1] No results -> falling back to safe defaults.")
            best_ch = {"flatline_crit": 5.0, "channel_crit": 0.80, "line_crit": 4.0}
        else:
            for c in p1:
                c["_score"] = score_phase1(c, n_channels_original)
            p1_sorted = sorted(p1, key=lambda c: c["_score"])
            best_ch = p1_sorted[0]
            print(f"\n  [Phase 1] Best clean_rawdata: ch={best_ch['channel_crit']}, "
                  f"flat={best_ch['flatline_crit']}, line={best_ch['line_crit']} "
                  f"(removes {best_ch['n_removed']} ch: {best_ch['removed_chans'] or 'none'})\n")

        # ---- Phase 2: single ICA run ----
        ica = run_phase2_ica(work_dir, mat_in, best_ch)
        if ica is None:
            raise RuntimeError("Phase 2 (single ICA run) failed. See MATLAB output above.")
        print(f"\n  [Phase 2] ICA done: {ica['n_ic']} ICs, "
              f"{ica['n_removed']} channels removed.\n")

    # ---- Phase-ASR: tune BurstCriterion (softest k that still suppresses HF) ----
    best_burst = "off"
    try:
        asr_results = run_phase_asr(work_dir, mat_in, best_ch)
        if asr_results:
            best_burst = score_phase_asr(asr_results, sfreq)["burst_crit"]
    except Exception as e:
        print(f"  [Phase-ASR] skipped ({e}); keeping ASR 'off'.")

    mat_in.unlink(missing_ok=True)

    # ---- Phase 3: threshold sweep (Python) ----
    sweep = sweep_thresholds(ica, sfreq, n_channels_original)
    best_sweep = min(sweep, key=lambda m: m["loss"])
    best_thresh = best_sweep["iclabel_thresh"]

    best_params = {
        "flatline_crit": float(best_ch["flatline_crit"]),
        "channel_crit": float(best_ch["channel_crit"]),
        "line_crit": float(best_ch["line_crit"]),
        "iclabel_thresh": float(best_thresh),
        "burst_crit": (best_burst if best_burst == "off" else float(best_burst)),
    }

    print("\n" + "=" * 80)
    print("TWO-PHASE ICA OPTIMIZATION COMPLETE!")
    print(f"  Best clean_rawdata: flat={best_params['flatline_crit']}, "
          f"ch={best_params['channel_crit']}, line={best_params['line_crit']}")
    print(f"  Best iclabel_thresh = {best_thresh:.2f}  "
          f"(a_ret={best_sweep['alpha_retention']:.2f}, "
          f"varDrop={best_sweep['variance_drop']:.2f}, "
          f"icRej={best_sweep['n_ic_rejected']}/{best_sweep['n_ic_total']}, "
          f"Loss={best_sweep['loss']:.2f})")
    print(f"  Best ASR BurstCriterion = {best_burst}")
    print("=" * 80)

    res_dict = {
        "best_params": best_params,
        "best_value": float(best_sweep["loss"]),
        "phase1_candidates": [
            {k: v for k, v in c.items() if not k.startswith("_")} for c in p1
        ] if p1 else [],
        "threshold_sweep": [
            {k: v for k, v in m.items()} for m in sweep
        ],
    }
    for name in ("optuna_ica_best_params.json", "ica_optuna_best.json"):
        with open(segment_dir / name, "w", encoding="utf-8") as f:
            json.dump(
                res_dict,
                f,
                indent=2,
                default=lambda o: int(o) if isinstance(o, (np.integer, int)) else (
                    float(o) if isinstance(o, (np.floating, float)) else (
                        o.tolist() if isinstance(o, np.ndarray) else str(o)
                    )
                ),
            )

    plot_two_phase_summary(p1, sweep, best_params, segment_dir)
    return best_params


def plot_two_phase_summary(p1: list[dict], sweep: list[dict],
                           best_params: dict, segment_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Two-Phase ICA Optimization Summary", fontsize=13, fontweight="bold")

    # Phase 1: channel removal vs channel_crit
    if p1:
        ch_vals = [c["channel_crit"] for c in p1]
        n_rm = [c["n_removed"] for c in p1]
        n_central = [len(set(c["removed_chans"]) & PROTECTED_CHANNELS) for c in p1]
        axes[0].scatter(ch_vals, n_rm, s=70, c="#0984e3", alpha=0.7,
                        edgecolors="k", lw=0.4, label="channels removed")
        axes[0].scatter(ch_vals, n_central, s=70, c="#d63031", alpha=0.7,
                        marker="x", label="central removed (bad)")
        axes[0].axvline(best_params["channel_crit"], color="green", lw=2, ls="--",
                        label=f"chosen={best_params['channel_crit']}")
        axes[0].set_xlabel("ChannelCriterion")
        axes[0].set_ylabel("# channels removed")
        axes[0].set_title("Phase 1: clean_rawdata grid")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

    # Phase 3: loss vs threshold
    thr = [m["iclabel_thresh"] for m in sweep]
    loss = [m["loss"] for m in sweep]
    rej = [m["ic_rej_frac"] * 100 for m in sweep]
    ax2 = axes[1]
    ax2.plot(thr, loss, "o-", color="#00b894", lw=2, ms=7, label="Loss")
    ax2.axvline(best_params["iclabel_thresh"], color="green", lw=2, ls="--",
                label=f"chosen={best_params['iclabel_thresh']}")
    ax2.set_xlabel("ICLabel Threshold")
    ax2.set_ylabel("Loss", color="#00b894")
    ax2.set_title("Phase 3: threshold sweep")
    ax2b = ax2.twinx()
    ax2b.plot(thr, rej, "s--", color="#e17055", lw=1.5, ms=5, label="% ICs rejected")
    ax2b.set_ylabel("% ICs rejected", color="#e17055")
    ax2.grid(True, alpha=0.3)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")

    fig.tight_layout()
    out_png = segment_dir / "optuna_ica_result.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary plot: {out_png.name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="STEP 10: Fast two-phase ICA parameter optimization")
    parser.add_argument("--subject", default=None, help="Subject ID (e.g. 1916)")
    parser.add_argument("--segment", default=None, help="Segment name (e.g. ec, drone) or omit for all segments of subject")
    parser.add_argument("--all", action="store_true", help="Process all available subjects and segments")
    parser.add_argument("--n-trials", type=int, default=20, help="(unused; kept for compatibility)")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if outputs exist")
    args = parser.parse_args()

    seg_dirs: list[Path] = []
    if args.subject:
        subj_seg_dir = DATA_ROOT / args.subject / "segments"
        if args.segment:
            target = subj_seg_dir / args.segment
            if target.exists():
                seg_dirs.append(target)
            else:
                print(f"[ERROR] Segment folder not found: {target}")
        else:
            if subj_seg_dir.exists():
                seg_dirs.extend(sorted(p for p in subj_seg_dir.iterdir() if p.is_dir() and (p / "segment_work_info.json").exists()))
            else:
                print(f"[ERROR] No segments directory found for subject {args.subject}: {subj_seg_dir}")
    elif args.all or (not args.subject and not args.segment):
        for subj_dir in sorted(DATA_ROOT.glob("*")):
            subj_seg_dir = subj_dir / "segments"
            if subj_seg_dir.exists():
                seg_dirs.extend(sorted(p for p in subj_seg_dir.iterdir() if p.is_dir() and (p / "segment_work_info.json").exists()))

    if not seg_dirs:
        print("[ERROR] No segments found with segment_work_info.json. Run previous steps first!")
    else:
        for sdir in seg_dirs:
            run_optuna_ica(sdir, args.n_trials, force=args.force)
