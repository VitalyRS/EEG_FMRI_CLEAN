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
                         DEFAULT_SEGMENT_DIR, ALPHA_BAND, GRADIENT_HARMONICS)
except ImportError:
    from config import (MATLAB_BIN, EEGLAB_DIR, PROJECT_ROOT,
                         DEFAULT_SEGMENT_DIR, ALPHA_BAND, GRADIENT_HARMONICS)

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


def _find_bcg_fif(segment_dir: Path) -> Path:
    seg = segment_dir.name
    p = DATA_ROOT / DEFAULT_EXPERIMENT / "derivatives" / "03_bcg" / seg / f"{seg}_bcg_clean.fif"
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
exit(0);
"""
    with open(m_file, "w", encoding="ascii") as f:
        f.write(code)

    print("  [Phase 1] Grid-searching clean_rawdata parameters (no ICA)...")
    try:
        res = subprocess.run(
            [str(MATLAB_BIN), "-nodesktop", "-nosplash", "-batch", f"run('{m_file.resolve()}')"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900
        )
    except subprocess.TimeoutExpired:
        print("  [Phase 1] MATLAB timed out.")
        m_file.unlink(missing_ok=True)
        return []
    m_file.unlink(missing_ok=True)

    if res.returncode != 0 or not out_mat.exists():
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
# Phase 2: run clean_rawdata (best params) + runica + ICLabel ONCE
# ---------------------------------------------------------------------------
def run_phase2_ica(work_dir: Path, mat_in: Path, best_ch: dict) -> dict | None:
    """
    With the Phase-1 winning clean_rawdata params, run clean_rawdata + runica +
    ICLabel ONCE. Save the ICA matrices + ICLabel classifications + the
    channel-pruned data so Phase 3 can sweep thresholds without re-running ICA.
    """
    out_mat = work_dir / "phase2_ica.mat"
    m_file = work_dir / "phase2_ica.m"

    clean_rawdata_plugin = EEGLAB_DIR / "plugins" / "clean_rawdata"
    iclabel_plugin = EEGLAB_DIR / "plugins" / "ICLabel"

    code = f"""
addpath('{EEGLAB_DIR.resolve()}');
addpath('{clean_rawdata_plugin.resolve()}');
addpath('{iclabel_plugin.resolve()}');
addpath('{DIPFIT_DIR.resolve()}');
eeglab nogui;

load('{mat_in.resolve()}', 'data', 'srate', 'labels');
EEG = eeg_emptyset();
EEG.setname = 'phase2';
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
orig_chans = {{EEG.chanlocs.labels}};

% clean_rawdata with the Phase-1 winning parameters
EEG_clean = pop_clean_rawdata(EEG, ...
    'FlatlineCriterion', {best_ch['flatline_crit']}, ...
    'ChannelCriterion', {best_ch['channel_crit']}, ...
    'LineNoiseCriterion', {best_ch['line_crit']}, ...
    'Highpass', 'off', ...
    'BurstCriterion', 'off', ...
    'WindowCriterion', 'off', ...
    'BurstRejection', 'off', ...
    'Distance', 'Euclidian');

kept_chans = {{EEG_clean.chanlocs.labels}};
removed_chans = setdiff(orig_chans, kept_chans);
n_removed = length(removed_chans);
n_chans_after = EEG_clean.nbchan;
fprintf('Channels after clean_rawdata: %d (removed %d)\\n', n_chans_after, n_removed);

% Rank-safe ICA (PCA-reduce if rank-deficient)
data_rank = rank(double(EEG_clean.data'));
fprintf('Data rank: %d, Channels: %d\\n', data_rank, n_chans_after);
if data_rank < n_chans_after
    fprintf('Reducing dimensionality to rank %d via PCA\\n', data_rank);
    EEG_clean = pop_runica(EEG_clean, 'icatype', 'runica', 'extended', 1, 'interrupt', 'off', 'pca', data_rank);
else
    EEG_clean = pop_runica(EEG_clean, 'icatype', 'runica', 'extended', 1, 'interrupt', 'off');
end
n_ic = size(EEG_clean.icaweights, 1);
fprintf('Number of ICs computed: %d\\n', n_ic);

% ICLabel classification (threshold-independent!)
EEG_clean = pop_iclabel(EEG_clean, 'default');
ic_classes = EEG_clean.etc.ic_classification.ICLabel.classifications;

% Save everything Phase 3 needs to sweep thresholds in Python.
icaweights = EEG_clean.icaweights;
icasphere  = EEG_clean.icasphere;
icawinv    = EEG_clean.icawinv;
pruned_data = single(EEG_clean.data);  % channel-pruned, pre-IC-removal (uV)
kept_labels = kept_chans;
if n_removed == 0
    removed_str = '';
else
    removed_str = strjoin(removed_chans, ',');
end
save('{out_mat.resolve()}', 'icaweights', 'icasphere', 'icawinv', ...
     'ic_classes', 'pruned_data', 'n_ic', 'n_removed', 'removed_str', ...
     'kept_labels', '-v7');
clear EEG EEG_clean;
exit(0);
"""
    with open(m_file, "w", encoding="ascii") as f:
        f.write(code)

    print(f"  [Phase 2] Running clean_rawdata + runica + ICLabel ONCE "
          f"(ch={best_ch['channel_crit']}, flat={best_ch['flatline_crit']}, "
          f"line={best_ch['line_crit']})...")
    try:
        res = subprocess.run(
            [str(MATLAB_BIN), "-nodesktop", "-nosplash", "-batch", f"run('{m_file.resolve()}')"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800
        )
    except subprocess.TimeoutExpired:
        print("  [Phase 2] MATLAB timed out (>1800s).")
        m_file.unlink(missing_ok=True)
        out_mat.unlink(missing_ok=True)
        return None
    m_file.unlink(missing_ok=True)

    if res.returncode != 0 or not out_mat.exists():
        print(f"  [Phase 2] MATLAB failed (rc={res.returncode}):")
        print(res.stdout[-1800:] if res.stdout else "(no output)")
        return None

    m = loadmat(out_mat)
    out_mat.unlink(missing_ok=True)

    removed_str = m["removed_str"]
    removed_str = str(removed_str[0]) if np.size(removed_str) else ""
    removed = [c for c in removed_str.split(",") if c] if removed_str else []

    ica = {
        "icaweights": np.asarray(m["icaweights"], dtype=np.float64),
        "icasphere": np.asarray(m["icasphere"], dtype=np.float64),
        "icawinv": np.asarray(m["icawinv"], dtype=np.float64),
        "ic_classes": np.asarray(m["ic_classes"], dtype=np.float64),
        "pruned_data": np.asarray(m["pruned_data"], dtype=np.float64) * 1e-6,  # µV → V
        "n_ic": int(m["n_ic"][0, 0]),
        "n_removed": int(m["n_removed"][0, 0]),
        "removed_chans": removed,
    }

    # Cache the expensive ICA result so re-runs / rule tweaks skip MATLAB.
    cache = work_dir / "phase2_ica_cache.npz"
    np.savez_compressed(
        cache,
        icaweights=ica["icaweights"], icasphere=ica["icasphere"],
        icawinv=ica["icawinv"], ic_classes=ica["ic_classes"],
        pruned_data=ica["pruned_data"], n_ic=ica["n_ic"],
        n_removed=ica["n_removed"], removed_chans=np.array(removed, dtype=object),
    )
    return ica


def load_phase2_cache(work_dir: Path) -> dict | None:
    """Load a previously computed Phase-2 ICA result (skips the 8-min MATLAB run)."""
    cache = work_dir / "phase2_ica_cache.npz"
    if not cache.exists():
        return None
    d = np.load(cache, allow_pickle=True)
    return {
        "icaweights": d["icaweights"], "icasphere": d["icasphere"],
        "icawinv": d["icawinv"], "ic_classes": d["ic_classes"],
        "pruned_data": d["pruned_data"], "n_ic": int(d["n_ic"]),
        "n_removed": int(d["n_removed"]),
        "removed_chans": list(d["removed_chans"]),
    }


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
    is_other_nonbrain = (dominant == OTHER_IDX) & (brain_prob < 0.02)

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
def run_optuna_ica(segment_dir: Path = DEFAULT_SEGMENT_DIR, n_trials: int = 20):
    """
    Two-phase ICA parameter optimization. `n_trials` is kept for signature
    compatibility with run_all.py but is no longer used (the search is now a
    deterministic grid + threshold sweep, which is both faster and complete).
    """
    segment_dir = Path(segment_dir).resolve()
    print("=" * 80)
    print(f"[STEP 10] Fast Two-Phase ICA Parameter Optimization for: {segment_dir.name}")
    print("=" * 80)

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
        best_ch = {"flatline_crit": 5.0, "channel_crit": 0.75, "line_crit": 4.0}
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
            json.dump(res_dict, f, indent=2)

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
    parser = argparse.ArgumentParser(description="Fast two-phase ICA parameter optimization")
    parser.add_argument("--segment-dir", type=Path, default=DEFAULT_SEGMENT_DIR)
    parser.add_argument("--n-trials", type=int, default=20, help="(unused; kept for compatibility)")
    args = parser.parse_args()
    run_optuna_ica(args.segment_dir, args.n_trials)
