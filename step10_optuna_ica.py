"""
STEP 10: Bayesian Optimization of ICA Pipeline Parameters via Optuna TPE
==========================================================================
Tunes clean_rawdata bad-channel detection + ICLabel rejection threshold.
Uses first 120s of BCG-cleaned data for speed (ICA is expensive).
"""
from pathlib import Path
import subprocess
import gc
import json
import numpy as np
import mne
from scipy.io import savemat, loadmat
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

try:
    from .config import (MATLAB_BIN, EEGLAB_DIR, PROJECT_ROOT,
                         DEFAULT_SEGMENT_DIR, ALPHA_BAND, GRADIENT_HARMONICS)
except ImportError:
    from config import (MATLAB_BIN, EEGLAB_DIR, PROJECT_ROOT,
                         DEFAULT_SEGMENT_DIR, ALPHA_BAND, GRADIENT_HARMONICS)

DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_EXPERIMENT = "1916"

# For optimization speed: use only first N seconds (runica is expensive)
OPTUNA_DURATION_SEC = 180.0

# Maximum fraction of channels that can be removed before pruning
MAX_CHANNEL_REMOVAL_FRACTION = 0.20  # 20%

DIPFIT_DIR = EEGLAB_DIR / "plugins" / "dipfit"
STD_1005 = DIPFIT_DIR / "standard_BEM" / "elec" / "standard_1005.elc"


def _find_bcg_fif(segment_dir: Path) -> Path:
    seg = segment_dir.name
    p = DATA_ROOT / DEFAULT_EXPERIMENT / "derivatives" / "03_bcg" / seg / f"{seg}_bcg_clean.fif"
    if not p.exists():
        raise FileNotFoundError(f"BCG input not found: {p}. Run step08 first.")
    return p


def compute_ica_metrics(data_after: np.ndarray, data_before: np.ndarray,
                        sfreq: float, n_ch_removed: int, n_ic_rejected: int,
                        n_ic_total: int, n_channels_original: int):
    """
    Compute alpha retention, variance drop, and composite loss.
    - alpha_retention: how much alpha power (8-13 Hz) survived ICA (want high)
    - variance_drop: fraction of total variance removed (want low, but not zero)
    - n_ch_removed: bad channels removed (want low)
    - n_ic_rejected: ICs rejected (informational)
    - n_ic_total: total ICs found by ICA (want close to n_channels_after_clean)
    """
    nperseg = int(min(4 * sfreq, data_after.shape[-1]))
    f_after, psd_after = welch(data_after, sfreq, nperseg=nperseg, axis=-1)
    f_before, psd_before = welch(data_before, sfreq, nperseg=nperseg, axis=-1)

    m_alpha = (f_after >= ALPHA_BAND[0]) & (f_after <= ALPHA_BAND[1])

    # Alpha retention: mean alpha power after / before across all channels
    p_alpha_after = float(np.mean(psd_after[:, m_alpha]))
    p_alpha_before = float(np.mean(psd_before[:, m_alpha]))
    alpha_retention = p_alpha_after / max(p_alpha_before, 1e-12)
    alpha_retention = float(np.clip(alpha_retention, 0.0, 1.5))

    # Variance drop: (var_before - var_after) / var_before
    var_before = float(np.var(data_before))
    var_after = float(np.var(data_after))
    variance_drop = (var_before - var_after) / max(var_before, 1e-12)
    variance_drop = float(np.clip(variance_drop, 0.0, 1.0))

    # Expected IC count: after removing bad channels, ICA should find ~that many ICs
    n_channels_after_clean = n_channels_original - n_ch_removed
    expected_ics = max(n_channels_after_clean, 1)
    ic_ratio = n_ic_total / expected_ics  # want close to 1.0

    # Composite loss to MINIMIZE:
    # - Penalize low alpha retention (want >= 0.85)
    # - Penalize excessive variance drop (> 15%)
    # - Penalize removing channels
    # - Penalize low IC count (degenerate ICA solutions)
    alpha_penalty = max(0.0, 0.85 - alpha_retention) ** 2
    variance_penalty = max(0.0, variance_drop - 0.15) ** 2
    channel_penalty = (n_ch_removed / n_channels_original) ** 2

    # IC count penalty: strongly penalize if ICA found far fewer ICs than expected
    # ic_ratio should be ~1.0; if < 0.5 it means data rank collapsed
    ic_penalty = max(0.0, 0.5 - ic_ratio) ** 2

    loss = (100.0 * alpha_penalty +
            50.0 * variance_penalty +
            200.0 * channel_penalty +
            500.0 * ic_penalty)  # Heavy penalty for degenerate ICA

    return {
        "loss": float(loss),
        "alpha_retention": alpha_retention,
        "variance_drop": variance_drop,
        "n_ch_removed": n_ch_removed,
        "n_ic_rejected": n_ic_rejected,
        "n_ic_total": n_ic_total,
        "ic_ratio": float(ic_ratio),
    }


def run_ica_trial_matlab(work_dir: Path, mat_in: Path, trial_id: int,
                         flatline_crit, channel_crit, line_crit,
                         iclabel_thresh: float, n_channels_original: int) -> dict | None:
    """
    Run one ICA trial: clean_rawdata -> runica -> ICLabel -> reject.
    Returns dict with metrics or None on failure.
    """
    tag = f"trial_{trial_id:03d}"
    out_mat = work_dir / f"{tag}_result.mat"
    m_file = work_dir / f"{tag}.m"

    helper_dir = Path(__file__).parent.resolve()
    clean_rawdata_plugin = EEGLAB_DIR / "plugins" / "clean_rawdata"
    iclabel_plugin = EEGLAB_DIR / "plugins" / "ICLabel"

    code = f"""
addpath('{EEGLAB_DIR.resolve()}');
addpath('{clean_rawdata_plugin.resolve()}');
addpath('{iclabel_plugin.resolve()}');
addpath('{DIPFIT_DIR.resolve()}');
addpath('{helper_dir}');
eeglab nogui;

load('{mat_in.resolve()}', 'data', 'srate', 'labels');
EEG = eeg_emptyset();
EEG.setname = '{tag}';
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

% Load standard_1005 channel locations
EEG = pop_chanedit(EEG, 'lookup', '{STD_1005.resolve()}');
EEG = eeg_checkset(EEG);

chanlocs_full = EEG.chanlocs;
orig_chans = {{EEG.chanlocs.labels}};
n_orig = EEG.nbchan;

% clean_rawdata artifact detection
EEG_clean = pop_clean_rawdata(EEG, ...
    'FlatlineCriterion', {flatline_crit}, ...
    'ChannelCriterion', {channel_crit}, ...
    'LineNoiseCriterion', {line_crit}, ...
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

% Check data rank before ICA
data_rank = rank(double(EEG_clean.data'));
fprintf('Data rank: %d, Channels: %d\\n', data_rank, n_chans_after);

% Use PCA reduction if rank < number of channels (prevents degenerate ICA)
if data_rank < n_chans_after
    fprintf('Reducing dimensionality to rank %d via PCA\\n', data_rank);
    EEG_clean = pop_runica(EEG_clean, 'icatype', 'runica', 'extended', 1, 'interrupt', 'off', 'pca', data_rank);
else
    EEG_clean = pop_runica(EEG_clean, 'icatype', 'runica', 'extended', 1, 'interrupt', 'off');
end

n_ic = size(EEG_clean.icaweights, 1);
fprintf('Number of ICs computed: %d\\n', n_ic);

% ICLabel classification
EEG_clean = pop_iclabel(EEG_clean, 'default');
ic_classes = EEG_clean.etc.ic_classification.ICLabel.classifications;

% Auto-reject artifacts:
% Reject if dominant class is artifact (Muscle=2, Eye=3, Heart=4, Line=5, Chan=6)
% AND probability >= threshold
% Also reject if Brain probability < 0.20 (almost certainly not brain)
% Also reject "Other" (col 7) if probability >= 0.90
artifact_cols = [2, 3, 4, 5, 6];
rej_mask = false(1, n_ic);
for c = 1:n_ic
    probs = ic_classes(c, :);
    [max_prob, dominant] = max(probs);
    brain_prob = probs(1);

    % Reject if dominant is artifact type and above threshold
    if any(dominant == artifact_cols) && max_prob >= {iclabel_thresh}
        rej_mask(c) = true;
    end

    % Reject if Brain probability is very low (regardless of dominant class)
    if brain_prob < 0.20
        rej_mask(c) = true;
    end

    % Reject "Other" (column 7) if very high probability
    if dominant == 7 && max_prob >= 0.90
        rej_mask(c) = true;
    end
end
reject_ic_list = find(rej_mask);
n_ic_rejected = length(reject_ic_list);
fprintf('ICs rejected: %d / %d\\n', n_ic_rejected, n_ic);

if n_ic_rejected > 0
    % Manual IC removal (compatible with PCA-reduced ICA, unlike pop_subcomp)
    % Compute IC activations
    ica_act = (EEG_clean.icaweights * EEG_clean.icasphere) * double(EEG_clean.data);
    % Zero out rejected components
    ica_act(reject_ic_list, :) = 0;
    % Project back to channel space
    EEG_clean.data = EEG_clean.icawinv * ica_act;
    fprintf('Removed %d artifact ICs via manual projection\\n', n_ic_rejected);
end

% Interpolate removed channels back to original montage
if n_removed > 0
    EEG_clean = pop_interp(EEG_clean, chanlocs_full, 'spherical');
end

% Average reference
EEG_clean = pop_reref(EEG_clean, []);

% Save result
after_data = single(EEG_clean.data);
save('{out_mat.resolve()}', 'after_data', 'removed_chans', 'n_removed', 'n_ic_rejected', 'n_ic', '-v7');
clear EEG EEG_clean;
exit(0);
"""

    with open(m_file, "w", encoding="ascii") as f:
        f.write(code)

    res = subprocess.run(
        [str(MATLAB_BIN), "-nodesktop", "-nosplash", "-batch", f"run('{m_file.resolve()}')"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600
    )
    m_file.unlink(missing_ok=True)

    if res.returncode != 0 or not out_mat.exists():
        tail = res.stdout[-1500:] if res.stdout else "(no output)"
        print(f"  [WARN] Trial {trial_id} MATLAB failed (rc={res.returncode}, out_mat exists={out_mat.exists()}):")
        print(tail)
        return None

    try:
        m_res = loadmat(out_mat)
        after_data = m_res["after_data"] * 1e-6  # µV → V (back to MNE units)
        n_removed = int(m_res["n_removed"][0, 0])
        n_ic_rejected = int(m_res["n_ic_rejected"][0, 0])
        n_ic = int(m_res["n_ic"][0, 0])
        out_mat.unlink(missing_ok=True)
        return {
            "after_data": after_data,
            "n_removed": n_removed,
            "n_ic_rejected": n_ic_rejected,
            "n_ic": n_ic,
        }
    except Exception as e:
        print(f"  [WARN] Trial {trial_id} MAT load failed: {e}")
        out_mat.unlink(missing_ok=True)
        return None


def run_optuna_ica(segment_dir: Path = DEFAULT_SEGMENT_DIR, n_trials: int = 20):
    segment_dir = Path(segment_dir).resolve()
    print("=" * 80)
    print(f"[STEP 10] Optuna ICA Parameter Optimization for: {segment_dir.name}")
    print("=" * 80)

    fif = _find_bcg_fif(segment_dir)
    print(f"  Loading BCG-cleaned data: {fif.name}")
    raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])

    # Crop to first OPTUNA_DURATION_SEC for speed
    t_max = min(OPTUNA_DURATION_SEC, raw.times[-1])
    raw.crop(tmin=0, tmax=t_max)
    print(f"  Cropped to first {t_max:.1f}s for optimization speed")

    # Drop ECG channel if present
    eeg_ch_names = [ch for ch in raw.ch_names if ch != "ECG"]
    raw.pick_channels(eeg_ch_names)
    data = raw.get_data()  # (n_ch, n_samples)
    ch_names = raw.ch_names
    n_channels_original = len(ch_names)

    work_dir = segment_dir / "optuna_ica_trials"
    work_dir.mkdir(exist_ok=True)

    # Save input MAT — convert V → µV for EEGLAB (MNE stores in V, EEGLAB expects µV)
    mat_in = work_dir / "input_bcg.mat"
    data_uv = data * 1e6  # Volts → microvolts
    savemat(mat_in, {
        "data": data_uv.astype(np.float32),
        "srate": sfreq,
        "labels": np.array(ch_names, dtype=object)
    }, do_compression=True)
    print(f"  Prepared input MAT: {data.shape[0]} ch x {data.shape[1]} samples (converted to µV)")

    db_path = segment_dir / "optuna_ica_study.db"
    storage_url = f"sqlite:///{db_path.resolve()}"
    study_name = f"ica_{segment_dir.name}_v2"

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
        direction="minimize",
        load_if_exists=True
    )

    def objective(trial: optuna.Trial) -> float:
        flatline_crit_val = trial.suggest_float("flatline_crit", 3.0, 15.0, step=2.0)
        flatline_crit = flatline_crit_val

        # Narrowed range: 0.70-0.85 (was 0.60-0.90; 0.9 is too aggressive)
        channel_crit_val = trial.suggest_float("channel_crit", 0.70, 0.85, step=0.05)
        channel_crit = channel_crit_val

        line_crit_val = trial.suggest_float("line_crit", 3.0, 8.0, step=1.0)
        line_crit = line_crit_val

        # Narrowed range: 0.50-0.80 (was 0.60-0.90; 0.9 misses most artifacts)
        iclabel_thresh = trial.suggest_float("iclabel_thresh", 0.50, 0.80, step=0.05)

        result = run_ica_trial_matlab(
            work_dir, mat_in, trial.number,
            flatline_crit, channel_crit, line_crit,
            iclabel_thresh, n_channels_original
        )

        if result is None:
            raise optuna.exceptions.TrialPruned()

        # Hard prune if too many channels removed
        removal_fraction = result["n_removed"] / n_channels_original
        if removal_fraction > MAX_CHANNEL_REMOVAL_FRACTION:
            print(f"  Trial {trial.number:3d} | PRUNED: {result['n_removed']}/{n_channels_original} "
                  f"channels removed ({removal_fraction:.0%} > {MAX_CHANNEL_REMOVAL_FRACTION:.0%} limit)")
            raise optuna.exceptions.TrialPruned()

        metrics = compute_ica_metrics(
            result["after_data"], data, sfreq,
            result["n_removed"], result["n_ic_rejected"],
            result["n_ic"], n_channels_original
        )

        flat_str = f"{flatline_crit_val:.0f}"
        ch_str = f"{channel_crit_val:.2f}"
        line_str = f"{line_crit_val:.0f}"
        print(f"  Trial {trial.number:3d} | "
              f"flat={flat_str:>3}, "
              f"ch={ch_str:>4}, "
              f"line={line_str:>3}, "
              f"ICth={iclabel_thresh:.2f} -> "
              f"a_ret={metrics['alpha_retention']:.2f}, "
              f"varDrop={metrics['variance_drop']:.2f}, "
              f"chRm={metrics['n_ch_removed']}, "
              f"nIC={metrics['n_ic_total']}, "
              f"icRej={metrics['n_ic_rejected']} "
              f"(Loss={metrics['loss']:.2f})")

        return metrics["loss"]

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)
    if remaining > 0:
        print(f"Running {remaining} Optuna ICA trials (already complete: {completed})...")
        study.optimize(objective, n_trials=remaining)

    mat_in.unlink(missing_ok=True)

    print("\n" + "=" * 80)
    print("OPTUNA ICA OPTIMIZATION COMPLETE!")
    print(f"Best Trial #{study.best_trial.number}:")
    for k, v in study.best_params.items():
        print(f"  {k:25s} = {v}")
    print(f"  Best Loss = {study.best_value:.4f}")
    print("=" * 80)

    # Save best parameters in both standard filenames for compatibility
    res_dict = {
        "best_trial": study.best_trial.number,
        "best_params": study.best_params,
        "best_value": study.best_value
    }
    best_params_json = segment_dir / "optuna_ica_best_params.json"
    with open(best_params_json, "w", encoding="utf-8") as f:
        json.dump(res_dict, f, indent=2)

    alt_params_json = segment_dir / "ica_optuna_best.json"
    with open(alt_params_json, "w", encoding="utf-8") as f:
        json.dump(res_dict, f, indent=2)

    plot_optuna_ica_summary(study, segment_dir)
    return study.best_params


def plot_optuna_ica_summary(study: optuna.Study, segment_dir: Path):
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        return

    scores = [t.value for t in trials]
    idx = list(range(len(trials)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Optuna ICA Parameter Optimization Summary\n"
        f"Best Loss={study.best_value:.2f}",
        fontsize=12, fontweight="bold"
    )

    # Progress
    axes[0, 0].plot(idx, scores, "o-", lw=1, ms=4, color="#636e72", alpha=0.6, label="Trial")
    axes[0, 0].plot(idx, np.minimum.accumulate(scores), lw=2.5, color="#00b894", label="Best so far")
    axes[0, 0].set_xlabel("Trial")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Optimization Progress")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # channel_crit vs score
    ch_crit_vals = [t.params.get("channel_crit", 0.7) for t in trials]
    axes[0, 1].scatter(ch_crit_vals, scores, c=scores, cmap="RdYlGn_r", s=60, alpha=0.8, edgecolors="k", lw=0.3)
    if "channel_crit" in study.best_params:
        axes[0, 1].axvline(study.best_params["channel_crit"], color="green", lw=2, ls="--")
    axes[0, 1].set_xlabel("ChannelCriterion")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("ChannelCriterion vs Loss")
    axes[0, 1].grid(True, alpha=0.3)

    # iclabel_thresh vs score
    iclabel_vals = [t.params.get("iclabel_thresh", 0.7) for t in trials]
    axes[1, 0].scatter(iclabel_vals, scores, c=scores, cmap="RdYlGn_r", s=60, alpha=0.8, edgecolors="k", lw=0.3)
    if "iclabel_thresh" in study.best_params:
        axes[1, 0].axvline(study.best_params["iclabel_thresh"], color="green", lw=2, ls="--")
    axes[1, 0].set_xlabel("ICLabel Threshold")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("ICLabel Threshold vs Loss")
    axes[1, 0].grid(True, alpha=0.3)

    # flatline_crit vs score
    flat_vals = [t.params.get("flatline_crit", 10) for t in trials]
    axes[1, 1].scatter(flat_vals, scores, c=scores, cmap="RdYlGn_r", s=60, alpha=0.8, edgecolors="k", lw=0.3)
    if "flatline_crit" in study.best_params:
        axes[1, 1].axvline(study.best_params["flatline_crit"], color="green", lw=2, ls="--")
    axes[1, 1].set_xlabel("FlatlineCriterion (sec)")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_title("FlatlineCriterion vs Loss")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_png = segment_dir / "optuna_ica_result.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Optuna ICA plot: {out_png.name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optuna ICA parameter optimization")
    parser.add_argument("--segment-dir", type=Path, default=DEFAULT_SEGMENT_DIR)
    parser.add_argument("--n-trials", type=int, default=20)
    args = parser.parse_args()
    run_optuna_ica(args.segment_dir, args.n_trials)
