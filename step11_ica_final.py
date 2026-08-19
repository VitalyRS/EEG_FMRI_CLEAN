"""
STEP 11: Apply Optimized ICA Parameters to Full BCG-Cleaned Data
==================================================================
Uses best parameters from step10 Optuna optimization.
Runs clean_rawdata + runica + ICLabel + reject on full dataset.
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
    from .config import MATLAB_BIN, EEGLAB_DIR, PROJECT_ROOT, DEFAULT_SEGMENT_DIR, ALPHA_BAND, EVAL_CHANNELS
except ImportError:
    from config import MATLAB_BIN, EEGLAB_DIR, PROJECT_ROOT, DEFAULT_SEGMENT_DIR, ALPHA_BAND, EVAL_CHANNELS

DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_EXPERIMENT = "1916"
DIPFIT_DIR = EEGLAB_DIR / "plugins" / "dipfit"
STD_1005 = DIPFIT_DIR / "standard_BEM" / "elec" / "standard_1005.elc"


def _find_bcg_fif(segment_dir: Path) -> Path:
    seg = segment_dir.name
    p = DATA_ROOT / DEFAULT_EXPERIMENT / "derivatives" / "03_bcg" / seg / f"{seg}_bcg_clean.fif"
    if not p.exists():
        raise FileNotFoundError(f"BCG input not found: {p}. Run step08 first.")
    return p


def apply_optimized_ica(segment_dir: Path = DEFAULT_SEGMENT_DIR):
    segment_dir = Path(segment_dir).resolve()
    seg = segment_dir.name
    print("=" * 80)
    print(f"[STEP 11] Applying Optimized ICA Parameters to Full Dataset: {seg}")
    print("=" * 80)

    # Load best params from step10 with fallback to standard robust defaults
    params_json = segment_dir / "optuna_ica_best_params.json"
    if not params_json.exists():
        alt_json = segment_dir / "ica_optuna_best.json"
        if alt_json.exists():
            params_json = alt_json

    if params_json.exists():
        with open(params_json, "r") as f:
            pdata = json.load(f)
            best = pdata.get("best_params", pdata)
        print(f"  Loaded best parameters from {params_json.name}:")
    else:
        print(f"  [WARN] Parameter file not found in {segment_dir}. Using robust defaults.")
        best = {
            "flatline_crit": 5.0,
            "channel_crit": 0.85,
            "line_crit": 4.0,
            "iclabel_thresh": 0.80  # Conservative: only clear artifacts rejected
        }

    # SAFETY GUARD: clamp Optuna params to prevent over-cleaning.
    # A threshold below 0.65 rejects "probably brain" ICs (P(brain)~0.4) and
    # destroys the signal (previously 0.6 -> 69% of ICs rejected, 77% var drop).
    thr_raw = float(best.get("iclabel_thresh", 0.80))
    if thr_raw < 0.65:
        print(f"    [GUARD] iclabel_thresh={thr_raw} is unsafe (< 0.65), clamping to 0.80")
        best["iclabel_thresh"] = 0.80
    ch_raw = float(best.get("channel_crit", 0.85))
    if ch_raw < 0.70:
        print(f"    [GUARD] channel_crit={ch_raw} is unsafe (< 0.70), clamping to 0.80")
        best["channel_crit"] = 0.80

    for k, v in best.items():
        print(f"    {k:25s} = {v}")

    # Load BCG-cleaned data
    fif = _find_bcg_fif(segment_dir)
    print(f"\n  Loading BCG-cleaned data: {fif.name}")
    raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])

    # Drop ECG channel (it was used in BCG stage, not for ICA)
    eeg_ch_names = [ch for ch in raw.ch_names if ch != "ECG"]
    raw.pick_channels(eeg_ch_names)
    data = raw.get_data()
    ch_names = raw.ch_names

    print(f"  {len(ch_names)} EEG channels @ {sfreq} Hz, {raw.n_times} samples ({raw.times[-1]:.1f}s)")

    work_dir = segment_dir / "ica_final"
    work_dir.mkdir(exist_ok=True)

    mat_in = work_dir / "input_full.mat"
    data_uv = data * 1e6  # Volts → microvolts for EEGLAB
    savemat(mat_in, {
        "data": data_uv.astype(np.float32),
        "srate": sfreq,
        "labels": np.array(ch_names, dtype=object)
    }, do_compression=True)
    print(f"  Prepared input MAT: {data.shape[0]} ch x {data.shape[1]} samples (converted to µV)")

    flatline_crit = best.get("flatline_crit", 5.0)
    channel_crit = best.get("channel_crit", 0.80)
    line_crit = best.get("line_crit", 4.0)
    iclabel_thresh = best.get("iclabel_thresh", 0.70)

    out_mat = work_dir / "result_full.mat"
    m_file = work_dir / "run_final_ica.m"

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

fprintf('Loading full BCG-cleaned dataset...\\n');
load('{mat_in.resolve()}', 'data', 'srate', 'labels');
EEG = eeg_emptyset();
EEG.setname = '{seg}_ica_final';
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

% Lookup standard_1005 coordinates
EEG = pop_chanedit(EEG, 'lookup', '{STD_1005.resolve()}');
EEG = eeg_checkset(EEG);

chanlocs_full = EEG.chanlocs;
orig_chans = {{EEG.chanlocs.labels}};
n_orig = EEG.nbchan;

fprintf('Running clean_rawdata with optimized parameters...\\n');
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
fprintf('Running runica (extended Infomax)...\\n');
if data_rank < n_chans_after
    fprintf('Reducing dimensionality to rank %d via PCA\\n', data_rank);
    EEG_clean = pop_runica(EEG_clean, 'icatype', 'runica', 'extended', 1, 'interrupt', 'off', 'pca', data_rank);
else
    EEG_clean = pop_runica(EEG_clean, 'icatype', 'runica', 'extended', 1, 'interrupt', 'off');
end

n_ic = size(EEG_clean.icaweights, 1);
fprintf('Number of ICs computed: %d\\n', n_ic);

fprintf('Running ICLabel classification...\\n');
EEG_clean = pop_iclabel(EEG_clean, 'default');
ic_classes = EEG_clean.etc.ic_classification.ICLabel.classifications;

% Balanced rejection rule for EEG-fMRI (matches step10 Phase-3 sweep):
% (1) dominant class is a confident artifact (Muscle=2,Eye=3,Heart=4,Line=5,Chan=6)
%     AND probability >= threshold, OR
% (2) "Other" (col 7) with brain probability < 0.02 (residual fMRI noise that
%     ICLabel can't name, but is 98% sure is NOT brain).
% We do NOT reject on brain_prob<0.20 or Other>=0.90 -- on EEG-fMRI those fire
% on nearly every component and destroy the alpha rhythm (a_ret 0.93 -> 0.14).
artifact_cols = [2, 3, 4, 5, 6];
rej_mask = false(1, n_ic);
for c = 1:n_ic
    probs = ic_classes(c, :);
    [max_prob, dominant] = max(probs);
    brain_prob = probs(1);

    % Rule 1: confident artifact above threshold
    if any(dominant == artifact_cols) && max_prob >= {iclabel_thresh}
        rej_mask(c) = true;
    end

    % Rule 2: "Other" (column 7) with near-zero brain probability
    if dominant == 7 && brain_prob < 0.02
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

if n_removed > 0
    fprintf('Interpolating removed channels spherical...\\n');
    EEG_clean = pop_interp(EEG_clean, chanlocs_full, 'spherical');
end

fprintf('Applying average reference...\\n');
EEG_clean = pop_reref(EEG_clean, []);

after_data = single(EEG_clean.data);
save('{out_mat.resolve()}', 'after_data', 'removed_chans', 'n_removed', 'n_ic_rejected', 'ic_classes', 'reject_ic_list', 'n_ic', '-v7');
clear EEG EEG_clean;
fprintf('=== ICA FINISHED SUCCESSFULLY! ===\\n');
exit(0);
"""

    with open(m_file, "w", encoding="ascii") as f:
        f.write(code)

    print("\n  Running MATLAB ICA (this takes 1-3 minutes)...")
    res = subprocess.run(
        [str(MATLAB_BIN), "-nodesktop", "-nosplash", "-batch", f"run('{m_file.resolve()}')"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    m_file.unlink(missing_ok=True)

    if res.returncode != 0 or not out_mat.exists():
        print(res.stdout[-2000:] if res.stdout else "")
        raise RuntimeError(f"MATLAB ICA failed (code {res.returncode})")

    # Load result
    print("\n  Loading ICA result...")
    m_res = loadmat(out_mat)
    after_data = m_res["after_data"] * 1e-6  # µV → V (back to MNE units)
    n_removed = int(m_res["n_removed"][0, 0])
    n_ic_rejected = int(m_res["n_ic_rejected"][0, 0])
    n_ic = int(m_res["n_ic"][0, 0])
    ic_classes = m_res["ic_classes"]
    if "reject_ic_list" in m_res and m_res["reject_ic_list"].size > 0:
        reject_ic_list = (m_res["reject_ic_list"].flatten() - 1).tolist()
    else:
        reject_ic_list = []

    removed_chans_list = []
    if n_removed > 0 and "removed_chans" in m_res:
        rc = m_res["removed_chans"]
        if hasattr(rc, "size") and rc.size > 0:
            if rc.ndim == 2:
                for i in range(rc.shape[1]):
                    val = rc[0, i]
                    removed_chans_list.append(str(val[0]) if isinstance(val, (np.ndarray, list)) else str(val))
            elif rc.ndim == 1:
                for val in rc:
                    removed_chans_list.append(str(val[0]) if isinstance(val, (np.ndarray, list)) else str(val))

    out_mat.unlink(missing_ok=True)
    mat_in.unlink(missing_ok=True)

    # Compute metrics
    print("  Computing metrics...")
    nperseg = int(min(4 * sfreq, after_data.shape[-1]))
    f_after, psd_after = welch(after_data, sfreq, nperseg=nperseg, axis=-1)
    f_before, psd_before = welch(data, sfreq, nperseg=nperseg, axis=-1)

    m_alpha = (f_after >= ALPHA_BAND[0]) & (f_after <= ALPHA_BAND[1])
    p_alpha_after = float(np.mean(psd_after[:, m_alpha]))
    p_alpha_before = float(np.mean(psd_before[:, m_alpha]))
    alpha_retention = p_alpha_after / max(p_alpha_before, 1e-12)

    var_before = float(np.var(data))
    var_after = float(np.var(after_data))
    variance_drop = (var_before - var_after) / max(var_before, 1e-12)

    metrics = {
        "alpha_retention": float(alpha_retention),
        "variance_drop": float(variance_drop),
        "n_channels_removed": n_removed,
        "removed_channels": removed_chans_list,
        "n_ic": n_ic,
        "n_ic_rejected": n_ic_rejected,
        "rejected_ics": [int(x) for x in reject_ic_list],
        "sfreq": sfreq,
        "iclabel_thresh": float(iclabel_thresh)
    }

    print(f"\n  Alpha retention: {alpha_retention:.2%}")
    print(f"  Variance drop:   {variance_drop:.2%}")
    print(f"  Channels removed: {n_removed} / {len(ch_names)}")
    print(f"  ICs found:        {n_ic}")
    print(f"  ICs rejected:     {n_ic_rejected} / {n_ic}")

    # Save final cleaned data as FIF
    out_deriv_dir = DATA_ROOT / DEFAULT_EXPERIMENT / "derivatives" / "05_ica" / seg
    out_deriv_dir.mkdir(parents=True, exist_ok=True)
    out_fif = out_deriv_dir / f"{seg}_ica_clean.fif"

    montage = mne.channels.make_standard_montage("standard_1005")
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    raw_clean = mne.io.RawArray(after_data, info)
    raw_clean.set_montage(montage, on_missing="ignore")
    raw_clean.save(out_fif, overwrite=True)
    print(f"\n  Saved: {out_fif.name} ({out_fif.stat().st_size / 1e6:.1f} MB)")

    # Save metrics
    metrics_json = out_deriv_dir / f"{seg}_ica_metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save removed channels list
    ch_deriv_dir = DATA_ROOT / DEFAULT_EXPERIMENT / "derivatives" / "04_channels" / seg
    ch_deriv_dir.mkdir(parents=True, exist_ok=True)
    removed_json = ch_deriv_dir / "removed_channels.json"
    with open(removed_json, "w") as f:
        json.dump({"removed": removed_chans_list, "n_total": len(ch_names)}, f, indent=2)

    # Generate QC plots and HTML report
    print("\n  Generating QC plots & report...")
    generate_ica_report(data, after_data, ch_names, sfreq, ic_classes,
                        reject_ic_list, removed_chans_list, metrics, seg)

    print("\n" + "=" * 80)
    print(f"[STEP 11] ICA Complete! Output: {out_fif.name}")
    print("=" * 80)
    return out_fif


def generate_ica_report(data_before, data_after, ch_names, sfreq, ic_classes,
                        reject_ic_list, removed_chans, metrics, seg):
    """Generate HTML report with PSD, ICLabel, and channel plots."""
    qc_dir = PROJECT_ROOT / "qc" / DEFAULT_EXPERIMENT / "ica"
    qc_dir.mkdir(parents=True, exist_ok=True)

    # 1. PSD comparison for eval channels
    eval_ch = ["O1", "Oz", "O2", "Pz", "P3", "P4", "Fz", "Cz"]
    eval_idx = [i for i, ch in enumerate(ch_names) if ch in eval_ch]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    fig.suptitle("PSD before vs after ICA (blue=alpha 8-13 Hz)", fontsize=14, fontweight="bold")

    nperseg = int(min(4 * sfreq, data_before.shape[-1]))
    for i, idx in enumerate(eval_idx):
        f_b, psd_b = welch(data_before[idx], sfreq, nperseg=nperseg)
        f_a, psd_a = welch(data_after[idx], sfreq, nperseg=nperseg)

        axes[i].semilogy(f_b, psd_b, label="Before ICA", color="#00b894", lw=1.5, alpha=0.8)
        axes[i].semilogy(f_a, psd_a, label="After ICA", color="#d63031", lw=1.5, alpha=0.8)
        axes[i].axvspan(8, 13, color="blue", alpha=0.1)
        axes[i].set_xlim(0, 100)
        axes[i].set_title(ch_names[idx])
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend()

    fig.tight_layout()
    spectra_png = qc_dir / f"{seg}_spectra.png"
    fig.savefig(spectra_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. ICLabel classification plot
    n_ic = ic_classes.shape[0]
    classes = ["Brain", "Muscle", "Eye", "Heart", "LineNoise", "ChannelNoise", "Other"]
    colors = ["#00b894", "#e17055", "#0984e3", "#d63031", "#6c5ce7", "#fdcb6e", "#b2bec3"]

    dominant_class = np.argmax(ic_classes, axis=1)
    class_counts = [int(np.sum(dominant_class == i)) for i in range(7)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"IC dominant class ({n_ic} ICs, {len(reject_ic_list)} rejected)", fontweight="bold")

    ax1.bar(classes, class_counts, color=colors, edgecolor="k", lw=0.5)
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.tick_params(axis="x", rotation=45)

    # Stacked bar per IC
    ax2.set_title("Per-IC class probabilities (dotted = rejected)")
    bottom = np.zeros(n_ic)
    for i, (cls, col) in enumerate(zip(classes, colors)):
        ax2.bar(range(n_ic), ic_classes[:, i], bottom=bottom, color=col, width=1.0, edgecolor="none")
        bottom += ic_classes[:, i]

    for ic_idx in reject_ic_list:
        ax2.bar(ic_idx, 1.0, color="none", edgecolor="k", lw=1.5, linestyle=":", width=1.0)

    ax2.set_xlabel("IC #")
    ax2.set_ylabel("Probability")
    ax2.set_xlim(-0.5, n_ic - 0.5)
    ax2.set_ylim(0, 1)
    ax2.legend(classes, loc="upper right", fontsize=8, ncol=7)

    fig.tight_layout()
    iclabel_png = qc_dir / f"{seg}_iclabel.png"
    fig.savefig(iclabel_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Removed channels plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Removed Channels ({len(removed_chans)} / {len(ch_names)})", fontweight="bold")
    if removed_chans:
        ax.barh(removed_chans, [1] * len(removed_chans), color="#d63031", edgecolor="k", lw=0.5)
        ax.set_xlabel("Removed")
    else:
        ax.text(0.5, 0.5, "No channels removed", ha="center", va="center", fontsize=14, color="green")
    ax.set_xlim(0, 1.2)
    fig.tight_layout()
    channels_png = qc_dir / f"{seg}_channels.png"
    fig.savefig(channels_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. HTML report
    import base64
    with open(spectra_png, "rb") as f:
        b64_spectra = base64.b64encode(f.read()).decode("ascii")
    with open(iclabel_png, "rb") as f:
        b64_iclabel = base64.b64encode(f.read()).decode("ascii")
    with open(channels_png, "rb") as f:
        b64_channels = base64.b64encode(f.read()).decode("ascii")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{seg} ICA Report</title>
<style>
body {{font-family:system-ui,sans-serif;margin:2rem;background:#f7f4ef;color:#1f2421;}}
h1,h2 {{color:#1f2421;}}
.card {{background:#fff;border:1px solid #e7e1d7;border-radius:8px;padding:1.5rem;margin:1rem 0;}}
.metrics {{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;}}
.metric {{background:#fbf9f5;padding:1rem;border-radius:4px;border-left:3px solid #c4612f;}}
.metric-label {{font-size:0.85rem;color:#5c635d;text-transform:uppercase;}}
.metric-value {{font-size:1.5rem;font-weight:bold;color:#1f2421;}}
img {{max-width:100%;height:auto;border-radius:4px;}}
</style></head><body>
<h1>{seg} ICA Report</h1>
<div class="card"><h2>Metrics</h2><div class="metrics">
<div class="metric"><div class="metric-label">Alpha Retention</div><div class="metric-value">{metrics['alpha_retention']:.1%}</div></div>
<div class="metric"><div class="metric-label">Variance Drop</div><div class="metric-value">{metrics['variance_drop']:.1%}</div></div>
<div class="metric"><div class="metric-label">Channels Removed</div><div class="metric-value">{metrics['n_channels_removed']} / {len(ch_names)}</div></div>
<div class="metric"><div class="metric-label">ICs Found</div><div class="metric-value">{metrics['n_ic']}</div></div>
<div class="metric"><div class="metric-label">ICs Rejected</div><div class="metric-value">{metrics['n_ic_rejected']} / {metrics['n_ic']}</div></div>
<div class="metric"><div class="metric-label">ICLabel Threshold</div><div class="metric-value">{metrics['iclabel_thresh']:.2f}</div></div>
</div></div>
<div class="card"><h2>PSD Before vs After ICA</h2><img src="data:image/png;base64,{b64_spectra}"></div>
<div class="card"><h2>ICLabel Classification</h2><img src="data:image/png;base64,{b64_iclabel}"></div>
<div class="card"><h2>Removed Channels</h2><img src="data:image/png;base64,{b64_channels}"></div>
</body></html>"""

    out_deriv_dir = DATA_ROOT / DEFAULT_EXPERIMENT / "derivatives" / "05_ica" / seg
    report_html = out_deriv_dir / f"{seg}_ica_report.html"
    with open(report_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"    {report_html.name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Apply optimized ICA to full dataset")
    parser.add_argument("--segment-dir", type=Path, default=DEFAULT_SEGMENT_DIR)
    args = parser.parse_args()
    apply_optimized_ica(args.segment_dir)
