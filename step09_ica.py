"""
STEP 09 (ICA): Bad-channel detection + ICA + ICLabel auto-rejection
====================================================================
Input : data/1916/derivatives/03_bcg/segmentXX/segmentXX_bcg_clean.fif
        (250 Hz, band-passed 1-100 Hz, BCG-corrected, 95 EEG + ECG)

Pipeline (single MATLAB/EEGLAB session, mirrors the project's stage plan):
  1. Attach standard_1005 channel locations (needed by clean_rawdata & ICLabel).
  2. Bad-channel detection with clean_rawdata  -- DETECTION ONLY:
       - FlatlineCriterion + ChannelCriterion + LineNoiseCriterion ON
       - BurstCriterion = 'off'   (NO ASR)
       - WindowCriterion = 'off'  (NO time-window cutting -> continuum intact
                                    so nothing downstream is torn at the seams)
  3. runica (extended Infomax) on the channel-pruned data.
  4. ICLabel classification (Brain/Muscle/Eye/Heart/LineNoise/ChannelNoise/Other).
  5. Automatic component rejection: flag & remove non-brain ICs whose class
     probability exceeds ARTIFACT_PROB_THRESH (Muscle/Eye/Heart/Line/Chan).
  6. Interpolate the removed channels back (spherical) and set average reference
     -> every segment ends with the SAME full montage, comparable across runs.

The ECG channel is dropped up front: it already did its job in BCG and would
only confuse clean_rawdata (flagged bad) and ICLabel.

Outputs:
  data/1916/derivatives/05_ica/segmentXX/segmentXX_ica_clean.fif
  data/1916/derivatives/05_ica/segmentXX/segmentXX_ica_metrics.json
  data/1916/derivatives/05_ica/segmentXX/segmentXX_ica_report.html
  data/1916/derivatives/04_channels/segmentXX/removed_channels.json
  qc/1916/ica/segmentXX_iclabel.png
  qc/1916/ica/segmentXX_spectra.png
  qc/1916/ica/segmentXX_channels.png
"""
from __future__ import annotations
from pathlib import Path
import json, gc, subprocess
import numpy as np
import scipy.io as sio
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

try:
    from .config import (PROJECT_ROOT, DATA_ROOT, DEFAULT_EXPERIMENT, MATLAB_BIN,
                         EEGLAB_DIR, DEFAULT_SEGMENT_DIR, ALPHA_BAND, EVAL_CHANNELS)
except ImportError:
    from config import (PROJECT_ROOT, DATA_ROOT, DEFAULT_EXPERIMENT, MATLAB_BIN,
                        EEGLAB_DIR, DEFAULT_SEGMENT_DIR, ALPHA_BAND, EVAL_CHANNELS)

ECG_CH = "ECG"
DIPFIT_DIR = EEGLAB_DIR / "plugins" / "dipfit"
STD_1005 = DIPFIT_DIR / "standard_BEM" / "elec" / "standard_1005.elc"

# clean_rawdata channel-detection parameters (NO ASR, NO window cutting)
FLATLINE_SEC       = 5       # a channel flat for > this many seconds is bad
CHANNEL_CORR       = 0.80    # min correlation with reconstruction from neighbours
LINE_NOISE_CRIT    = 4       # line-noise / signal ratio threshold (higher = laxer)

# ICLabel auto-rejection: remove ICs whose non-brain class prob exceeds this.
ARTIFACT_PROB_THRESH = 0.80
# classes order in ICLabel output columns:
ICLABEL_CLASSES = ["Brain", "Muscle", "Eye", "Heart", "LineNoise", "ChannelNoise", "Other"]
# which classes are auto-removed when above threshold:
REJECT_CLASSES = ["Muscle", "Eye", "Heart", "LineNoise", "ChannelNoise"]


# ─────────────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_bcg_fif(segment_dir: Path) -> Path:
    seg = segment_dir.name
    p = DATA_ROOT / DEFAULT_EXPERIMENT / "derivatives" / "03_bcg" / seg / f"{seg}_bcg_clean.fif"
    if not p.exists():
        raise FileNotFoundError(f"BCG-clean input not found: {p}. Run step08 first.")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# MATLAB stage: clean_rawdata (detect) + runica + ICLabel + reject + interp + avgref
# ─────────────────────────────────────────────────────────────────────────────

def run_ica_matlab(eeg_data: np.ndarray, ch_names: list[str], sfreq: float,
                   work_dir: Path):
    """
    Returns dict loaded from MATLAB output:
      clean        : (n_ch, n_samp) fully processed EEG (interp + avg ref)
      removed_chans: list[str] channels clean_rawdata flagged bad
      ic_classes   : (n_ic, 7) ICLabel probabilities
      ic_labels    : (n_ic,) argmax class index (1-based)
      rejected_ics : list[int] 1-based IC indices auto-removed
      n_ic         : number of ICs
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    in_mat  = work_dir / "ica_input.mat"
    out_mat = work_dir / "ica_output.mat"
    m_file  = work_dir / "ica_run.m"

    sio.savemat(in_mat, {
        "data": eeg_data.astype(np.float32),
        "srate": float(sfreq),
        "labels": np.array(ch_names, dtype=object),
    }, do_compression=True)

    reject_idx = [ICLABEL_CLASSES.index(c) + 1 for c in REJECT_CLASSES]  # 1-based cols
    reject_ml = "[" + " ".join(str(i) for i in reject_idx) + "]"

    lines = [
        f"addpath('{EEGLAB_DIR.resolve()}');",
        f"addpath('{(EEGLAB_DIR / 'plugins' / 'clean_rawdata').resolve()}');",
        f"addpath('{(EEGLAB_DIR / 'plugins' / 'ICLabel').resolve()}');",
        "eeglab nogui;",
        f"S = load('{in_mat.resolve()}');",
        "EEG = eeg_emptyset();",
        "EEG.data = double(S.data);",
        "EEG.srate = double(S.srate);",
        "EEG.nbchan = size(S.data,1);",
        "EEG.pnts = size(S.data,2);",
        "EEG.trials = 1;",
        "EEG.xmin = 0; EEG.xmax = (EEG.pnts-1)/EEG.srate;",
        "EEG.chanlocs = struct([]);",
        "for i = 1:EEG.nbchan",
        "  if iscell(S.labels); EEG.chanlocs(i).labels = char(S.labels{i});",
        "  else; EEG.chanlocs(i).labels = deblank(S.labels(i,:)); end",
        "end",
        "EEG = eeg_checkset(EEG);",
        # channel locations from standard_1005 (match by label)
        f"EEG = pop_chanedit(EEG, 'lookup', '{STD_1005.resolve()}');",
        "EEG = eeg_checkset(EEG);",
        "chanlocs_full = EEG.chanlocs;",   # keep for interpolation
        "labels_full = {EEG.chanlocs.labels};",
        # ---- bad channel detection ONLY (no ASR, no window removal) ----
        "EEG_clean = pop_clean_rawdata(EEG, "
        f"'FlatlineCriterion', {FLATLINE_SEC}, "
        f"'ChannelCriterion', {CHANNEL_CORR}, "
        f"'LineNoiseCriterion', {LINE_NOISE_CRIT}, "
        "'Highpass', 'off', "
        "'BurstCriterion', 'off', "
        "'WindowCriterion', 'off', "
        "'BurstRejection', 'off', "
        "'Distance', 'Euclidian');",
        "labels_kept = {EEG_clean.chanlocs.labels};",
        "removed = setdiff(labels_full, labels_kept);",
        "EEG = EEG_clean; clear EEG_clean;",
        "EEG = eeg_checkset(EEG);",
        # ---- ICA (extended infomax) ----
        "fprintf('Running runica on %d channels...\\n', EEG.nbchan);",
        "EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'interrupt', 'off');",
        "EEG = eeg_checkset(EEG);",
        # ---- ICLabel ----
        "EEG = pop_iclabel(EEG, 'default');",
        "ic_classes = EEG.etc.ic_classification.ICLabel.classifications;",  # (n_ic,7)
        # ---- auto reject non-brain ICs above threshold ----
        f"reject_cols = {reject_ml};",
        f"thr = {ARTIFACT_PROB_THRESH};",
        "n_ic = size(ic_classes,1);",
        "rej = false(1, n_ic);",
        "for c = 1:n_ic",
        "  probs = ic_classes(c,:);",
        "  [~, cls] = max(probs);",
        "  if any(cls == reject_cols) && probs(cls) >= thr",
        "    rej(c) = true;",
        "  end",
        "end",
        "rejected_ics = find(rej);",
        "fprintf('Rejecting %d of %d ICs\\n', numel(rejected_ics), n_ic);",
        "EEG = pop_subcomp(EEG, rejected_ics, 0);",
        "EEG = eeg_checkset(EEG);",
        # ---- interpolate removed channels back + average reference ----
        "if ~isempty(removed)",
        "  EEG = pop_interp(EEG, chanlocs_full, 'spherical');",
        "end",
        "EEG = pop_reref(EEG, []);",   # average reference
        "EEG = eeg_checkset(EEG);",
        # ---- export ----
        "clean = single(EEG.data);",
        "out_labels = {EEG.chanlocs.labels};",
        "removed_c = removed;",
        f"save('{out_mat.resolve()}', 'clean', 'out_labels', 'removed_c', "
        "'ic_classes', 'rejected_ics', 'n_ic', '-v7');",
        "exit(0);",
    ]
    m_file.write_text("\n".join(lines) + "\n", encoding="ascii")

    print("  Launching MATLAB (clean_rawdata + runica + ICLabel + reject)...")
    res = subprocess.run(
        [str(MATLAB_BIN), "-nodesktop", "-nosplash", "-batch", f"run('{m_file.resolve()}')"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0 or not out_mat.exists():
        tail = "\n".join(res.stdout.splitlines()[-40:])
        raise RuntimeError(f"MATLAB ICA stage failed (code {res.returncode}):\n{tail}")

    m = sio.loadmat(out_mat, squeeze_me=True)
    out = {
        "clean": np.array(m["clean"], dtype=np.float64),
        "out_labels": [str(x) for x in np.atleast_1d(m["out_labels"])],
        "removed_chans": [str(x) for x in np.atleast_1d(m["removed_c"])] if m["removed_c"].size else [],
        "ic_classes": np.atleast_2d(np.array(m["ic_classes"], dtype=float)),
        "rejected_ics": [int(x) for x in np.atleast_1d(m["rejected_ics"])] if np.size(m["rejected_ics"]) else [],
        "n_ic": int(m["n_ic"]),
    }
    in_mat.unlink(missing_ok=True); m_file.unlink(missing_ok=True); out_mat.unlink(missing_ok=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# QC metrics + plots
# ─────────────────────────────────────────────────────────────────────────────

def _bandpower(f, psd, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(np.trapz(psd[m], f[m])) if np.any(m) else 0.0


def compute_ica_metrics(before, after, sfreq, names_before, names_after, eval_chs):
    """Alpha retention + broadband variance drop on eval channels."""
    nperseg = int(min(4 * sfreq, before.shape[1]))
    alpha_b = alpha_a = var_b = var_a = 0.0
    used = 0
    for c in eval_chs:
        if c in names_before and c in names_after:
            ib, ia = names_before.index(c), names_after.index(c)
            f, pb = welch(before[ib], sfreq, nperseg=nperseg)
            _, pa = welch(after[ia],  sfreq, nperseg=nperseg)
            alpha_b += _bandpower(f, pb, *ALPHA_BAND)
            alpha_a += _bandpower(f, pa, *ALPHA_BAND)
            var_b += float(np.var(before[ib]))
            var_a += float(np.var(after[ia]))
            used += 1
    alpha_ret = float(np.clip(alpha_a / max(alpha_b, 1e-20), 0.0, 1.5)) if used else 0.0
    var_drop  = float(np.clip(1.0 - var_a / max(var_b, 1e-20), -1.0, 1.0)) if used else 0.0
    return {"alpha_retention": alpha_ret, "variance_drop": var_drop, "n_eval": used}


def plot_iclabel(ic_classes, rejected_ics, out_png: Path):
    n_ic = ic_classes.shape[0]
    top_cls = np.argmax(ic_classes, axis=1)
    counts = np.bincount(top_cls, minlength=7)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#00b894", "#e17055", "#0984e3", "#d63031", "#6c5ce7", "#fdcb6e", "#b2bec3"]
    axes[0].bar(ICLABEL_CLASSES, counts, color=colors)
    axes[0].set_title(f"IC dominant class ({n_ic} ICs, {len(rejected_ics)} rejected)",
                      fontweight="bold")
    axes[0].tick_params(axis="x", rotation=40); axes[0].grid(True, alpha=0.3, axis="y")
    # per-IC stacked probability
    bottom = np.zeros(n_ic)
    x = np.arange(n_ic)
    for k in range(7):
        axes[1].bar(x, ic_classes[:, k], bottom=bottom, color=colors[k],
                    width=0.9, label=ICLABEL_CLASSES[k])
        bottom += ic_classes[:, k]
    for r in rejected_ics:
        axes[1].axvline(r - 1, color="k", lw=0.6, ls=":")
    axes[1].set_title("Per-IC class probabilities (dotted = rejected)", fontweight="bold")
    axes[1].set_xlabel("IC #"); axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_spectra(before, after, sfreq, names_b, names_a, eval_chs, out_png: Path):
    idx = [c for c in eval_chs if c in names_b and c in names_a]
    n = len(idx); ncol = 2; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.6 * nrow), squeeze=False)
    nperseg = int(min(4 * sfreq, before.shape[1]))
    lp = min(100, sfreq / 2)
    for k, c in enumerate(idx):
        ax = axes[k // ncol][k % ncol]
        f, pb = welch(before[names_b.index(c)], sfreq, nperseg=nperseg)
        _, pa = welch(after[names_a.index(c)],  sfreq, nperseg=nperseg)
        m = f <= lp
        ax.semilogy(f[m], pb[m], color="#e17055", lw=0.9, label="Before ICA")
        ax.semilogy(f[m], pa[m], color="#00b894", lw=1.1, label="After ICA")
        ax.axvspan(*ALPHA_BAND, color="#0984e3", alpha=0.08)
        ax.set_title(c, fontsize=9, fontweight="bold")
        ax.set_xlim(0, lp); ax.grid(True, alpha=0.3, which="both")
        if k == 0: ax.legend(fontsize=7, loc="upper right")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("PSD before vs after ICA (blue=alpha 8-13 Hz)", fontweight="bold", y=1.0)
    fig.tight_layout(); fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_channels(all_names, removed, out_png: Path):
    fig, ax = plt.subplots(figsize=(12, 2.2))
    status = [1 if n in removed else 0 for n in all_names]
    colors = ["#d63031" if s else "#00b894" for s in status]
    ax.bar(range(len(all_names)), np.ones(len(all_names)), color=colors, width=1.0)
    ax.set_xticks(range(len(all_names)))
    ax.set_xticklabels(all_names, rotation=90, fontsize=5)
    ax.set_yticks([]); ax.set_xlim(-0.5, len(all_names) - 0.5)
    ax.set_title(f"Channels: {len(removed)} removed (red) of {len(all_names)} "
                 f"[interpolated back]", fontweight="bold")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────────────

def _b64(path: Path) -> str:
    import base64
    return base64.b64encode(path.read_bytes()).decode() if path.exists() else ""


def _grade(metrics, n_removed, n_total, n_rej, n_ic):
    if metrics["alpha_retention"] < 0.5:
        return ("REVIEW", "#fdcb6e")
    if n_removed > 0.25 * n_total:
        return ("REVIEW", "#fdcb6e")
    if n_ic > 0 and n_rej > 0.6 * n_ic:
        return ("REVIEW", "#fdcb6e")
    return ("GOOD", "#00b894")


def generate_ica_html(seg_name, metrics, removed, all_names, ic_classes, rejected_ics,
                      n_ic, sfreq, iclabel_png, spectra_png, channels_png, out_html):
    from datetime import datetime
    label, color = _grade(metrics, len(removed), len(all_names), len(rejected_ics), n_ic)
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top_cls = np.argmax(ic_classes, axis=1) if n_ic else np.array([])
    counts = np.bincount(top_cls, minlength=7) if n_ic else np.zeros(7, int)
    cls_rows = "".join(
        f"<tr><td>{ICLABEL_CLASSES[k]}</td><td>{int(counts[k])}</td></tr>"
        for k in range(7))
    removed_txt = ", ".join(removed) if removed else "none"
    rej_txt = ", ".join(f"IC{r}" for r in rejected_ics) if rejected_ics else "none"
    html = (
        '<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">'
        f'<title>ICA Report - {seg_name}</title><style>'
        'body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f5f6fa;color:#2d3436;margin:0;padding:24px;}'
        '.card{background:#fff;border-radius:12px;padding:20px 26px;margin:0 auto 20px;max-width:1100px;box-shadow:0 2px 10px rgba(0,0,0,.06);}'
        'h1{font-size:22px;} h2{font-size:17px;border-left:4px solid #6c5ce7;padding-left:10px;}'
        'table{border-collapse:collapse;width:auto;font-size:14px;} th,td{border:1px solid #dfe6e9;padding:6px 14px;text-align:center;} th{background:#f1f2f6;}'
        f'.badge{{display:inline-block;padding:6px 16px;border-radius:20px;color:#fff;font-weight:700;background:{color};}}'
        '.kpi{display:flex;gap:18px;flex-wrap:wrap;} .kpi div{flex:1;min-width:150px;background:#f8f9fa;border-radius:10px;padding:14px;text-align:center;}'
        '.kpi b{display:block;font-size:26px;color:#6c5ce7;} img{max-width:100%;border-radius:8px;} code{background:#f1f2f6;padding:2px 6px;border-radius:4px;}'
        '</style></head><body>'
        f'<div class="card"><h1>&#129504; ICA / ICLabel Report - {seg_name}</h1>'
        f'<p>Generated: {dt} | {sfreq:.0f} Hz | runica extended | ICLabel default</p>'
        f'<p>Grade: <span class="badge">{label}</span></p></div>'

        '<div class="card"><h2>&#127919; Summary</h2><div class="kpi">'
        f'<div><b>{len(removed)}</b>Bad channels (interp. back)</div>'
        f'<div><b>{len(rejected_ics)}/{n_ic}</b>ICs rejected</div>'
        f'<div><b>{metrics["alpha_retention"]*100:.1f}%</b>Alpha retention 8-13 Hz</div>'
        f'<div><b>{metrics["variance_drop"]*100:.1f}%</b>Broadband variance drop</div></div>'
        f'<p style="font-size:13px;color:#636e72;margin-top:12px;">'
        f'Removed channels: <code>{removed_txt}</code><br>Rejected ICs: <code>{rej_txt}</code></p>'
        '<p style="font-size:13px;color:#636e72;">clean_rawdata ran in detection-only '
        'mode (no ASR, no window cutting). Removed channels were interpolated back and '
        'the data re-referenced to average, so the montage is complete and comparable '
        'across segments.</p></div>'

        '<div class="card"><h2>&#128202; ICLabel component classification</h2>'
        f'<img src="data:image/png;base64,{_b64(iclabel_png)}">'
        f'<table><tr><th>Class</th><th>#ICs (dominant)</th></tr>{cls_rows}</table></div>'

        '<div class="card"><h2>&#128225; Bad-channel map</h2>'
        f'<img src="data:image/png;base64,{_b64(channels_png)}"></div>'

        '<div class="card"><h2>&#128200; Spectra before/after ICA</h2>'
        f'<img src="data:image/png;base64,{_b64(spectra_png)}"></div>'
        '</body></html>'
    )
    out_html.write_text(html, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_ica_pipeline(segment_dir: Path = DEFAULT_SEGMENT_DIR):
    segment_dir = Path(segment_dir).resolve()
    seg_name = segment_dir.name
    print("=" * 75)
    print(f"[STEP 09 - ICA] Bad-channel + runica + ICLabel for: {seg_name}")
    print("=" * 75)

    deriv = DATA_ROOT / DEFAULT_EXPERIMENT / "derivatives"
    chan_dir = deriv / "04_channels" / seg_name
    ica_dir  = deriv / "05_ica" / seg_name
    qc_dir   = PROJECT_ROOT / "qc" / DEFAULT_EXPERIMENT / "ica"
    for d in (chan_dir, ica_dir, qc_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load BCG-clean fif, drop ECG (EEG-only for ICA)
    fif = _find_bcg_fif(segment_dir)
    print(f"  Loading: {fif.name}")
    raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])
    if ECG_CH in raw.ch_names:
        raw.drop_channels([ECG_CH])
    ch_names = list(raw.ch_names)
    before_eeg = raw.get_data() * 1e6      # V -> uV
    print(f"  {len(ch_names)} EEG channels @ {sfreq:.0f} Hz, {raw.n_times} samples")

    # 2. MATLAB stage
    work_dir = ica_dir / "_matlab_tmp"
    r = run_ica_matlab(before_eeg, ch_names, sfreq, work_dir)
    after_eeg = r["clean"]                  # already interp + avg ref, uV
    names_after = r["out_labels"]
    removed = r["removed_chans"]
    print(f"  Bad channels removed: {len(removed)} -> {removed if removed else 'none'}")
    print(f"  ICs: {r['n_ic']}, rejected: {len(r['rejected_ics'])} -> {r['rejected_ics']}")

    # 3. QC metrics + plots
    eval_chs = [c for c in EVAL_CHANNELS if c in ch_names]
    metrics = compute_ica_metrics(before_eeg, after_eeg, sfreq, ch_names, names_after, eval_chs)
    print(f"  Alpha retention: {metrics['alpha_retention']*100:.1f}% | "
          f"variance drop: {metrics['variance_drop']*100:.1f}%")

    iclabel_png = qc_dir / f"{seg_name}_iclabel.png"
    spectra_png = qc_dir / f"{seg_name}_spectra.png"
    channels_png = qc_dir / f"{seg_name}_channels.png"
    plot_iclabel(r["ic_classes"], r["rejected_ics"], iclabel_png)
    plot_spectra(before_eeg, after_eeg, sfreq, ch_names, names_after, eval_chs, spectra_png)
    plot_channels(ch_names, removed, channels_png)

    # 4. Save cleaned fif (final EEG for this stage)
    info = mne.create_info(names_after, sfreq, ch_types="eeg")
    raw_out = mne.io.RawArray(after_eeg * 1e-6, info, verbose=False)   # uV -> V
    out_fif = ica_dir / f"{seg_name}_ica_clean.fif"
    raw_out.save(out_fif, overwrite=True, verbose=False)
    print(f"  Saved ICA-clean: {out_fif}")

    # 5. Persist channel + metrics json
    (chan_dir / "removed_channels.json").write_text(
        json.dumps({"removed": removed, "n_total": len(ch_names)}, indent=2), encoding="utf-8")
    metrics_out = {
        **metrics,
        "n_channels_removed": len(removed),
        "removed_channels": removed,
        "n_ic": r["n_ic"],
        "n_ic_rejected": len(r["rejected_ics"]),
        "rejected_ics": r["rejected_ics"],
        "sfreq": sfreq,
        "artifact_prob_thresh": ARTIFACT_PROB_THRESH,
    }
    (ica_dir / f"{seg_name}_ica_metrics.json").write_text(
        json.dumps(metrics_out, indent=2), encoding="utf-8")

    # 6. HTML report
    out_html = ica_dir / f"{seg_name}_ica_report.html"
    generate_ica_html(seg_name, metrics, removed, ch_names, r["ic_classes"],
                      r["rejected_ics"], r["n_ic"], sfreq,
                      iclabel_png, spectra_png, channels_png, out_html)
    print(f"  HTML report: {out_html}")

    # cleanup
    try: work_dir.rmdir()
    except OSError: pass
    print("=" * 75); print("  [STEP 09 - ICA] DONE."); print("=" * 75)
    return out_html


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Bad-channel detection + ICA + ICLabel")
    ap.add_argument("--segment-dir", type=Path, default=DEFAULT_SEGMENT_DIR)
    args = ap.parse_args()
    run_ica_pipeline(args.segment_dir.resolve())
