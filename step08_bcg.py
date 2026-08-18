"""
STEP 08 (BCG): Resample 5000->250 Hz  +  BCG Artifact Removal via EEGLAB fMRIB OBS
==================================================================================
Stage 02: Anti-aliased resampling (5000 -> 250 Hz, Nyquist = 125 Hz).
Stage 03: BCG (ballistocardiogram) removal using the reference EEGLAB fMRIB
          plugin (Niazy/Iannetti, Oxford):
            pop_fmrib_qrsdetect -> detect R-peaks from the ECG channel
            pop_fmrib_pas('obs', npc) -> Optimal Basis Set subtraction
          We do NOT reimplement OBS in Python; the plugin is the peer-reviewed
          implementation. Python only orchestrates, evaluates quality, and
          sweeps the one hyperparameter (npc = number of PCA basis components).

Search: npc is a single small integer, so we run the whole grid npc in {3..8}
        inside a single MATLAB session (avoids launching MATLAB per Optuna
        trial) and pick the npc that maximises the quality metric below.

Quality metric - BCG Suppression Index (BSI):
  BSI = 0.6 * (cardiac-band 0.7-4 Hz suppression)
      + 0.4 * (drop in |correlation| between EEG and ECG)
      - 0.5 * (alpha-band 8-13 Hz power LOSS)
  It cannot be gamed by deleting real EEG: eating alpha is penalised.

Outputs:
  data/1916/derivatives/02_resampled250/segmentXX/segmentXX_250hz.fif
  data/1916/derivatives/03_bcg/segmentXX/segmentXX_bcg_clean.fif
  data/1916/derivatives/03_bcg/segmentXX/segmentXX_bcg_metrics.json
  data/1916/derivatives/03_bcg/segmentXX/segmentXX_bcg_report.html
  qc/1916/bcg/segmentXX_rpeaks.png
  qc/1916/bcg/segmentXX_spectra.png
  qc/1916/bcg/segmentXX_npc_sweep.png
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

TARGET_SFREQ = 250.0          # Hz after resampling (Nyquist = 125 Hz)
ECG_CH       = "ECG"          # name of cardiac channel in the .set
CARDIAC_BAND = (0.7, 4.0)     # Hz - fundamental + harmonics of BCG
NPC_GRID     = [1, 2, 3, 4, 5, 6, 7, 8]   # OBS basis components to sweep (incl. gentle 1-2)
FMRIB_DIR    = EEGLAB_DIR / "plugins" / "fMRIb2.1"

# Band-pass applied AFTER resampling and BEFORE BCG. High-pass removes the
# sub-1 Hz drift that otherwise makes OBS smear its subtraction across the
# whole low band (eating alpha) and destabilises the quality metric. Low-pass
# at 100 Hz sits well below the 125 Hz Nyquist of the 250 Hz data, leaving a
# ~25 Hz transition margin, so 1-100 Hz is clean. 100 Hz keeps low/mid gamma.
FILTER_HP = 1.0               # Hz high-pass
FILTER_LP = 100.0             # Hz low-pass

# npc SELECTION POLICY (OBS -> ICA architecture):
# BCG is NOT required to be surgically alpha-preserving here. Its job is to
# remove the bulk of the pulse artifact; the fine "alpha vs BCG residual"
# separation is left to the later ICA / ICLabel stage. So we pick npc by
# strongest cardiac suppression. Alpha retention is reported as a diagnostic
# only (it cannot cleanly distinguish "ate the rhythm" from "removed BCG power
# that overlapped the 8-13 Hz band"), and is used just to warn, not to gate.
ALPHA_WARN_MIN = 0.60   # below this, flag REVIEW for a human/QC glance

# Channels used to SCORE each npc. Must be the SAME set used for the final
# metric, otherwise the sweep is biased. Frontal channels have the strongest
# BCG but tolerate aggressive npc; occipital carry the alpha we must protect.
# Scoring on both together prevents picking an npc that over-cleans the alpha.
PILOT_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "Fz", "Cz", "C3", "C4",
                  "O1", "Oz", "O2", "Pz", "P3", "P4"]


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def _find_bergen_set(segment_dir: Path) -> Path:
    """Search for a Bergen-cleaned .set in segment_dir or derivatives/01_bergen."""
    candidates = list(segment_dir.glob("*bergen*.set"))
    if not candidates:
        deriv = DATA_ROOT / DEFAULT_EXPERIMENT / "derivatives" / "01_bergen" / segment_dir.name
        candidates = list(deriv.glob("*.set"))
    if not candidates:
        raise FileNotFoundError(
            f"No Bergen-cleaned .set found in {segment_dir} or "
            f"derivatives/01_bergen/{segment_dir.name}. Run run_all.py first."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_set_as_raw(set_path: Path) -> mne.io.RawArray:
    """Load EEGLAB .set (MAT v5) into an MNE RawArray (units: V)."""
    m = sio.loadmat(str(set_path), squeeze_me=True)
    data = np.array(m["data"], dtype=np.float64)          # (n_ch, n_pnts), uV
    sfreq = float(m["srate"])
    cl = m["chanlocs"]
    if cl.dtype.names and "labels" in cl.dtype.names:
        ch_names = [str(c["labels"]) for c in cl.flat]
    else:
        ch_names = [f"Ch{i+1}" for i in range(data.shape[0])]
    ch_types = ["ecg" if n == ECG_CH else "eeg" for n in ch_names]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    return mne.io.RawArray(data * 1e-6, info, verbose=False)   # uV -> V


# ─────────────────────────────────────────────────────────────────────────────
# Stage 02: Resample
# ─────────────────────────────────────────────────────────────────────────────

def resample_raw(raw: mne.io.BaseRaw, target_sfreq: float = TARGET_SFREQ) -> mne.io.BaseRaw:
    """Anti-aliased resample (MNE low-passes automatically before decimation)."""
    if abs(raw.info["sfreq"] - target_sfreq) < 1.0:
        print(f"  Already at {target_sfreq} Hz - skipping resample.")
        return raw.copy()
    print(f"  Resampling {raw.info['sfreq']:.0f} -> {target_sfreq:.0f} Hz ...")
    return raw.copy().resample(target_sfreq, npad="auto", verbose=False)


def bandpass_raw(raw: mne.io.BaseRaw, hp: float = FILTER_HP,
                 lp: float = FILTER_LP) -> mne.io.BaseRaw:
    """
    Band-pass EEG channels only (the ECG channel is left untouched so fMRIB
    QRS detection still sees the raw cardiac waveform). FIR, zero-phase.
    lp is clamped just under Nyquist as a safety guard.
    """
    nyq = raw.info["sfreq"] / 2.0
    lp_eff = min(lp, nyq - 5.0)
    print(f"  Band-pass EEG {hp:.1f}-{lp_eff:.0f} Hz (Nyquist {nyq:.0f} Hz), ECG left raw ...")
    out = raw.copy().filter(
        l_freq=hp, h_freq=lp_eff,
        picks="eeg", phase="zero", fir_design="firwin", verbose=False)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 03: BCG via MATLAB fMRIB plugin (QRS detect + OBS sweep in one session)
# ─────────────────────────────────────────────────────────────────────────────

def run_fmrib_bcg_matlab(data_all: np.ndarray, ch_names: list[str], ecg_idx: int,
                         sfreq: float, npc_grid: list[int], work_dir: Path):
    """
    Run the EEGLAB fMRIB plugin once: detect QRS, then OBS-clean the data for
    every npc in npc_grid. Returns (qrs_latencies_0based, {npc: cleaned_mat_path}).

    Cleaned data for each npc is saved to work_dir/bcg_npc{N}.mat (var 'clean',
    float32, all channels) so Python can score each and keep only the winner.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    in_mat = work_dir / "bcg_input.mat"
    qrs_mat = work_dir / "bcg_qrs.mat"
    m_file = work_dir / "bcg_fmrib.m"

    sio.savemat(in_mat, {
        "data": data_all.astype(np.float32),
        "srate": float(sfreq),
        "labels": np.array(ch_names, dtype=object),
    }, do_compression=True)

    npc_list_ml = "[" + " ".join(str(n) for n in npc_grid) + "]"
    out_pattern = str(work_dir / "bcg_npc")   # -> bcg_npc%d.mat

    lines = [
        f"addpath('{EEGLAB_DIR.resolve()}');",
        f"addpath('{FMRIB_DIR.resolve()}');",
        "eeglab nogui;",
        f"S = load('{in_mat.resolve()}');",
        "EEG = eeg_emptyset();",
        "EEG.data = double(S.data);",
        "EEG.srate = double(S.srate);",
        "EEG.nbchan = size(S.data,1);",
        "EEG.pnts = size(S.data,2);",
        "EEG.trials = 1;",
        "EEG.xmin = 0;",
        "EEG.xmax = (EEG.pnts-1)/EEG.srate;",
        "EEG.chanlocs = struct([]);",
        "for i = 1:EEG.nbchan",
        "  if iscell(S.labels)",
        "    EEG.chanlocs(i).labels = char(S.labels{i});",
        "  else",
        "    EEG.chanlocs(i).labels = deblank(S.labels(i,:));",
        "  end",
        "end",
        "EEG = eeg_checkset(EEG);",
        f"ecgchan = {ecg_idx + 1};",   # 1-based for MATLAB
        "fprintf('Detecting QRS on channel %d...\\n', ecgchan);",
        "EEG = pop_fmrib_qrsdetect(EEG, ecgchan, 'qrs', 'no');",
        "qrs_lat = [];",
        "for i = 1:length(EEG.event)",
        "  if strcmp(EEG.event(i).type, 'qrs')",
        "    qrs_lat(end+1) = EEG.event(i).latency;",   # 1-based sample latency
        "  end",
        "end",
        f"save('{qrs_mat.resolve()}', 'qrs_lat', '-v7');",
        "fprintf('QRS events: %d\\n', numel(qrs_lat));",
        f"npc_list = {npc_list_ml};",
        "for k = 1:numel(npc_list)",
        "  npc = npc_list(k);",
        "  fprintf('OBS with npc=%d...\\n', npc);",
        "  EEGc = pop_fmrib_pas(EEG, 'qrs', 'obs', npc);",
        "  clean = single(EEGc.data);",
        f"  fname = sprintf('{out_pattern}%d.mat', npc);",
        "  save(fname, 'clean', '-v7');",
        "  clear EEGc clean;",
        "end",
        "exit(0);",
    ]
    m_file.write_text("\n".join(lines) + "\n", encoding="ascii")

    print(f"  Launching MATLAB fMRIB (QRS + OBS sweep npc={npc_grid})...")
    res = subprocess.run(
        [str(MATLAB_BIN), "-nodesktop", "-nosplash", "-batch", f"run('{m_file.resolve()}')"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if res.returncode != 0:
        tail = "\n".join(res.stdout.splitlines()[-30:])
        raise RuntimeError(f"MATLAB fMRIB failed (code {res.returncode}):\n{tail}")

    if not qrs_mat.exists():
        raise RuntimeError("MATLAB did not produce QRS output.")
    qrs = np.array(sio.loadmat(qrs_mat)["qrs_lat"], dtype=float).flatten()
    qrs = np.round(qrs).astype(int) - 1        # 1-based -> 0-based
    qrs = qrs[(qrs >= 0) & (qrs < data_all.shape[1])]

    cleaned_paths = {}
    for npc in npc_grid:
        p = work_dir / f"bcg_npc{npc}.mat"
        if p.exists():
            cleaned_paths[npc] = p

    in_mat.unlink(missing_ok=True)
    m_file.unlink(missing_ok=True)
    qrs_mat.unlink(missing_ok=True)
    return qrs, cleaned_paths


# ─────────────────────────────────────────────────────────────────────────────
# Quality metric: BCG Suppression Index (BSI)
# ─────────────────────────────────────────────────────────────────────────────

def _bandpower(f, psd, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(np.trapz(psd[m], f[m])) if np.any(m) else 0.0


def compute_bcg_metrics(before: np.ndarray, after: np.ndarray, ecg: np.ndarray,
                        sfreq: float, ch_names: list[str], eval_chs: list[str]) -> dict:
    """before/after: EEG-only arrays with the SAME channel order as ch_names."""
    nperseg = int(min(4 * sfreq, before.shape[1]))
    idx = [ch_names.index(c) for c in eval_chs if c in ch_names]

    card_b = card_a = alpha_b = alpha_a = 0.0
    for i in idx:
        f, pb = welch(before[i], sfreq, nperseg=nperseg)
        _, pa = welch(after[i],  sfreq, nperseg=nperseg)
        card_b  += _bandpower(f, pb, *CARDIAC_BAND)
        card_a  += _bandpower(f, pa, *CARDIAC_BAND)
        alpha_b += _bandpower(f, pb, *ALPHA_BAND)
        alpha_a += _bandpower(f, pa, *ALPHA_BAND)

    cardiac_suppression = float(np.clip(1.0 - card_a / max(card_b, 1e-20), 0.0, 1.0))
    alpha_retention = float(np.clip(alpha_a / max(alpha_b, 1e-20), 0.0, 1.5))

    def mean_abs_corr(dat):
        cs = []
        e = ecg - ecg.mean()
        for i in idx:
            x = dat[i] - dat[i].mean()
            denom = np.std(x) * np.std(e)
            if denom > 1e-20:
                cs.append(abs(np.dot(x, e) / (len(x) * denom)))
        return float(np.mean(cs)) if cs else 0.0

    corr_before = mean_abs_corr(before)
    corr_after  = mean_abs_corr(after)

    alpha_penalty = max(0.0, 1.0 - alpha_retention)      # only penalise LOSS
    corr_drop = max(0.0, corr_before - corr_after) / max(corr_before, 1e-6)
    bsi = float(0.6 * cardiac_suppression + 0.4 * corr_drop - 0.5 * alpha_penalty)

    return {
        "cardiac_suppression": cardiac_suppression,
        "alpha_retention": alpha_retention,
        "ecg_corr_before": corr_before,
        "ecg_corr_after": corr_after,
        "corr_drop": corr_drop,
        "bsi": bsi,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_rpeaks(ecg, r_peaks, sfreq, out_png: Path):
    t = np.arange(len(ecg)) / sfreq
    mid = len(ecg) // 2
    w = int(10 * sfreq)
    s, e = max(0, mid - w // 2), min(len(ecg), mid + w // 2)
    pk = r_peaks[(r_peaks >= s) & (r_peaks < e)]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(t[s:e], ecg[s:e], color="#2d3436", lw=0.7, label="ECG")
    ax.plot(t[pk], ecg[pk], "rx", ms=7, label=f"R-peaks (n={len(r_peaks)} total)")
    hr = 60.0 / np.median(np.diff(r_peaks) / sfreq) if len(r_peaks) > 1 else 0
    ax.set_title(f"fMRIB QRS detection (median HR ~ {hr:.0f} bpm)", fontweight="bold")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("ECG (a.u.)")
    ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def plot_spectra(before, after, sfreq, ch_names, eval_chs, out_png: Path):
    idx = [(c, ch_names.index(c)) for c in eval_chs if c in ch_names]
    n = len(idx); ncol = 2; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.6 * nrow), squeeze=False)
    nperseg = int(min(4 * sfreq, before.shape[1]))
    for k, (name, i) in enumerate(idx):
        ax = axes[k // ncol][k % ncol]
        f, pb = welch(before[i], sfreq, nperseg=nperseg)
        _, pa = welch(after[i],  sfreq, nperseg=nperseg)
        m = f <= min(FILTER_LP + 5, sfreq / 2)
        ax.semilogy(f[m], pb[m], color="#e17055", lw=0.9, label="Before BCG")
        ax.semilogy(f[m], pa[m], color="#00b894", lw=1.1, label="After BCG")
        ax.axvspan(*CARDIAC_BAND, color="#d63031", alpha=0.08)
        ax.axvspan(*ALPHA_BAND, color="#0984e3", alpha=0.08)
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlim(0, min(FILTER_LP + 5, sfreq / 2)); ax.grid(True, alpha=0.3, which="both")
        if k == 0:
            ax.legend(fontsize=7, loc="upper right")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("PSD before vs after BCG (red=cardiac 0.7-4 Hz, blue=alpha 8-13 Hz)",
                 fontweight="bold", y=1.0)
    fig.tight_layout(); fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_npc_sweep(sweep: dict, best_npc: int, out_png: Path):
    """sweep: {npc: metrics_dict}."""
    npcs = sorted(sweep)
    bsi   = [sweep[n]["bsi"] for n in npcs]
    card  = [sweep[n]["cardiac_suppression"] * 100 for n in npcs]
    alpha = [sweep[n]["alpha_retention"] * 100 for n in npcs]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(npcs, bsi, "-o", color="#0984e3", lw=2, label="BSI (composite)")
    ax.axvline(best_npc, color="#d63031", ls="--", lw=2, label=f"best npc={best_npc}")
    ax.set_xlabel("npc (OBS basis components)"); ax.set_ylabel("BSI")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(npcs, card, "-s", color="#00b894", alpha=0.6, label="Cardiac suppr. %")
    ax2.plot(npcs, alpha, "-^", color="#fdcb6e", alpha=0.8, label="Alpha retention %")
    ax2.set_ylabel("%")
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, fontsize=8, loc="lower right")
    ax.set_title("npc sweep: BCG suppression vs alpha preservation", fontweight="bold")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────────────

def _b64(path: Path) -> str:
    import base64
    return base64.b64encode(path.read_bytes()).decode() if path.exists() else ""


def _status(cardiac_suppression, alpha_flag):
    """Grade BCG by how well it did its job (cardiac suppression). A low alpha
    retention downgrades to REVIEW, since the alpha/residual split is deferred
    to the ICA stage and a human may want to glance first."""
    if alpha_flag == "REVIEW":
        return ("REVIEW", "#fdcb6e")
    if cardiac_suppression >= 0.40: return ("EXCELLENT", "#00b894")
    if cardiac_suppression >= 0.25: return ("GOOD", "#0984e3")
    if cardiac_suppression >= 0.10: return ("REVIEW", "#fdcb6e")
    return ("FAIL", "#d63031")


def generate_bcg_html(seg_name, metrics, best_npc, per_channel, sfreq_before,
                      sfreq_after, n_rpeaks, alpha_flag, rpeaks_png, spectra_png,
                      sweep_png, out_html):
    from datetime import datetime
    label, color = _status(metrics["cardiac_suppression"], alpha_flag)
    rows = "".join(
        f"<tr><td><b>{c}</b></td><td>{d['cardiac_suppression']*100:.1f}%</td>"
        f"<td>{d['alpha_retention']*100:.1f}%</td>"
        f"<td>{d['ecg_corr_before']:.3f}</td><td>{d['ecg_corr_after']:.3f}</td></tr>"
        for c, d in per_channel.items()
    )
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    arrow = f"{metrics['ecg_corr_before']:.3f}&rarr;{metrics['ecg_corr_after']:.3f}"
    html = (
        '<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">'
        f'<title>BCG Report - {seg_name}</title><style>'
        'body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f5f6fa;color:#2d3436;margin:0;padding:24px;}'
        '.card{background:#fff;border-radius:12px;padding:20px 26px;margin:0 auto 20px;max-width:1100px;box-shadow:0 2px 10px rgba(0,0,0,.06);}'
        'h1{font-size:22px;} h2{font-size:17px;border-left:4px solid #0984e3;padding-left:10px;}'
        'table{border-collapse:collapse;width:100%;font-size:14px;}'
        'th,td{border:1px solid #dfe6e9;padding:7px 10px;text-align:center;} th{background:#f1f2f6;}'
        f'.badge{{display:inline-block;padding:6px 16px;border-radius:20px;color:#fff;font-weight:700;background:{color};}}'
        '.kpi{display:flex;gap:18px;flex-wrap:wrap;} .kpi div{flex:1;min-width:150px;background:#f8f9fa;border-radius:10px;padding:14px;text-align:center;}'
        '.kpi b{display:block;font-size:26px;color:#0984e3;} img{max-width:100%;border-radius:8px;}'
        '</style></head><body>'
        f'<div class="card"><h1>&#129728; BCG Artifact Removal Report - {seg_name}</h1>'
        f'<p>Generated: {dt} | Resample: {sfreq_before:.0f} &rarr; {sfreq_after:.0f} Hz | '
        f'Band-pass: {FILTER_HP:.1f}-{FILTER_LP:.0f} Hz | '
        f'EEGLAB fMRIB OBS | R-peaks: {n_rpeaks}</p>'
        f'<p>Overall cleaning grade: <span class="badge">{label}</span> '
        f'(alpha flag: {alpha_flag})</p></div>'

        '<div class="card"><h2>&#127919; Key metrics</h2><div class="kpi">'
        f'<div><b>{metrics["cardiac_suppression"]*100:.1f}%</b>Cardiac suppr. 0.7-4 Hz <i>(primary)</i></div>'
        f'<div><b>{arrow}</b>|corr| with ECG</div>'
        f'<div><b>{metrics["alpha_retention"]*100:.1f}%</b>Alpha retention 8-13 Hz <i>(diagnostic)</i></div>'
        f'<div><b>{metrics["bsi"]:.3f}</b>BSI (composite, ref)</div></div>'
        '<p style="font-size:13px;color:#636e72;margin-top:14px;">'
        'BCG is graded by <b>cardiac-band suppression</b> - its actual job in the '
        'OBS&rarr;ICA pipeline. <b>Alpha retention</b> is shown as a diagnostic only: '
        'it cannot cleanly separate "removed real alpha" from "removed BCG power that '
        'overlapped 8-13 Hz", so the fine alpha/residual split is left to the later '
        'ICA / ICLabel stage. A low alpha value raises a REVIEW flag, not a failure.</p></div>'

        '<div class="card"><h2>&#9881; Optimal OBS parameter (npc sweep)</h2><div class="kpi">'
        f'<div><b>{best_npc}</b>npc (basis components)</div></div>'
        f'<img src="data:image/png;base64,{_b64(sweep_png)}"></div>'

        '<div class="card"><h2>&#128147; ECG R-peak detection (fMRIB)</h2>'
        f'<img src="data:image/png;base64,{_b64(rpeaks_png)}"></div>'

        '<div class="card"><h2>&#128202; Spectra before/after BCG</h2>'
        f'<img src="data:image/png;base64,{_b64(spectra_png)}"></div>'

        '<div class="card"><h2>&#128200; Per-channel quality</h2>'
        '<table><tr><th>Channel</th><th>Cardiac suppr.</th><th>Alpha retention</th>'
        '<th>|corr| ECG (before)</th><th>|corr| ECG (after)</th></tr>'
        f'{rows}</table></div>'
        '</body></html>'
    )
    out_html.write_text(html, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_bcg_pipeline(segment_dir: Path = DEFAULT_SEGMENT_DIR, npc_grid=None):
    segment_dir = Path(segment_dir).resolve()
    seg_name = segment_dir.name
    npc_grid = npc_grid or NPC_GRID
    print("=" * 75)
    print(f"[STEP 08 - BCG] Resample 250 Hz + fMRIB OBS for: {seg_name}")
    print("=" * 75)

    subject_deriv = DATA_ROOT / DEFAULT_EXPERIMENT / "derivatives"
    resamp_dir = subject_deriv / "02_resampled250" / seg_name
    bcg_dir    = subject_deriv / "03_bcg" / seg_name
    qc_dir     = PROJECT_ROOT / "qc" / DEFAULT_EXPERIMENT / "bcg"
    for d in (resamp_dir, bcg_dir, qc_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load Bergen-cleaned .set
    set_path = _find_bergen_set(segment_dir)
    print(f"  Loading Bergen-clean: {set_path.name} ({set_path.stat().st_size/1e6:.0f} MB)")
    raw = _load_set_as_raw(set_path)
    print(f"  {len(raw.ch_names)} channels @ {raw.info['sfreq']:.0f} Hz, {raw.n_times} samples")

    # 2. Resample -> 250 Hz, then band-pass EEG (1-100 Hz). The high-pass is
    #    essential: without it sub-1 Hz drift dominates the low band, OBS smears
    #    its subtraction across 0.7-15 Hz (eating alpha) and the metric is noisy.
    raw_rs = resample_raw(raw, TARGET_SFREQ)
    del raw; gc.collect()
    raw_rs = bandpass_raw(raw_rs, FILTER_HP, FILTER_LP)
    resamp_fif = resamp_dir / f"{seg_name}_250hz.fif"
    raw_rs.save(resamp_fif, overwrite=True, verbose=False)
    print(f"  Saved resampled+filtered: {resamp_fif}")

    sfreq = float(raw_rs.info["sfreq"])
    ch_names = list(raw_rs.ch_names)
    if ECG_CH not in ch_names:
        raise RuntimeError(f"ECG channel '{ECG_CH}' not found. Channels: {ch_names[:35]}...")

    all_data = raw_rs.get_data() * 1e6                 # V -> uV for the plugin
    ecg_idx = ch_names.index(ECG_CH)
    ecg = all_data[ecg_idx].copy()
    eeg_mask = np.ones(len(ch_names), dtype=bool); eeg_mask[ecg_idx] = False
    eeg_names = [c for i, c in enumerate(ch_names) if eeg_mask[i]]
    before_eeg = all_data[eeg_mask]

    # 3. MATLAB fMRIB: QRS detect + OBS sweep over npc (one session)
    work_dir = bcg_dir / "_matlab_tmp"
    r_peaks, cleaned_paths = run_fmrib_bcg_matlab(
        all_data, ch_names, ecg_idx, sfreq, npc_grid, work_dir)
    print(f"  QRS peaks: {len(r_peaks)} "
          f"(median HR ~ {60.0/np.median(np.diff(r_peaks)/sfreq):.0f} bpm)"
          if len(r_peaks) > 1 else f"  QRS peaks: {len(r_peaks)}")
    plot_rpeaks(ecg, r_peaks, sfreq, qc_dir / f"{seg_name}_rpeaks.png")

    if not cleaned_paths:
        raise RuntimeError("fMRIB produced no cleaned output for any npc.")

    # 4. Score each npc on pilot channels -> pick best
    eval_chs = [c for c in EVAL_CHANNELS if c in eeg_names]
    pilot = [c for c in PILOT_CHANNELS if c in eeg_names]
    sweep = {}
    for npc in sorted(cleaned_paths):
        clean_all = np.array(sio.loadmat(cleaned_paths[npc])["clean"], dtype=np.float64)
        after_eeg = clean_all[eeg_mask]
        m = compute_bcg_metrics(before_eeg, after_eeg, ecg, sfreq, eeg_names, pilot)
        sweep[npc] = m
        print(f"    npc={npc}: BSI={m['bsi']:.4f} "
              f"cardiac={m['cardiac_suppression']*100:.1f}% alpha={m['alpha_retention']*100:.1f}%")
        del clean_all, after_eeg; gc.collect()

    # Selection: strongest cardiac suppression (OBS -> ICA architecture). Alpha
    # retention is reported only, and used to raise a soft REVIEW flag when it
    # drops low enough that a human/QC glance is worth it before ICA.
    best_npc = max(sweep, key=lambda n: sweep[n]["cardiac_suppression"])
    alpha_flag = "OK" if sweep[best_npc]["alpha_retention"] >= ALPHA_WARN_MIN else "REVIEW"
    if alpha_flag == "REVIEW":
        print(f"  [NOTE] alpha retention {sweep[best_npc]['alpha_retention']*100:.1f}% "
              f"< {ALPHA_WARN_MIN*100:.0f}% -> flagged REVIEW (residual/alpha split left to ICA).")
    print(f"  Selected npc = {best_npc} "
          f"(cardiac={sweep[best_npc]['cardiac_suppression']*100:.1f}%, "
          f"alpha={sweep[best_npc]['alpha_retention']*100:.1f}%, flag={alpha_flag})")
    plot_npc_sweep(sweep, best_npc, qc_dir / f"{seg_name}_npc_sweep.png")

    # 5. Load winning cleaned data, compute full + per-channel metrics
    clean_all = np.array(sio.loadmat(cleaned_paths[best_npc])["clean"], dtype=np.float64)
    after_eeg = clean_all[eeg_mask]
    metrics = compute_bcg_metrics(before_eeg, after_eeg, ecg, sfreq, eeg_names, eval_chs)
    per_channel = {c: compute_bcg_metrics(before_eeg, after_eeg, ecg, sfreq, eeg_names, [c])
                   for c in eval_chs}
    print(f"  FINAL BSI={metrics['bsi']:.3f} | cardiac={metrics['cardiac_suppression']*100:.1f}% "
          f"| alpha={metrics['alpha_retention']*100:.1f}% "
          f"| corr {metrics['ecg_corr_before']:.3f}->{metrics['ecg_corr_after']:.3f}")

    plot_spectra(before_eeg, after_eeg, sfreq, eeg_names, eval_chs,
                 qc_dir / f"{seg_name}_spectra.png")

    # 6. Reassemble full dataset (ECG restored untouched) and save
    out_full = clean_all.copy()
    out_full[ecg_idx] = ecg
    raw_clean = mne.io.RawArray(out_full * 1e-6, raw_rs.info, verbose=False)  # uV -> V
    bcg_fif = bcg_dir / f"{seg_name}_bcg_clean.fif"
    raw_clean.save(bcg_fif, overwrite=True, verbose=False)
    print(f"  Saved BCG-clean: {bcg_fif}")

    # 7. Metrics JSON + HTML report
    metrics_out = {**metrics, "best_npc": int(best_npc), "alpha_flag": alpha_flag,
                   "alpha_warn_min": ALPHA_WARN_MIN, "n_rpeaks": int(len(r_peaks)),
                   "sfreq": sfreq, "npc_sweep": {int(k): v for k, v in sweep.items()}}
    (bcg_dir / f"{seg_name}_bcg_metrics.json").write_text(
        json.dumps(metrics_out, indent=2), encoding="utf-8")
    out_html = bcg_dir / f"{seg_name}_bcg_report.html"
    generate_bcg_html(seg_name, metrics, best_npc, per_channel, TARGET_SFREQ, sfreq,
                      len(r_peaks), alpha_flag, qc_dir / f"{seg_name}_rpeaks.png",
                      qc_dir / f"{seg_name}_spectra.png",
                      qc_dir / f"{seg_name}_npc_sweep.png", out_html)
    print(f"  HTML report: {out_html}")

    # 8. Cleanup temp MATLAB outputs
    for p in cleaned_paths.values():
        p.unlink(missing_ok=True)
    try:
        work_dir.rmdir()
    except OSError:
        pass

    print("=" * 75)
    print("  [STEP 08 - BCG] DONE.")
    print("=" * 75)
    return out_html


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="BCG removal (EEGLAB fMRIB OBS) + resample 250 Hz")
    ap.add_argument("--segment-dir", type=Path, default=DEFAULT_SEGMENT_DIR)
    ap.add_argument("--npc", type=int, nargs="+", default=None, help="npc grid to sweep")
    args = ap.parse_args()
    run_bcg_pipeline(args.segment_dir.resolve(), npc_grid=args.npc)
