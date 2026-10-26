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
                         EEGLAB_DIR, DEFAULT_SEGMENT_DIR, ALPHA_BAND, EVAL_CHANNELS,
                         TARGET_SFREQ, FILTER_HP, FILTER_LP, BCG_ALPHA_GATE_MIN,
                         BRAINBEATS_DIR)
except ImportError:
    from config import (PROJECT_ROOT, DATA_ROOT, DEFAULT_EXPERIMENT, MATLAB_BIN,
                        EEGLAB_DIR, DEFAULT_SEGMENT_DIR, ALPHA_BAND, EVAL_CHANNELS,
                        TARGET_SFREQ, FILTER_HP, FILTER_LP, BCG_ALPHA_GATE_MIN,
                        BRAINBEATS_DIR)

ECG_CH       = "ECG"          # name of cardiac channel in the .set
CARDIAC_BAND = (0.7, 4.0)     # Hz - fundamental + harmonics of BCG
FILTER_NOTCH = 50.0           # Hz - Notch filter for 50 Hz powerline interference
NPC_GRID     = [1, 2, 3, 4, 5, 6, 7, 8]   # OBS basis components to sweep (incl. gentle 1-2)
FMRIB_DIR    = EEGLAB_DIR / "plugins" / "fMRIb2.1"

# npc SELECTION POLICY (OBS -> ICA architecture):
ALPHA_WARN_MIN = 0.60   # below this, flag REVIEW for a human/QC glance
ALPHA_GATE_MIN = BCG_ALPHA_GATE_MIN
MIN_CARDIAC_SUPPR = 0.20 # minimum acceptable cardiac suppression (20%)


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
        subject_id = segment_dir.parent.parent.name if segment_dir.parent.name == "segments" else segment_dir.parent.name
        deriv = DATA_ROOT / subject_id / "derivatives" / "01_bergen" / segment_dir.name
        if deriv.exists():
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
                 lp: float = FILTER_LP, notch: float | None = None) -> mne.io.BaseRaw:
    """
    Band-pass EEG channels only (the ECG channel is left untouched so fMRIB
    QRS detection still sees the raw cardiac waveform). FIR, zero-phase.
    lp is clamped just under Nyquist as a safety guard.
    Applies Notch filter (default 50 Hz, 49-51 Hz) to EEG channels if notch < Nyquist.
    """
    if notch is None:
        notch = FILTER_NOTCH
    nyq = raw.info["sfreq"] / 2.0
    lp_eff = min(lp, nyq - 5.0)
    print(f"  Band-pass EEG {hp:.1f}-{lp_eff:.0f} Hz (Nyquist {nyq:.0f} Hz), ECG left raw ...")
    out = raw.copy().filter(
        l_freq=hp, h_freq=lp_eff,
        picks="eeg", phase="zero", fir_design="firwin", verbose=False)
    if notch is not None and 0 < notch < nyq:
        lo, hi = notch - 1.0, notch + 1.0
        print(f"  Applying Notch filter {notch:.1f} Hz ({lo:.0f}-{hi:.0f} Hz) to EEG channels ...")
        # notch_widths=2.0 -> stop-band 49-51 Hz (MNE default freq/200 would be only 0.25 Hz)
        out.notch_filter(freqs=[notch], picks="eeg", notch_widths=2.0,
                         fir_design="firwin", verbose=False)
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
    
    sio.savemat(in_mat, {
        "data": data_all.astype(np.float32),
        "srate": float(sfreq),
        "labels": np.array(ch_names, dtype=object),
    }, do_compression=True)

    out_pattern = str(work_dir / "bcg_npc")   # -> bcg_npc%d.mat

    bb_funcs = BRAINBEATS_DIR / "functions"
    base_ml = [
        f"addpath('{EEGLAB_DIR.resolve()}');",
        f"addpath('{FMRIB_DIR.resolve()}');",
    ]
    if bb_funcs.exists():
        base_ml.append(f"addpath('{bb_funcs.resolve()}');")
    base_ml += [
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
        "EEG = eeg_checkset(EEG);"
    ]

    # --- Step 1: Detect QRS and save ---
    m_qrs = work_dir / "run_qrs.m"
    lines_qrs = base_ml + [
        f"ecgchan = {ecg_idx + 1};",
        "if exist('get_RR', 'file') == 2",
        "  fprintf('Detecting QRS via BrainBeats (Pan-Tompkins on ECG)...\\n');",
        "  ecg_sig = double(EEG.data(ecgchan, :));",
        "  times_ms = (0:(EEG.pnts-1)) * (1000.0 / EEG.srate);",
        "  % Outlier clipping of residual fMRI gradient spikes on ECG",
        "  med_v = median(ecg_sig);",
        "  mad_v = median(abs(ecg_sig - med_v));",
        "  if mad_v > 0",
        "    clip_lim = 10 * mad_v;",
        "    ecg_clean = max(min(ecg_sig, med_v + clip_lim), med_v - clip_lim);",
        "  else",
        "    ecg_clean = ecg_sig;",
        "  end",
        "  % Adaptive Pan-Tompkins tuning to reach physiological resting HR (45-125 bpm)",
        "  best_peaks = [];",
        "  best_diff_hr = 999;",
        "  target_hr = 72.0;",
        "  for pthresh = [0.16, 0.20, 0.24, 0.30, 0.35]",
        "    bb_params.fs = EEG.srate;",
        "    bb_params.heart_signal = 'ecg';",
        "    bb_params.ecg_bandpass = [4 25];",
        "    bb_params.ecg_peakthresh = pthresh;",
        "    bb_params.ecg_searchback = true;",
        "    bb_params.ecg_refperiod = 0.40; % 400ms prevents T-wave false positives",
        "    try",
        "      [RR, ~, peaks] = get_RR(ecg_clean, times_ms, bb_params);",
        "      if ~isempty(RR)",
        "        med_hr = 60.0 / median(RR);",
        "        if abs(med_hr - target_hr) < best_diff_hr && med_hr >= 45 && med_hr <= 135",
        "          best_diff_hr = abs(med_hr - target_hr);",
        "          best_peaks = peaks;",
        "        end",
        "      end",
        "    catch",
        "    end",
        "  end",
        "  if isempty(best_peaks) || numel(best_peaks) < 100",
        "    fprintf('BrainBeats found only %d peaks; falling back to pop_fmrib_qrsdetect...\\n', numel(best_peaks));",
        "    EEG_fm = pop_fmrib_qrsdetect(EEG, ecgchan, 'qrs', 'no');",
        "    fm_peaks = [];",
        "    for i = 1:length(EEG_fm.event)",
        "      if strcmp(EEG_fm.event(i).type, 'qrs')",
        "        fm_peaks(end+1) = EEG_fm.event(i).latency;",
        "      end",
        "    end",
        "    if numel(fm_peaks) > numel(best_peaks)",
        "      best_peaks = fm_peaks;",
        "    end",
        "  end",
        "  qrs_lat = double(best_peaks(:));",
        "else",
        "  fprintf('Detecting QRS via pop_fmrib_qrsdetect on channel %d...\\n', ecgchan);",
        "  EEG = pop_fmrib_qrsdetect(EEG, ecgchan, 'qrs', 'no');",
        "  qrs_lat = [];",
        "  for i = 1:length(EEG.event)",
        "    if strcmp(EEG.event(i).type, 'qrs')",
        "      qrs_lat(end+1) = EEG.event(i).latency;",
        "    end",
        "  end",
        "end",
        f"save('{qrs_mat.resolve()}', 'qrs_lat', '-v7');",
        "fprintf('Raw QRS events: %d\\n', numel(qrs_lat));",
        "exit(0);"
    ]
    m_qrs.write_text("\n".join(lines_qrs) + "\n", encoding="ascii")
    
    print(f"  Launching MATLAB QRS detection (BrainBeats / fMRIB)...")
    res = None
    try:
        res = subprocess.run(
            [str(MATLAB_BIN), "-nodisplay", "-nodesktop", "-nosplash", "-batch", f"run('{m_qrs.resolve()}')"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240
        )
    except subprocess.TimeoutExpired as tex:
        print(f"    [WARN] MATLAB QRS timed out after 240s; checking if output was written...")
        res = None

    if not qrs_mat.exists() or qrs_mat.stat().st_size < 100:
        tail = "\n".join((res.stdout if res else "").splitlines()[-30:])
        raise RuntimeError(f"MATLAB QRS detection failed to produce valid output (code {res.returncode if res else 'timeout'}):\n{tail}")
        
    qrs = np.array(sio.loadmat(qrs_mat)["qrs_lat"], dtype=float).flatten()
    qrs = np.round(qrs).astype(int) - 1        # 1-based -> 0-based
    qrs = qrs[(qrs >= 0) & (qrs < data_all.shape[1])]

    # Physiological deduplication and refractory check (min RR = 0.38s -> max HR = ~158 bpm)
    qrs = np.sort(np.unique(qrs))
    if len(qrs) > 1:
        min_dist = int(0.38 * sfreq)
        cleaned_qrs = [qrs[0]]
        for p in qrs[1:]:
            if p - cleaned_qrs[-1] >= min_dist:
                cleaned_qrs.append(p)
        qrs = np.array(cleaned_qrs, dtype=int)
    # Re-save 1-based sanitized QRS coordinates for all OBS sessions
    sio.savemat(qrs_mat, {"qrs_lat": (qrs + 1).reshape(-1, 1)})
    med_rr = float(np.median(np.diff(qrs)) / sfreq) if len(qrs) > 1 else 1.0
    print(f"  Valid QRS peaks: {len(qrs)} (median RR: {med_rr:.3f} s, ~{60.0/med_rr:.1f} bpm)")

    # --- Step 2: Loop over npc in separate MATLAB sessions ---
    cleaned_paths = {}
    for npc in npc_grid:
        m_obs = work_dir / f"run_obs_{npc}.m"
        fname = f"{out_pattern}{npc}.mat"
        lines_obs = base_ml + [
            f"load('{qrs_mat.resolve()}', 'qrs_lat');",
            "for i = 1:length(qrs_lat)",
            "  EEG.event(end+1).type = 'qrs';",
            "  EEG.event(end).latency = qrs_lat(i);",
            "end",
            "EEG = eeg_checkset(EEG);",
            f"fprintf('OBS with npc={npc}...\\n');",
            f"EEGc = pop_fmrib_pas(EEG, 'qrs', 'obs', {npc});",
            "clean = single(EEGc.data);",
            f"save('{fname}', 'clean', '-v7');",
            "exit(0);"
        ]
        m_obs.write_text("\n".join(lines_obs) + "\n", encoding="ascii")
        print(f"    Launching MATLAB fMRIB OBS (npc={npc})...")
        res_obs = None
        try:
            res_obs = subprocess.run(
                [str(MATLAB_BIN), "-nodisplay", "-nodesktop", "-nosplash", "-batch", f"run('{m_obs.resolve()}')"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300
            )
        except subprocess.TimeoutExpired:
            print(f"      [WARN] MATLAB fMRIB OBS timed out after 300s for npc={npc}")
            res_obs = None

        p = Path(fname)
        if p.exists() and p.stat().st_size > 1000:
            cleaned_paths[npc] = p
        else:
            tail = "\n".join((res_obs.stdout if res_obs else "").splitlines()[-30:])
            print(f"      [WARN] MATLAB fMRIB OBS failed for npc={npc}:\n{tail}")
        m_obs.unlink(missing_ok=True)

    in_mat.unlink(missing_ok=True)
    m_qrs.unlink(missing_ok=True)
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
        # Plot up to 100 Hz (or Nyquist) so the low-pass roll-off at FILTER_LP
        # is visible as QC: the helium-pump comb above 45 Hz should be gone.
        fmax_plot = min(100.0, sfreq / 2)
        m = f <= fmax_plot
        ax.semilogy(f[m], pb[m], color="#e17055", lw=0.9, label="Before BCG")
        ax.semilogy(f[m], pa[m], color="#00b894", lw=1.1, label="After BCG")
        ax.axvline(FILTER_LP, color="gray", ls="--", alpha=0.5, lw=0.8)
        ax.axvspan(*CARDIAC_BAND, color="#d63031", alpha=0.08)
        ax.axvspan(*ALPHA_BAND, color="#0984e3", alpha=0.08)
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlim(0, fmax_plot); ax.grid(True, alpha=0.3, which="both")
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

def run_bcg_pipeline(segment_dir: Path = DEFAULT_SEGMENT_DIR, npc_grid=None, force: bool = False,
                     input_set: Path | None = None):
    """input_set: explicit Bergen-cleaned .set to use (e.g. <seg>_bergen_250hz.set from
    cleanP01). If None, the newest *bergen*.set in segment_dir is used."""
    segment_dir = Path(segment_dir).resolve()
    seg_name = segment_dir.name
    subject_id = segment_dir.parent.parent.name if segment_dir.parent.name == "segments" else segment_dir.parent.name
    npc_grid = npc_grid or NPC_GRID
    print("=" * 75)
    print(f"[STEP 08 - BCG] Resample 250 Hz + fMRIB OBS for: {subject_id}/{seg_name}")
    print("=" * 75)

    subject_deriv = DATA_ROOT / subject_id / "derivatives"
    resamp_dir = subject_deriv / "02_resampled250" / seg_name
    bcg_dir    = subject_deriv / "03_bcg" / seg_name
    qc_dir     = PROJECT_ROOT / "qc" / subject_id / "bcg"

    bcg_fif = bcg_dir / f"{seg_name}_bcg_clean.fif"
    out_html = bcg_dir / f"{seg_name}_bcg_report.html"
    if not force and bcg_fif.exists() and out_html.exists():
        print(f"  [SKIP] BCG-clean output already exists: {bcg_fif.name} (use --force to recompute)")
        print("=" * 75)
        print("  [STEP 08 - BCG] ALREADY DONE.")
        print("=" * 75)
        return out_html

    for d in (resamp_dir, bcg_dir, qc_dir):
        d.mkdir(parents=True, exist_ok=True)

    resamp_fif = resamp_dir / f"{seg_name}_250hz.fif"
    subject_id = segment_dir.parent.parent.name if segment_dir.parent.name == "segments" else segment_dir.parent.name
    bergen_fif = DATA_ROOT / subject_id / "derivatives" / "01_bergen_250hz" / seg_name / f"{seg_name}_bergen_250hz_eeg.fif"

    # NOTE: resamp_fif is already band-passed. With force=True we must rebuild it
    # from the unfiltered Bergen source, NOT re-filter the stored file (that would
    # stack another HP/LP pass on every forced re-run and erode low-freq/alpha power).
    if resamp_fif.exists() and not force:
        print(f"  Found pre-existing resampled 250 Hz data: {resamp_fif.name}, loading directly...")
        raw_rs = mne.io.read_raw_fif(resamp_fif, preload=True, verbose=False)
    elif bergen_fif.exists():
        print(f"  Found 250 Hz Bergen data: {bergen_fif.name}, loading...")
        raw_rs = mne.io.read_raw_fif(bergen_fif, preload=True, verbose=False)
        raw_rs = bandpass_raw(raw_rs, FILTER_HP, FILTER_LP)
        raw_rs.save(resamp_fif, overwrite=True, verbose=False)
        print(f"  Saved resampled+filtered: {resamp_fif}")
    elif input_set and Path(input_set).exists():
        set_path = Path(input_set)
        print(f"  Loading input .set: {set_path.name}")
        raw = _load_set_as_raw(set_path)
        raw_rs = resample_raw(raw, TARGET_SFREQ)
        del raw; gc.collect()
        raw_rs = bandpass_raw(raw_rs, FILTER_HP, FILTER_LP)
        raw_rs.save(resamp_fif, overwrite=True, verbose=False)
        print(f"  Saved resampled+filtered: {resamp_fif}")
    else:
        # Fallback to searching 5000 Hz Bergen .set
        set_path = _find_bergen_set(segment_dir)
        print(f"  Loading Bergen-clean: {set_path.name} ({set_path.stat().st_size/1e6:.0f} MB)")
        raw = _load_set_as_raw(set_path)
        raw_rs = resample_raw(raw, TARGET_SFREQ)
        del raw; gc.collect()
        raw_rs = bandpass_raw(raw_rs, FILTER_HP, FILTER_LP)
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
    np.save(bcg_dir / f"{seg_name}_qrs.npy", np.asarray(r_peaks, dtype=np.int64))

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

    # Selection (OBS -> ICA architecture): strongest cardiac suppression AMONG
    # the npc that keep alpha retention >= ALPHA_GATE_MIN. Previously this took
    # the raw max cardiac suppression, which always picked the most aggressive
    # npc and halved the alpha rhythm (e.g. npc=8: cardiac 82%, alpha 53%). BCG
    # is meant to be conservative; residual BCG is left for ICA to remove.
    # 1. Primary: candidates that satisfy both alpha gate and minimum cardiac suppression
    eligible = [n for n in sweep if sweep[n]["alpha_retention"] >= ALPHA_GATE_MIN and sweep[n]["cardiac_suppression"] >= MIN_CARDIAC_SUPPR]
    if eligible:
        best_npc = max(eligible, key=lambda n: sweep[n]["cardiac_suppression"])
    else:
        # 2. If no candidate meets both, check if any candidate meets ALPHA_GATE_MIN with decent suppression (>=15%)
        alpha_eligible = [n for n in sweep if sweep[n]["alpha_retention"] >= ALPHA_GATE_MIN and sweep[n]["cardiac_suppression"] >= 0.15]
        if alpha_eligible:
            best_npc = max(alpha_eligible, key=lambda n: sweep[n]["cardiac_suppression"])
        else:
            # 3. Otherwise (e.g. open eyes / visual tasks where BCG harmonics depress 8-13 Hz power):
            # Select the best composite BSI (balances cardiac suppression, ECG correlation drop, and alpha penalty)
            best_npc = max(sweep, key=lambda n: sweep[n]["bsi"])
            print(f"  [NOTE] Fallback to composite BSI selection: picked npc={best_npc} "
                  f"(BSI={sweep[best_npc]['bsi']:.4f}, cardiac={sweep[best_npc]['cardiac_suppression']*100:.1f}%, "
                  f"alpha={sweep[best_npc]['alpha_retention']*100:.1f}%).")
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
    ap = argparse.ArgumentParser(description="STEP 08: BCG removal (EEGLAB fMRIB OBS) + resample 250 Hz")
    ap.add_argument("--subject", default=None, help="Subject ID (e.g. 1916)")
    ap.add_argument("--segment", default=None, help="Segment name (e.g. ec, drone) or omit for all segments of subject")
    ap.add_argument("--all", action="store_true", help="Process all available subjects and segments")
    ap.add_argument("--npc", type=int, nargs="+", default=None, help="npc grid to sweep")
    ap.add_argument("--force", action="store_true", help="Force recomputation even if outputs exist")
    args = ap.parse_args()

    seg_dirs: list[Path] = []
    if args.subject:
        subj_seg_dir = DATA_ROOT / args.subject / "segments"
        if args.segment:
            target = subj_seg_dir / args.segment
            target.mkdir(parents=True, exist_ok=True)
            seg_dirs.append(target)
        else:
            if subj_seg_dir.exists():
                seg_dirs.extend(sorted(p for p in subj_seg_dir.iterdir() if p.is_dir() and (p / "segment_work_info.json").exists()))
            if not seg_dirs:
                for st in ("02_resampled250", "01_bergen_250hz", "03_bcg"):
                    deriv_st = DATA_ROOT / args.subject / "derivatives" / st
                    if deriv_st.exists():
                        for p in sorted(deriv_st.iterdir()):
                            if p.is_dir():
                                target = subj_seg_dir / p.name
                                target.mkdir(parents=True, exist_ok=True)
                                if target not in seg_dirs:
                                    seg_dirs.append(target)
                        if seg_dirs:
                            break
    elif args.all or (not args.subject and not args.segment):
        for subj_dir in sorted(DATA_ROOT.glob("*")):
            if not subj_dir.is_dir() or subj_dir.name in ("copy.sh", "logs"):
                continue
            subj_seg_dir = subj_dir / "segments"
            sub_found = []
            if subj_seg_dir.exists():
                sub_found.extend(sorted(p for p in subj_seg_dir.iterdir() if p.is_dir() and (p / "segment_work_info.json").exists()))
            if not sub_found:
                for st in ("02_resampled250", "01_bergen_250hz", "03_bcg"):
                    deriv_st = subj_dir / "derivatives" / st
                    if deriv_st.exists():
                        for p in sorted(deriv_st.iterdir()):
                            if p.is_dir():
                                target = subj_seg_dir / p.name
                                target.mkdir(parents=True, exist_ok=True)
                                if target not in sub_found:
                                    sub_found.append(target)
                        if sub_found:
                            break
            seg_dirs.extend(sub_found)

    if not seg_dirs:
        print("[ERROR] No segments found with segment_work_info.json or derivatives. Run previous steps first!")
    else:
        for sdir in seg_dirs:
            run_bcg_pipeline(sdir.resolve(), npc_grid=args.npc, force=args.force)
