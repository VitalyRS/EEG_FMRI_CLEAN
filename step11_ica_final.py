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
import datetime
import shutil
import yaml

try:
    from .config import (MATLAB_BIN, EEGLAB_DIR, PROJECT_ROOT, DEFAULT_SEGMENT_DIR, ALPHA_BAND, EVAL_CHANNELS,
                         FLATLINE_CRIT, CHANNEL_CRIT, LINE_NOISE_CRIT, ICLABEL_THRESH, ASR_BURST_CRIT,
                         ICLABEL_OTHER_BRAIN_MAX)
except ImportError:
    from config import (MATLAB_BIN, EEGLAB_DIR, PROJECT_ROOT, DEFAULT_SEGMENT_DIR, ALPHA_BAND, EVAL_CHANNELS,
                        FLATLINE_CRIT, CHANNEL_CRIT, LINE_NOISE_CRIT, ICLABEL_THRESH, ASR_BURST_CRIT,
                        ICLABEL_OTHER_BRAIN_MAX)

# High-pass filter exclusively for ICA training (Hz).
ICA_HP_HZ = 1.0


DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_EXPERIMENT = "1916"
DIPFIT_DIR = EEGLAB_DIR / "plugins" / "dipfit"
STD_1005 = DIPFIT_DIR / "standard_BEM" / "elec" / "standard_1005.elc"

# Rule-2 cutoff: reject an 'Other'-dominant IC only if P(Brain) < this value.
OTHER_BRAIN_MAX = ICLABEL_OTHER_BRAIN_MAX
# Optional cap on the number of ICA components (35 components optimal for ~95 ch).
ICA_PCA_MAX = 35
# Safety guards against over-cleaning.
GUARD_MIN_ICLABEL_THRESH = 0.65   # -> clamped to 0.80 if lower
GUARD_MAX_CHANNEL_CRIT = 0.70     # -> clamped to 0.70 if higher (higher ChannelCriterion = more aggressive deletion)
GUARD_MIN_BURST_CRIT = 5.0        # -> clamped to 20.0 if lower (lower = more aggressive reconstruction)
# ICLabel backend: "python" (mne-icalabel, recommended) or "matlab"
ICLABEL_BACKEND = "python"
# QRS-locked residual-BCG detector: an IC whose energy is largely explained by the
# QRS-locked average (evoked-energy ratio) is residual ballistocardiogram. Linear
# ECG correlation / ICLabel 'heart' miss these (r<=0.3). 0 disables the rule.
CARDIAC_QRS_RATIO_THRESH = 0.05   # noise level is ~0.002; true BCG ICs were 0.08-0.64
CARDIAC_QRS_MAX_ICS = 4           # never remove more than this many ICs by this rule
CARDIAC_QRS_MAX_BRAIN = 0.80      # keep an IC if ICLabel is very sure it is brain


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


def get_git_info() -> dict:
    info = {
        "branch": "unknown",
        "commit_hash": "unknown",
        "commit_short": "unknown",
        "commit_message": "unknown",
        "commit_date": "unknown",
        "is_dirty": False,
    }
    for cmd, key in [
        (["git", "rev-parse", "--abbrev-ref", "HEAD"], "branch"),
        (["git", "rev-parse", "HEAD"], "commit_hash"),
        (["git", "log", "-1", "--format=%s"], "commit_message"),
        (["git", "log", "-1", "--format=%ci"], "commit_date"),
    ]:
        try:
            res = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            info[key] = res.stdout.strip()
        except Exception:
            pass
    if info["commit_hash"] != "unknown":
        info["commit_short"] = info["commit_hash"][:7]
    try:
        res = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        info["is_dirty"] = bool(res.stdout.strip())
    except Exception:
        info["is_dirty"] = False
    return info


def save_pipeline_info(subject_id: str, seg: str, out_deriv_dir: Path,
                       in_fif: Path, out_fif: Path, metrics: dict,
                       params_used: dict, cfg: dict | None = None) -> Path:
    """Save pipeline_info.yaml containing git info, data provenance and config snapshot."""
    out_deriv_dir = Path(out_deriv_dir).resolve()
    info_path = out_deriv_dir / "pipeline_info.yaml"

    if cfg is None:
        cfg_file = PROJECT_ROOT / "config" / "cleanP02.yaml"
        if cfg_file.exists():
            try:
                cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
            except Exception:
                cfg = {}
        else:
            cfg = {}

    git_info = get_git_info()

    def _rel(p: Path | str | None) -> str | None:
        if not p:
            return None
        p = Path(p).resolve()
        try:
            return str(p.relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            return str(p)

    source_set = DATA_ROOT / subject_id / "segments" / seg / f"{seg}_bergen_250hz.set"
    source_fif = DATA_ROOT / subject_id / "derivatives" / "01_bergen_250hz" / seg / f"{seg}_bergen_250hz_eeg.fif"
    out_set = out_deriv_dir / f"{seg}_ica_clean.set"
    out_html = out_deriv_dir / f"{seg}_ica_report.html"
    out_metrics = out_deriv_dir / f"{seg}_ica_metrics.json"

    input_source = _rel(source_set) if source_set.exists() else (_rel(source_fif) if source_fif.exists() else None)

    meta_src = PROJECT_ROOT / "config" / "ica_meta_model.json"
    rejection_mode = str(params_used.get("rejection_mode", (cfg.get("ica_final", {}) if cfg else {}).get("rejection_mode", ""))).strip().lower()
    meta_json_path = None
    if rejection_mode == "meta_model" and meta_src.exists():
        out_meta_json = out_deriv_dir / "ica_meta_model.json"
        try:
            shutil.copy2(meta_src, out_meta_json)
            meta_json_path = _rel(out_meta_json)
            print(f"  Saved meta-model: {out_meta_json.name}")
        except Exception:
            pass

    provenance = {
        "input_source_250hz": input_source,
        "input_bcg_fif": _rel(in_fif) if Path(in_fif).exists() else _rel(in_fif),
        "output_dir": _rel(out_deriv_dir),
        "output_clean_fif": _rel(out_fif),
        "output_clean_set": _rel(out_set) if out_set.exists() else None,
        "output_report_html": _rel(out_html) if out_html.exists() else None,
        "output_metrics_json": _rel(out_metrics) if out_metrics.exists() else None,
        "meta_model_json": meta_json_path,
    }

    payload = {
        "generated_at": datetime.datetime.now().isoformat(),
        "subject": str(subject_id),
        "segment": str(seg),
        "git": git_info,
        "provenance": provenance,
        "metrics": metrics,
        "params_used": params_used,
        "config": cfg,
    }

    out_deriv_dir.mkdir(parents=True, exist_ok=True)
    with open(info_path, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"  Saved pipeline provenance: {info_path}")
    return info_path


def apply_optimized_ica(segment_dir: Path = DEFAULT_SEGMENT_DIR, force: bool = False,
                        params_override: dict | None = None, cfg_snapshot: dict | None = None):
    """params_override: values that replace the step10-optimized ones
    (keys: flatline_crit, channel_crit, line_crit, iclabel_thresh, burst_crit)."""
    segment_dir = Path(segment_dir).resolve()
    seg = segment_dir.name
    subject_id = segment_dir.parent.parent.name if segment_dir.parent.name == "segments" else segment_dir.parent.name
    print("=" * 80)
    print(f"[STEP 11] Applying Optimized ICA Parameters to Full Dataset: {subject_id}/{seg}")
    print("=" * 80)

    out_deriv_dir = DATA_ROOT / subject_id / "derivatives" / "05_ica" / seg
    out_fif = out_deriv_dir / f"{seg}_ica_clean.fif"
    report_html = out_deriv_dir / f"{seg}_ica_report.html"
    if not force and out_fif.exists() and report_html.exists():
        print(f"  [SKIP] Final ICA output already exists: {out_fif.name} (use --force to recompute)")
        print("=" * 80)
        print("  [STEP 11] ALREADY DONE.")
        print("=" * 80)
        return out_fif

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
        print(f"  [WARN] Parameter file not found in {segment_dir}. Using robust defaults from config.")
        best = {
            "flatline_crit": FLATLINE_CRIT,
            "channel_crit": CHANNEL_CRIT,
            "line_crit": LINE_NOISE_CRIT,
            "iclabel_thresh": ICLABEL_THRESH,
            "burst_crit": ASR_BURST_CRIT
        }


    if params_override:
        print(f"  Applying parameter overrides: {params_override}")
        best = {**best, **params_override}

    # SAFETY GUARD: clamp Optuna params to prevent over-cleaning.
    # A threshold below 0.65 rejects "probably brain" ICs (P(brain)~0.4) and
    # destroys the signal (previously 0.6 -> 69% of ICs rejected, 77% var drop).
    thr_raw = float(best.get("iclabel_thresh", 0.80))
    if thr_raw < GUARD_MIN_ICLABEL_THRESH:
        print(f"    [GUARD] iclabel_thresh={thr_raw} is unsafe (< {GUARD_MIN_ICLABEL_THRESH}), clamping to 0.80")
        best["iclabel_thresh"] = 0.80
    ch_raw = best.get("channel_crit", "off")
    if isinstance(ch_raw, str) and ch_raw.lower() == "off":
        channel_crit = "off"
    else:
        try:
            ch_val = float(ch_raw)
            if GUARD_MAX_CHANNEL_CRIT is not None and ch_val > GUARD_MAX_CHANNEL_CRIT:
                print(f"    [GUARD] channel_crit={ch_val} is too aggressive (> {GUARD_MAX_CHANNEL_CRIT}), clamping to {GUARD_MAX_CHANNEL_CRIT}")
                ch_val = float(GUARD_MAX_CHANNEL_CRIT)
            channel_crit = ch_val
        except (ValueError, TypeError):
            channel_crit = "off"
    best["channel_crit"] = channel_crit

    # ASR burst-correction strength (SD cutoff k). Lower k = more aggressive.
    # Below 5 ASR starts reconstructing real neural bursts as "artifact" and
    # flattens the signal, so we clamp to a conservative floor. 'off' disables
    # ASR entirely (channel removal only, the old behaviour).
    burst_raw = best.get("burst_crit", 20.0)
    if isinstance(burst_raw, str) and burst_raw.lower() == "off":
        burst_crit = "off"
    else:
        burst_val = float(burst_raw)
        if burst_val < GUARD_MIN_BURST_CRIT:
            print(f"    [GUARD] burst_crit={burst_val} is unsafe (< {GUARD_MIN_BURST_CRIT}), clamping to 20.0")
            burst_val = 20.0
        burst_crit = burst_val

    for k, v in best.items():
        print(f"    {k:25s} = {v}")

    # Load BCG-cleaned data
    fif = _find_bcg_fif(segment_dir)
    print(f"\n  Loading BCG-cleaned data: {fif.name}")
    raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])

    # Extract ECG signal if available for ECG-guided residual BCG classification
    ecg_data = raw.get_data(picks="ECG")[0] if "ECG" in raw.ch_names else None
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
    channel_crit = best.get("channel_crit", "off")
    line_crit = best.get("line_crit", 4.0)
    iclabel_thresh = best.get("iclabel_thresh", 0.70)
    thresh_eye = float(best.get("thresh_eye", min(0.70, iclabel_thresh)))
    thresh_heart = float(best.get("thresh_heart", min(0.70, iclabel_thresh)))
    thresh_muscle = float(best.get("thresh_muscle", min(0.65, iclabel_thresh)))
    thresh_line_chan = float(best.get("thresh_line_chan", min(0.70, iclabel_thresh)))
    thresh_other = float(best.get("thresh_other", 0.50))
    other_brain_max = float(best.get("other_brain_max", 0.08))
    manual_reject_ics = best.get("manual_reject_ics", [])
    r_ecg_thresh = float(best.get("r_ecg_thresh", 0.25))
    max_brain_prob = float(best.get("max_brain_prob", 0.20))
    max_rej_fraction = float(best.get("max_rej_fraction", 0.25))
    ica_hp_hz = best.get("ica_hp_hz", ICA_HP_HZ)
    rejection_mode = str(best.get("rejection_mode", "thresholds")).strip().lower()

    # MATLAB literal formatting: 'off' (string) or bare float
    flat_ml = "'off'" if str(flatline_crit).lower() == "off" else f"{float(flatline_crit)}"
    channel_ml = "'off'" if str(channel_crit).lower() == "off" else f"{float(channel_crit)}"
    line_ml = "'off'" if str(line_crit).lower() == "off" else f"{float(line_crit)}"
    burst_ml = "'off'" if str(burst_crit).lower() == "off" else f"{float(burst_crit)}"
    print(f"    ASR BurstCriterion = {burst_crit}  (BurstRejection off -> correct, not cut)")

    out_mat = work_dir / "result_full.mat"
    m_file = work_dir / "run_final_ica.m"

    for f in work_dir.glob("temp_*"):
        f.unlink(missing_ok=True)

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

pop_saveset(EEG, 'filename', 'temp_full.set', 'filepath', '{work_dir.resolve().as_posix()}');
chanlocs_full = EEG.chanlocs;
save('{work_dir.resolve().as_posix()}/temp_chanlocs.mat', 'chanlocs_full', '-v7');
orig_chans = {{EEG.chanlocs.labels}};
n_orig = EEG.nbchan;

fprintf('Running clean_rawdata with optimized parameters...\\n');
% BurstCriterion enables ASR (matches the reference 'asr' stage). BurstRejection
% stays 'off' so bad bursts are RECONSTRUCTED in place rather than cut out --
% this preserves the sample count / duration, exactly like the reference .set.
% WindowCriterion stays 'off' for the same duration-preservation reason.
EEG_clean = pop_clean_rawdata(EEG, ...
    'FlatlineCriterion', {flat_ml}, ...
    'ChannelCriterion', {channel_ml}, ...
    'LineNoiseCriterion', {line_ml}, ...
    'Highpass', 'off', ...
    'BurstCriterion', {burst_ml}, ...
    'WindowCriterion', 'off', ...
    'BurstRejection', 'off', ...
    'Distance', 'Euclidian');

pop_saveset(EEG_clean, 'filename', 'temp_clean.set', 'filepath', '{work_dir.resolve().as_posix()}');
try, java.lang.System.exit(0); catch, exit(0); end
"""
    m_file1 = work_dir / "step11_part1.m"
    with open(m_file1, "w", encoding="ascii") as f:
        f.write(code)

    print(f"  [Part 1] Running clean_rawdata in MATLAB...")
    res1 = subprocess.run([str(MATLAB_BIN), "-nodisplay", "-nodesktop", "-nosplash", "-batch", f"run('{m_file1.resolve()}')"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if not (work_dir / "temp_clean.set").exists():
        raise RuntimeError(f"MATLAB Part 1 failed (rc={res1.returncode}):\\n{res1.stdout[-1500:] if res1.stdout else ''}")

    if ICLABEL_BACKEND == "python":
        print(f"  [Part 2 & 3] Running Picard ICA + mne-icalabel in pure Python...")
        from mne.preprocessing import ICA
        from mne_icalabel.iclabel import iclabel_label_components

        raw_full = mne.io.read_raw_eeglab(work_dir / "temp_full.set", preload=True, verbose=False)
        raw_clean = mne.io.read_raw_eeglab(work_dir / "temp_clean.set", preload=True, verbose=False)

        orig_chans = list(raw_full.ch_names)
        kept_chans = list(raw_clean.ch_names)
        removed_chans_list = [c for c in orig_chans if c not in kept_chans]
        n_removed = len(removed_chans_list)
        print(f"  Channels after clean_rawdata: {len(kept_chans)} (removed {n_removed}: {removed_chans_list})")

        # Standard ICLabel preprocessing: 1-100 Hz bandpass and average reference
        raw_for_ica = raw_clean.copy().filter(l_freq=1.0, h_freq=100.0, verbose=False).set_eeg_reference('average', verbose=False)

        data_rank = np.linalg.matrix_rank(raw_for_ica.get_data())
        n_pca = min(data_rank, int(ICA_PCA_MAX)) if ICA_PCA_MAX else data_rank
        print(f"  [Part 2] Picard ICA: {len(kept_chans)} channels, rank {data_rank}, {n_pca} components")

        ica = ICA(n_components=n_pca, method='picard', fit_params=dict(ortho=False, extended=True), random_state=42, max_iter=500)
        ica.fit(raw_for_ica, verbose=False)

        print("  [Part 3] Running ICLabel classification (mne-icalabel)...")
        ic_classes = iclabel_label_components(raw_for_ica, ica)

        # QRS-locked evoked-energy ratio per IC (needs <seg>_qrs.npy saved by step08 or bcg_qrs.mat)
        qrs_ratio = np.zeros(n_pca)
        cardiac_ics: set[int] = set()
        qrs_idx = None
        qrs_file = fif.with_name(fif.name.replace("_bcg_clean.fif", "_qrs.npy"))
        qrs_mat_file = fif.parent / "_matlab_tmp" / "bcg_qrs.mat"
        if qrs_file.exists():
            qrs_idx = np.load(qrs_file).astype(int)
        elif qrs_mat_file.exists():
            import scipy.io as sio
            mat_q = sio.loadmat(qrs_mat_file)
            if "qrs_lat" in mat_q:
                qrs_idx = (np.atleast_1d(mat_q["qrs_lat"]).flatten() - 1).astype(int)
                try:
                    np.save(qrs_file, qrs_idx)
                except Exception:
                    pass

        if CARDIAC_QRS_RATIO_THRESH and qrs_idx is not None:
            src_q = ica.get_sources(raw_for_ica).get_data()
            qa, qb = int(-0.2 * sfreq), int(0.8 * sfreq)
            ok_idx = [i for i in qrs_idx if i + qa >= 0 and i + qb < src_q.shape[1]]
            if len(ok_idx) >= 50:
                for c in range(n_pca):
                    ep = np.stack([src_q[c, i + qa:i + qb] for i in ok_idx])
                    ep = ep - ep.mean(axis=1, keepdims=True)
                    qrs_ratio[c] = float(np.mean(ep.mean(axis=0) ** 2) / (np.mean(ep ** 2) + 1e-30))
                ranked = [int(c) for c in np.argsort(qrs_ratio)[::-1][:CARDIAC_QRS_MAX_ICS]
                          if qrs_ratio[c] >= CARDIAC_QRS_RATIO_THRESH]
                cardiac_ics = {c for c in ranked if (qrs_ratio[c] >= 0.10 or float(ic_classes[c][0]) < CARDIAC_QRS_MAX_BRAIN)}
                print(f"    [QRS-LOCKED] {len(ok_idx)} beats; evoked-energy ratio top: "
                      f"{[(int(c), round(float(qrs_ratio[c]), 3)) for c in np.argsort(qrs_ratio)[::-1][:5]]} "
                      f"-> cardiac ICs {sorted(cardiac_ics)}")
            else:
                print(f"    [QRS-LOCKED] only {len(ok_idx)} usable beats; rule skipped")
        elif CARDIAC_QRS_RATIO_THRESH:
            print(f"    [QRS-LOCKED] neither {qrs_file.name} nor {qrs_mat_file.name} found; rule skipped")

        # Check cross-correlation with ECG over physiological lag window (0..350 ms)
        # to reliably detect residual BCG components (pulse delay to head is ~150-250 ms)
        ecg_corrs = np.zeros(n_pca)
        ecg_best_lags = np.zeros(n_pca)
        if ecg_data is not None:
            sources = ica.get_sources(raw_clean).get_data()
            min_len = min(sources.shape[1], len(ecg_data))
            e = ecg_data[:min_len] - np.mean(ecg_data[:min_len])
            e_std = np.std(e)
            max_lag = int(0.35 * sfreq)  # 350 ms physiological window
            if e_std > 1e-12 and min_len > max_lag:
                for c in range(n_pca):
                    s = sources[c, :min_len] - np.mean(sources[c, :min_len])
                    s_std = np.std(s)
                    if s_std > 1e-12:
                        lags_r = [abs(float(np.dot(s[lag:], e[:min_len - lag]) / ((min_len - lag) * s_std * e_std)))
                                  for lag in range(0, max_lag + 1, 2)]
                        if lags_r:
                            best_idx = int(np.argmax(lags_r))
                            ecg_corrs[c] = float(lags_r[best_idx])
                            ecg_best_lags[c] = (best_idx * 2) / sfreq * 1000.0

        # Candidate rejection selection (supports two modes: 'thresholds' and 'meta_model')
        candidate_rejects = []
        classes_names = ["Brain", "Muscle", "Eye", "Heart", "Line", "Chan", "Other"]

        if rejection_mode == "meta_model":
            meta_model_path = PROJECT_ROOT / "config" / "ica_meta_model.json"
            if meta_model_path.exists():
                try:
                    with open(meta_model_path, "r", encoding="utf-8") as f:
                        mm = json.load(f)
                    mm_weights = np.array(mm["weights"], dtype=np.float64)
                    mm_intercept = float(mm["intercept"])
                    mm_thresh = float(mm.get("threshold", 0.50))
                    print(f"  [REJECTION MODE] Regularized Meta-Model ({meta_model_path.name}, threshold={mm_thresh:.2f})")

                    sources = ica.get_sources(raw_clean).get_data()
                    var_k = np.sum(sources**2, axis=1)
                    tot_var = np.sum(var_k) + 1e-30

                    for c in range(n_pca):
                        probs = ic_classes[c]
                        dominant = int(np.argmax(probs))
                        max_prob = float(probs[dominant])
                        brain_prob = float(probs[0])
                        gap = max_prob - brain_prob
                        rel_var = float(var_k[c] / tot_var)

                        feat = np.array([
                            float(probs[0]), float(probs[1]), float(probs[2]), float(probs[3]),
                            float(probs[4]), float(probs[5]), float(probs[6]),
                            max_prob, gap, rel_var
                        ])
                        z = mm_intercept + float(np.dot(mm_weights, feat))
                        prob_artifact = 1.0 / (1.0 + np.exp(-z))

                        is_rej = False
                        # The meta-model already encodes P(brain) with a strong negative
                        # weight (w_brain=-1.38), so its decision is self-protecting. We
                        # only add a loose safety net: never cut an IC that ICLabel thinks
                        # is more likely brain than anything else (brain_prob < 0.5).
                        if prob_artifact >= mm_thresh and brain_prob < 0.5:
                            is_rej = True
                        elif ecg_corrs[c] >= r_ecg_thresh and brain_prob < max_brain_prob:
                            is_rej = True
                            print(f"    [ECG-GUIDED] IC {c:02d}: Residual BCG detected (lagged r_ECG={ecg_corrs[c]:.3f} at lag={ecg_best_lags[c]:.0f}ms, P_brain={brain_prob:.2f})")
                        elif c in cardiac_ics:
                            is_rej = True
                            print(f"    [QRS-LOCKED] IC {c:02d}: Residual BCG (evoked-energy ratio={qrs_ratio[c]:.3f}, P_brain={brain_prob:.2f})")

                        if is_rej:
                            candidate_rejects.append((c, prob_artifact))
                            print(f"    [META-MODEL] IC {c:02d}: P(artifact)={prob_artifact:.3f} (Class={classes_names[dominant]} {max_prob:.2f}, Brain={brain_prob:.2f})")
                except Exception as e:
                    print(f"  [WARN] Failed loading meta-model ({e}); falling back to universal thresholds")
                    rejection_mode = "thresholds"
            else:
                print(f"  [WARN] {meta_model_path} not found; falling back to universal thresholds")
                rejection_mode = "thresholds"

        if rejection_mode != "meta_model":
            print(f"  [REJECTION MODE] Universal Thresholds (Optuna: other={thresh_other:.2f}, muscle={thresh_muscle:.2f}, line/chan={thresh_line_chan:.2f}, eye={thresh_eye:.2f}, heart={thresh_heart:.2f})")
            for c in range(n_pca):
                probs = ic_classes[c]
                dominant = int(np.argmax(probs))
                max_prob = float(probs[dominant])
                brain_prob = float(probs[0])
                is_rej = False

                if dominant == 2 and max_prob >= thresh_eye and brain_prob < max_brain_prob:  # Eye (blinks/saccades)
                    is_rej = True
                elif (dominant == 3 or probs[3] >= thresh_heart) and brain_prob < max_brain_prob:  # Heart (residual BCG)
                    is_rej = True
                elif ecg_corrs[c] >= r_ecg_thresh and brain_prob < max_brain_prob:  # ECG-guided BCG residual
                    is_rej = True
                    print(f"    [ECG-GUIDED] IC {c:02d}: Residual BCG detected (lagged r_ECG={ecg_corrs[c]:.3f} at lag={ecg_best_lags[c]:.0f}ms, P_brain={brain_prob:.2f})")
                elif c in cardiac_ics:  # QRS-locked residual BCG
                    is_rej = True
                    print(f"    [QRS-LOCKED] IC {c:02d}: Residual BCG (evoked-energy ratio={qrs_ratio[c]:.3f}, P_brain={brain_prob:.2f})")
                elif dominant in (4, 5) and max_prob >= thresh_line_chan and brain_prob < min(max_brain_prob, 0.15):  # Line or Channel noise
                    is_rej = True
                elif dominant == 1 and max_prob >= thresh_muscle and brain_prob < min(max_brain_prob, 0.10):  # High-confidence Muscle
                    is_rej = True
                elif dominant == 6 and max_prob >= thresh_other and brain_prob < other_brain_max:  # Non-brain Other/Scanner/Pump noise
                    is_rej = True
                    print(f"    [OTHER/SCANNER] IC {c:02d}: Dominant Other noise (prob={max_prob:.2f}, P_brain={brain_prob:.2f})")

                if is_rej:
                    conf_score = max_prob - brain_prob + 0.5 * ecg_corrs[c] + (1.0 + qrs_ratio[c] if c in cardiac_ics else 0.0)
                    candidate_rejects.append((c, conf_score))

        # Manual IC exclusion support (from config or expert review)
        if isinstance(manual_reject_ics, (list, tuple)):
            for ic_item in manual_reject_ics:
                try:
                    ic_int = int(ic_item)
                    idx = ic_int if ic_int < n_pca else ic_int - 1
                    if 0 <= idx < n_pca:
                        if not any(c == idx for c, _ in candidate_rejects):
                            print(f"    [MANUAL] IC {idx:02d}: Manually marked for rejection")
                            candidate_rejects.append((idx, 999.0))
                except Exception:
                    pass

        # Over-cleaning guard: cap rejection at max_rej_fraction of components (at least 5 to allow eye + ECG residuals)
        max_allowed = max(5, int(max_rej_fraction * n_pca))
        if len(candidate_rejects) > max_allowed:
            candidate_rejects.sort(key=lambda x: x[1], reverse=True)
            print(f"    [GUARD] Capping rejected ICs from {len(candidate_rejects)} to {max_allowed} (top confidence)")
            candidate_rejects = candidate_rejects[:max_allowed]

        reject_ic_list = sorted([c for c, _ in candidate_rejects])
        n_ic_rejected = len(reject_ic_list)
        n_ic = n_pca
        print(f"  ICs rejected: {n_ic_rejected} / {n_ic} ({reject_ic_list})")
        classes_names = ["Brain", "Muscle", "Eye", "Heart", "Line", "Chan", "Other"]
        for c in reject_ic_list:
            dom_idx = int(np.argmax(ic_classes[c]))
            print(f"    - IC {c:02d}: {classes_names[dom_idx]} (prob={ic_classes[c][dom_idx]:.2f}, Brain={ic_classes[c][0]:.2f})")

        # Exclude artifact ICs from the cleaned dataset
        ica.exclude = reject_ic_list
        raw_clean_ica = ica.apply(raw_clean.copy(), verbose=False)

        # Transfer cleaned signals back into raw_full
        for ch in kept_chans:
            raw_full._data[raw_full.ch_names.index(ch)] = raw_clean_ica.get_data(picks=ch)[0]

        # Spherical spline interpolation of removed channels in MNE
        if n_removed > 0:
            print(f"  Interpolating {n_removed} removed channels spherical in MNE...")
            raw_full.info['bads'] = removed_chans_list
            raw_full.interpolate_bads(reset_bads=True, mode='accurate', verbose=False)
        else:
            print(f"  All {len(kept_chans)} channels preserved intact (no channel removal, interpolation skipped).")

        print("  Applying common average reference...")
        raw_full.set_eeg_reference('average', verbose=False)
        after_data = raw_full.get_data()  # Volts

        m_file1.unlink(missing_ok=True)
        (work_dir / "temp_full.set").unlink(missing_ok=True)
        (work_dir / "temp_full.fdt").unlink(missing_ok=True)
        (work_dir / "temp_clean.set").unlink(missing_ok=True)
        (work_dir / "temp_clean.fdt").unlink(missing_ok=True)
        (work_dir / "temp_chanlocs.mat").unlink(missing_ok=True)
        mat_in.unlink(missing_ok=True)
    else:
        # --- MATLAB Part 2 & 3 Fallback ---
        print("  [Part 2] Running Picard ICA in Python (MNE)...")
        from mne.preprocessing import ICA
        import scipy.io as sio

        raw_clean = mne.io.read_raw_eeglab(work_dir / "temp_clean.set", preload=True, verbose=False)
        raw_ica = raw_clean.copy().filter(l_freq=ica_hp_hz, h_freq=None, verbose=False)

        pruned_data = raw_clean.get_data()
        data_rank = np.linalg.matrix_rank(pruned_data)
        n_pca = min(data_rank, int(ICA_PCA_MAX)) if ICA_PCA_MAX else data_rank
        print(f"    Channels: {len(raw_clean.ch_names)}, Rank: {data_rank}, ICA components: {n_pca}")

        ica = ICA(n_components=n_pca, method='picard', random_state=42, max_iter=500)
        ica.fit(raw_ica, verbose=False)

        icaweights = ica.unmixing_matrix_ @ ica.pca_components_[:ica.n_components_] * 1e-6
        icasphere = np.eye(len(raw_clean.ch_names))
        icawinv = np.linalg.pinv(icaweights)

        sio.savemat(work_dir / "temp_weights.mat", {
            "icaweights": icaweights.astype(np.float64),
            "icasphere": icasphere.astype(np.float64),
            "icawinv": icawinv.astype(np.float64)
        }, do_compression=True)

        code2 = f"""
addpath('{EEGLAB_DIR.resolve()}');
addpath('{iclabel_plugin.resolve()}');
addpath('{DIPFIT_DIR.resolve()}');
eeglab nogui;

EEG_clean = pop_loadset('filename', 'temp_clean.set', 'filepath', '{work_dir.resolve().as_posix()}');
load('{work_dir.resolve().as_posix()}/temp_weights.mat', 'icaweights', 'icasphere', 'icawinv');
load('{work_dir.resolve().as_posix()}/temp_chanlocs.mat', 'chanlocs_full');

orig_chans = {{'{ "','".join(raw.ch_names) }'}};
kept_chans = {{EEG_clean.chanlocs.labels}};
removed_chans = setdiff(orig_chans, kept_chans);
n_removed = length(removed_chans);
n_chans_after = EEG_clean.nbchan;

EEG_clean.icaweights = icaweights;
EEG_clean.icasphere = icasphere;
EEG_clean.icawinv = icawinv;
EEG_clean.icachansind = 1:EEG_clean.nbchan;

n_ic = size(EEG_clean.icaweights, 1);
EEG_clean = pop_iclabel(EEG_clean, 'default');
ic_classes = EEG_clean.etc.ic_classification.ICLabel.classifications;

rej_mask = false(1, n_ic);
conf = zeros(1, n_ic);
for c = 1:n_ic
    probs = ic_classes(c, :);
    [max_prob, dominant] = max(probs);
    brain_prob = probs(1);
    is_rej = false;
    if (dominant == 3) && max_prob >= min(0.70, {iclabel_thresh}) && brain_prob < 0.20
        is_rej = true;
    elseif (dominant == 4) && max_prob >= min(0.70, {iclabel_thresh}) && brain_prob < 0.20
        is_rej = true;
    elseif (dominant == 5 || dominant == 6) && max_prob >= {iclabel_thresh} && brain_prob < 0.15
        is_rej = true;
    elseif (dominant == 2) && max_prob >= max(0.80, {iclabel_thresh}) && brain_prob < 0.10
        is_rej = true;
    end
    if is_rej
        rej_mask(c) = true;
        conf(c) = max_prob - brain_prob;
    end
end
reject_ic_list = find(rej_mask);
max_allowed = max(1, floor(0.20 * n_ic));
if length(reject_ic_list) > max_allowed
    [~, sort_idx] = sort(conf(reject_ic_list), 'descend');
    reject_ic_list = reject_ic_list(sort_idx(1:max_allowed));
    reject_ic_list = sort(reject_ic_list);
end
n_ic_rejected = length(reject_ic_list);

if n_ic_rejected > 0
    ica_act = (EEG_clean.icaweights * EEG_clean.icasphere) * double(EEG_clean.data);
    ica_act(reject_ic_list, :) = 0;
    EEG_clean.data = EEG_clean.icawinv * ica_act;
end

if n_removed > 0
    EEG_clean = pop_interp(EEG_clean, chanlocs_full, 'spherical');
end

EEG_clean = pop_reref(EEG_clean, []);
after_data = single(EEG_clean.data);
save('{out_mat.resolve()}', 'after_data', 'removed_chans', 'n_removed', 'n_ic_rejected', 'ic_classes', 'reject_ic_list', 'n_ic', '-v7');
clear EEG EEG_clean;
exit(0);
"""
        m_file2 = work_dir / "step11_part3.m"
        with open(m_file2, "w", encoding="ascii") as f:
            f.write(code2)

        print("\n  [Part 3] Running ICLabel and Interpolation in MATLAB...")
        res2 = subprocess.run([str(MATLAB_BIN), "-nodisplay", "-nodesktop", "-nosplash", "-batch", f"run('{m_file2.resolve()}')"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if res2.returncode != 0 or not out_mat.exists():
            print(res2.stdout[-2000:] if res2.stdout else "")
            raise RuntimeError(f"MATLAB Part 3 failed (rc={res2.returncode})")

        m_file1.unlink(missing_ok=True)
        m_file2.unlink(missing_ok=True)
        (work_dir / "temp_clean.set").unlink(missing_ok=True)
        (work_dir / "temp_clean.fdt").unlink(missing_ok=True)
        (work_dir / "temp_weights.mat").unlink(missing_ok=True)
        (work_dir / "temp_chanlocs.mat").unlink(missing_ok=True)

        m_res = loadmat(out_mat)
        after_data = m_res["after_data"] * 1e-6
        n_removed = int(m_res["n_removed"][0, 0])
        n_ic_rejected = int(m_res["n_ic_rejected"][0, 0])
        n_ic = int(m_res["n_ic"][0, 0])
        ic_classes = m_res["ic_classes"]
        reject_ic_list = (m_res["reject_ic_list"].flatten() - 1).tolist() if ("reject_ic_list" in m_res and m_res["reject_ic_list"].size > 0) else []
        removed_chans_list = []
        if n_removed > 0 and "removed_chans" in m_res:
            rc = m_res["removed_chans"]
            if hasattr(rc, "size") and rc.size > 0:
                for val in rc.flatten():
                    removed_chans_list.append(str(val[0]) if isinstance(val, (np.ndarray, list)) else str(val))
        out_mat.unlink(missing_ok=True)
        mat_in.unlink(missing_ok=True)

    # Compute metrics
    print("  Computing metrics...")
    # Reference consistency: after_data is in average reference (CAR);
    # compute baseline on average-referenced data so montage re-referencing does not distort metrics.
    raw_before_car = raw.copy().set_eeg_reference('average', verbose=False)
    data_before_car = raw_before_car.get_data()

    nperseg = int(min(4 * sfreq, after_data.shape[-1]))
    f_after, psd_after = welch(after_data, sfreq, nperseg=nperseg, axis=-1)
    f_before, psd_before = welch(data_before_car, sfreq, nperseg=nperseg, axis=-1)

    m_alpha = (f_after >= ALPHA_BAND[0]) & (f_after <= ALPHA_BAND[1])

    # Occipital/parietal channels for true physiological alpha retention
    occ_par_names = {"O1", "Oz", "O2", "POz", "Pz", "P3", "P4", "P1", "P2"}
    occ_idx = [i for i, ch in enumerate(ch_names) if ch in occ_par_names]
    if len(occ_idx) > 0:
        p_alpha_after = float(np.mean(psd_after[occ_idx, :][:, m_alpha]))
        p_alpha_before = float(np.mean(psd_before[occ_idx, :][:, m_alpha]))
    else:
        p_alpha_after = float(np.mean(psd_after[:, m_alpha]))
        p_alpha_before = float(np.mean(psd_before[:, m_alpha]))
    alpha_retention = p_alpha_after / max(p_alpha_before, 1e-12)

    var_before = float(np.var(data_before_car))
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
        "iclabel_thresh": float(iclabel_thresh),
        "params_used": {
            "flatline_crit": flatline_crit if str(flatline_crit).lower() == "off" else float(flatline_crit),
            "channel_crit": channel_crit if str(channel_crit).lower() == "off" else float(channel_crit),
            "line_crit": line_crit if str(line_crit).lower() == "off" else float(line_crit),
            "iclabel_thresh": float(iclabel_thresh),
            "burst_crit": burst_crit if str(burst_crit).lower() == "off" else float(burst_crit),
            "other_brain_max": float(OTHER_BRAIN_MAX),
            "ica_pca_max": int(ICA_PCA_MAX) if ICA_PCA_MAX else None,
            "rejection_mode": rejection_mode,
        },
    }

    print(f"\n  Alpha retention: {alpha_retention:.2%}")
    print(f"  Variance drop:   {variance_drop:.2%}")
    print(f"  Channels removed: {n_removed} / {len(ch_names)}")
    print(f"  ICs found:        {n_ic}")
    print(f"  ICs rejected:     {n_ic_rejected} / {n_ic}")

    # Save final cleaned data as FIF
    out_deriv_dir = DATA_ROOT / subject_id / "derivatives" / "05_ica" / seg
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
        json.dump(
            metrics,
            f,
            indent=2,
            default=lambda o: int(o) if isinstance(o, (np.integer, int)) else (
                float(o) if isinstance(o, (np.floating, float)) else (
                    o.tolist() if isinstance(o, np.ndarray) else str(o)
                )
            ),
        )

    # Save removed channels list
    ch_deriv_dir = DATA_ROOT / subject_id / "derivatives" / "04_channels" / seg
    ch_deriv_dir.mkdir(parents=True, exist_ok=True)
    removed_json = ch_deriv_dir / "removed_channels.json"
    with open(removed_json, "w") as f:
        json.dump({"removed": removed_chans_list, "n_total": len(ch_names)}, f, indent=2)

    # Generate QC plots and HTML report
    print("\n  Generating QC plots & report...")
    generate_ica_report(data, after_data, ch_names, sfreq, ic_classes,
                        reject_ic_list, removed_chans_list, metrics, seg, subject_id)

    # Save pipeline provenance YAML
    save_pipeline_info(subject_id, seg, out_deriv_dir, in_fif=fif, out_fif=out_fif,
                       metrics=metrics, params_used=best, cfg=cfg_snapshot)

    print("\n" + "=" * 80)
    print(f"[STEP 11] ICA Complete! Output: {out_fif.name}")
    print("=" * 80)
    return out_fif


def generate_ica_report(data_before, data_after, ch_names, sfreq, ic_classes,
                        reject_ic_list, removed_chans, metrics, seg, subject_id: str = DEFAULT_EXPERIMENT):
    """Generate HTML report with PSD, ICLabel, and channel plots."""
    qc_dir = PROJECT_ROOT / "qc" / subject_id / "ica"
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

    out_deriv_dir = DATA_ROOT / subject_id / "derivatives" / "05_ica" / seg
    report_html = out_deriv_dir / f"{seg}_ica_report.html"
    with open(report_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"    {report_html.name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="STEP 11: Apply optimized ICA to full dataset")
    parser.add_argument("--subject", default=None, help="Subject ID (e.g. 1916)")
    parser.add_argument("--segment", default=None, help="Segment name (e.g. ec, drone) or omit for all segments of subject")
    parser.add_argument("--all", action="store_true", help="Process all available subjects and segments")
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
            apply_optimized_ica(sdir, force=args.force)
