"""
STEP 05: Clean Full Multi-Channel Dataset in MATLAB with Winner Optuna Parameters
==================================================================================
Loads the working sample range from the continuous raw BrainVision dataset,
reads slice trigger indices from slice_triggers.txt, and applies vectorized Bergen AAS.
Saves a single final EEGLAB dataset without creating intermediate duplicate files.
"""
from pathlib import Path
import subprocess
import json

import mne
import numpy as np
import gc
from scipy.io import savemat

try:
    from .config import MATLAB_BIN, EEGLAB_DIR, BERGEN_DIR, PROJECT_ROOT, DEFAULT_SEGMENT_DIR
except ImportError:
    from config import MATLAB_BIN, EEGLAB_DIR, BERGEN_DIR, PROJECT_ROOT, DEFAULT_SEGMENT_DIR


def clean_full_dataset(segment_dir: Path = DEFAULT_SEGMENT_DIR,
                       shift: int = None,
                       win_k: int = None,
                       motion_thresh: float = None,
                       pad_volumes: int = 3,
                       force: bool = False):
    """
    pad_volumes: give Bergen AAS extra whole fMRI volumes on each side of the
        work window (taken from the dummy-scan / post-task gradient that already
        exists in the continuous recording) so the moving-average template at the
        first/last real volume has neighbours. The padding is trimmed back off
        after correction, so the output length is unchanged. 0 = old behaviour
        (tight crop, edge residual). Padding is clamped to what the recording
        actually provides on each side.
    """
    segment_dir = Path(segment_dir).resolve()
    print("=" * 70)
    print(f"[STEP 05] Full Dataset Bergen Cleaning for: {segment_dir.name}")
    print("=" * 70)

    # Check if cleaned .set already exists
    existing_sets = list(segment_dir.glob("*bergen*optuna*.set")) or list(segment_dir.glob("*.set"))
    if not force and existing_sets:
        clean_set = max(existing_sets, key=lambda p: p.stat().st_mtime)
        print(f"  [SKIP] Bergen-cleaned dataset already exists: {clean_set.name} (use --force to recompute)")
        print("=" * 70)
        print("  [STEP 05] ALREADY DONE.")
        print("=" * 70)
        return clean_set

    work_info_path = segment_dir / "segment_work_info.json"
    if not work_info_path.exists():
        raise FileNotFoundError(f"Missing {work_info_path}. Run step03 first!")

    with open(work_info_path, "r", encoding="utf-8") as f:
        work_info = json.load(f)

    raw_vhdr = Path(work_info["raw_vhdr"]).resolve()
    t_start = float(work_info["t_work_start_sec"])
    t_stop  = float(work_info["t_work_stop_sec"])
    slices_per_volume = int(work_info.get("slices_per_volume", 25))
    nominal_slice_samples = int(work_info.get("nominal_slice_samples", 500))
    rp_file_str = work_info.get("rp_file") or work_info.get("rp_path")
    rp_path = Path(rp_file_str).resolve() if rp_file_str else None
    triggers_path = segment_dir / "slice_triggers.txt"

    # Load Optuna parameters if not provided
    params_json = segment_dir / "optuna_best_params.json"
    if shift is None or win_k is None or motion_thresh is None:
        if params_json.exists():
            with open(params_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                best = data.get("best_params", {})
                shift = shift if shift is not None else best.get("shift", 6)
                win_k = win_k if win_k is not None else best.get("win_k", 4)
                motion_thresh = motion_thresh if motion_thresh is not None else best.get("motion_thresh", 0.8)
        else:
            shift = 6 if shift is None else shift
            win_k = 4 if win_k is None else win_k
            motion_thresh = 0.8 if motion_thresh is None else motion_thresh

    print(f"  Applying Winner Parameters: shift={shift:+d} | win_k={win_k} | motion_thresh={motion_thresh:.2f}")

    # Extract cropped segment in Python via memory-mapped lazy reading (avoids EEGLAB 11GB pop_loadbv spike)
    raw = mne.io.read_raw_brainvision(raw_vhdr, preload=False, verbose=False)
    sfreq = float(raw.info["sfreq"])
    rec_end = float(raw.times[-1])
    nominal_volume_samples = slices_per_volume * nominal_slice_samples

    # ----- Edge padding for Bergen AAS -----------------------------------------
    # The AAS template for a volume is a moving average of neighbouring volumes'
    # slice-locked epochs. At the first/last work volume there are no neighbours
    # on one side -> poor subtraction -> a large residual gradient transient that
    # survives into P02 (seen as a huge spike in the first ~1 s / last ~3 s).
    # Fix: extend the crop by whole fMRI volumes of real gradient that already
    # exist in the continuous recording (dummy scans before the task, post-task
    # volumes after it), run Bergen on the padded window, then trim the pad back
    # off so the output window is exactly the original [t_start .. t_stop].
    t0_smp = int(round(t_start * sfreq))
    t1_smp = int(round(t_stop * sfreq))              # exclusive stop (t_work_stop)
    orig_length = t1_smp - t0_smp

    # Gradient-presence gate: a pad side is only useful if it actually contains
    # the MR gradient (so Bergen has real neighbouring volumes to average). Dummy
    # scans before the task are gradient-ON; but with per-task fMRI runs the
    # scanner stops after the last volume, so the region after t_stop is gradient
    # -OFF and padding there would corrupt the last volume's template. We compare
    # each candidate pad region's RMS to the task RMS and skip a side without it.
    _picks = [i for i, c in enumerate(raw.ch_names) if c != "ECG"][:8]
    def _rms(a_smp, b_smp):
        a = max(0, a_smp); b = min(raw.n_times, b_smp)
        if b - a < int(0.5 * sfreq):
            return 0.0
        d = raw.get_data(picks=_picks, start=a, stop=b) * 1e6
        return float(np.sqrt(np.mean(d ** 2)))
    task_rms = _rms(t0_smp, t0_smp + nominal_volume_samples)   # first work volume
    GRAD_FRAC = 0.5                                            # pad must have >=50% of task RMS

    left_pad_vols = right_pad_vols = 0
    if pad_volumes and pad_volumes > 0:
        cand_left = min(pad_volumes, t0_smp // nominal_volume_samples)
        if cand_left and _rms(t0_smp - cand_left * nominal_volume_samples, t0_smp) >= GRAD_FRAC * task_rms:
            left_pad_vols = cand_left
        elif cand_left:
            print(f"  [PAD] left pad skipped: no gradient before t_start "
                  f"(RMS {_rms(t0_smp - cand_left*nominal_volume_samples, t0_smp):.0f} < {GRAD_FRAC}*{task_rms:.0f})")
        rec_end_smp = int(round(rec_end * sfreq)) + 1
        cand_right = min(pad_volumes, max(0, (rec_end_smp - t1_smp) // nominal_volume_samples))
        if cand_right and _rms(t1_smp, t1_smp + cand_right * nominal_volume_samples) >= GRAD_FRAC * task_rms:
            right_pad_vols = cand_right
        elif cand_right:
            print(f"  [PAD] right pad skipped: no gradient after t_stop "
                  f"(RMS {_rms(t1_smp, t1_smp + cand_right*nominal_volume_samples):.0f} < {GRAD_FRAC}*{task_rms:.0f})")
    left_pad = left_pad_vols * nominal_volume_samples
    right_pad = right_pad_vols * nominal_volume_samples

    new_tmin = (t0_smp - left_pad) / sfreq
    new_tmax = min(rec_end, (t1_smp + right_pad) / sfreq)
    if left_pad or right_pad:
        print(f"  Bergen edge-padding: +{left_pad_vols} vol left, +{right_pad_vols} vol right "
              f"({left_pad/sfreq:.1f}s / {right_pad/sfreq:.1f}s); trimmed back after correction")
    print(f"  Loading padded segment interval [{new_tmin:.2f}s .. {new_tmax:.2f}s] with MNE "
          f"(work window [{t_start:.2f} .. {t_stop:.2f}]) ...")
    raw_crop = raw.crop(tmin=new_tmin, tmax=new_tmax)
    data_crop = raw_crop.get_data(units="uV").astype(np.float32)
    ch_names = list(raw_crop.ch_names)
    del raw, raw_crop
    gc.collect()

    # Extended slice triggers (periodic, 1-based relative to the padded crop).
    # Because the crop is extended by an integer number of volumes (=integer
    # multiple of nominal_slice_samples) the slice grid stays phase-aligned, so
    # the periodic triggers from sample 1 still land on real slice onsets.
    orig_triggers = np.loadtxt(triggers_path).astype(int).ravel()
    n_work_volumes = len(orig_triggers) // slices_per_volume
    total_slices_padded = (left_pad_vols + n_work_volumes + right_pad_vols) * slices_per_volume
    padded_triggers = np.array([1 + i * nominal_slice_samples for i in range(total_slices_padded)], dtype=int)
    triggers_padded_path = segment_dir / "slice_triggers_padded.txt"
    np.savetxt(triggers_padded_path, padded_triggers, fmt="%d")
    triggers_path = triggers_padded_path

    # Pad the SPM motion file by edge-replication (not zeros: a zero row would
    # create a fake motion jump at the dummy/work boundary and down-weight the
    # first real volume). The pad volumes thus inherit the nearest real motion.
    if rp_path and rp_path.exists() and (left_pad_vols or right_pad_vols):
        rp_data = np.loadtxt(rp_path)
        if rp_data.ndim == 1:
            rp_data = rp_data[None, :]
        rp_padded = np.vstack(
            [np.repeat(rp_data[:1], left_pad_vols, axis=0), rp_data,
             np.repeat(rp_data[-1:], right_pad_vols, axis=0)])
        rp_padded_path = segment_dir / "rp_padded.txt"
        np.savetxt(rp_padded_path, rp_padded, fmt="%.10f")
        rp_path = rp_padded_path

    temp_mat = segment_dir / "temp_multich_crop.mat"
    savemat(temp_mat, {
        "data": data_crop,
        "srate": sfreq,
        "labels": np.array(ch_names, dtype=object)
    })
    print(f"  Streamed segment to temporary MAT ({temp_mat.stat().st_size / (1024*1024):.1f} MB, {data_crop.shape[0]} ch, {data_crop.shape[1]} pnts)")
    del data_crop
    gc.collect()

    shift_str = f"p{shift}" if shift >= 0 else f"m{-shift}"
    out_name = f"{segment_dir.name}_bergen_optuna_sh_{shift_str}_k{win_k}_mt{str(motion_thresh).replace('.', 'p')}_nolpf.set"
    out_set = segment_dir / out_name
    m_script = segment_dir / "run_full_optuna_clean.m"

    helper_dir = Path(__file__).parent.resolve()

    # Trim the padding back off (in MATLAB, keeping the EEGLAB .set intact) so the
    # saved window is exactly the original [t_start .. t_stop]. 1-based inclusive.
    trim_code = ""
    if left_pad or right_pad:
        trim_code = (f"fprintf('Trimming edge padding back to work window "
                     f"({orig_length} pnts)...\\n');\n"
                     f"EEG = pop_select(EEG, 'point', [{left_pad + 1} {left_pad + orig_length}]);\n"
                     f"EEG = eeg_checkset(EEG);")

    rp_code = ""
    if rp_path and rp_path.exists():
        rp_code = f"""
        fprintf('Building motion-weighted matrix W (k={win_k}, thresh={motion_thresh})...\\n');
        rp_file = '{rp_path.resolve()}';
        n_vols = round(n_slices / {slices_per_volume});
        [motiondata, W_vol] = m_rp_info(rp_file, n_vols, {motion_thresh}, {win_k});
        W = kron(W_vol, eye({slices_per_volume}));
        clear W_vol motiondata;
        """
    else:
        rp_code = f"""
        fprintf('Building moving-average matrix W (k={win_k})...\\n');
        W_vol = m_moving_average(n_slices / {slices_per_volume}, {win_k});
        W = kron(W_vol, eye({slices_per_volume}));
        clear W_vol;
        """

    code = f"""
addpath('{EEGLAB_DIR.resolve()}');
addpath('{BERGEN_DIR.resolve()}');
addpath('{helper_dir}');
addpath('{PROJECT_ROOT.resolve()}');
eeglab nogui;

fprintf('Loading cropped multi-channel MAT...\\n');
S = load('{temp_mat.resolve()}');
EEG = eeg_emptyset();
EEG.setname = '{out_name}';
EEG.data = S.data;
EEG.srate = double(S.srate);
EEG.nbchan = size(S.data, 1);
EEG.pnts = size(S.data, 2);
EEG.trials = 1;
EEG.xmin = 0;
EEG.xmax = (EEG.pnts - 1) / EEG.srate;
EEG.chanlocs = struct([]);
for i = 1:EEG.nbchan
    if iscell(S.labels)
        EEG.chanlocs(i).labels = char(S.labels{{i}});
    else
        EEG.chanlocs(i).labels = deblank(S.labels(i, :));
    end
end
clear S;
EEG = eeg_checkset(EEG);

fprintf('Loading slice triggers from text file...\\n');
Peak_slices = load('{triggers_path.resolve()}');
Peak_slices = Peak_slices(:)';
"""
    code += f"""
% TR_sl is the FIXED nominal slice period. It must NOT be derived from
% Peak_slices(2)-Peak_slices(1): a negative shift clips the first trigger up to
% 1, which would shrink TR_sl and leave a periodic uncorrected gap between every
% slice (the 20/30/40 Hz comb survives). Deriving it from the nominal period
% keeps every correction window exactly one slice long regardless of shift.
TR_sl = {nominal_slice_samples};
"""
    if shift != 0:
        code += f"""
% Only shift the triggers. Do NOT clip to [1,pnts] (that shrank TR_sl and left a
% periodic uncorrected gap -> 20/30/40 Hz comb survived) and do NOT drop slices
% (that breaks the n_slices = n_vols*{slices_per_volume} invariant that
% kron(W_vol, eye) relies on). bergen_fast_correction already skips the 1-2
% out-of-bounds boundary slices via its own valid_slices mask, so the full
% trigger count is preserved for building W.
Peak_slices = Peak_slices + ({shift});
"""
    code += f"""
n_slices = length(Peak_slices);
fprintf('TR_slice=%d samples, n_slices=%d\\n', TR_sl, n_slices);

{rp_code}

fprintf('Executing vectorized Bergen AAS...\\n');
onset_val  = 0;
offset_val = onset_val + TR_sl - 1;
EEG = bergen_fast_correction(EEG, W, Peak_slices, onset_val, offset_val);
clear W;
{trim_code}
fprintf('Saving cleaned EEGLAB dataset: {out_name}...\\n');
pop_saveset(EEG, 'filename', '{out_name}', 'filepath', '{out_set.parent.resolve()}', 'savemode', 'onefile');
clear EEG;
fprintf('=== CLEANING FINISHED SUCCESSFULLY! ===\\n');
exit(0);
"""

    with open(m_script, "w", encoding="ascii") as f:
        f.write(code)

    res = subprocess.run(
        [str(MATLAB_BIN), "-nodisplay", "-nodesktop", "-nosplash", "-batch", f"run('{m_script.resolve()}')"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    m_script.unlink(missing_ok=True)
    temp_mat.unlink(missing_ok=True)
    (segment_dir / "slice_triggers_padded.txt").unlink(missing_ok=True)
    (segment_dir / "rp_padded.txt").unlink(missing_ok=True)
    print(res.stdout[-1000:] if res.stdout else "")

    if res.returncode == 0 and out_set.exists():
        print(f"\n[STEP 05] Successfully created cleaned dataset: {out_set.name} ({out_set.stat().st_size / (1024*1024):.1f} MB)")
        return out_set
    else:
        raise RuntimeError(f"MATLAB execution failed with return code {res.returncode}")


if __name__ == "__main__":
    import argparse
    try:
        from .config import DATA_ROOT
    except ImportError:
        from config import DATA_ROOT

    parser = argparse.ArgumentParser(description="STEP 05: Bergen gradient artifact cleaning")
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
        print("[ERROR] No segments found with segment_work_info.json. Run step03_trim_dummy.py first!")
    else:
        for sdir in seg_dirs:
            clean_full_dataset(segment_dir=sdir, force=args.force)

