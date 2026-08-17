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
                       motion_thresh: float = None):
    segment_dir = Path(segment_dir).resolve()
    print("=" * 70)
    print(f"[STEP 05] Full Dataset Bergen Cleaning for: {segment_dir.name}")
    print("=" * 70)

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
    rp_file_str = work_info.get("rp_file")
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
    print(f"  Loading cropped segment interval [{t_start:.2f}s .. {t_stop:.2f}s] with MNE...")
    raw = mne.io.read_raw_brainvision(raw_vhdr, preload=False, verbose=False)
    sfreq = float(raw.info["sfreq"])
    raw_crop = raw.crop(tmin=t_start, tmax=min(raw.times[-1], t_stop))
    data_crop = raw_crop.get_data(units="uV").astype(np.float32)
    ch_names = list(raw_crop.ch_names)
    del raw, raw_crop
    gc.collect()

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

fprintf('Saving cleaned EEGLAB dataset: {out_name}...\\n');
pop_saveset(EEG, 'filename', '{out_name}', 'filepath', '{out_set.parent.resolve()}', 'savemode', 'onefile');
clear EEG;
fprintf('=== CLEANING FINISHED SUCCESSFULLY! ===\\n');
exit(0);
"""

    with open(m_script, "w", encoding="ascii") as f:
        f.write(code)

    res = subprocess.run(
        [str(MATLAB_BIN), "-nodesktop", "-nosplash", "-batch", f"run('{m_script.resolve()}')"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    m_script.unlink(missing_ok=True)
    temp_mat.unlink(missing_ok=True)
    print(res.stdout[-1000:] if res.stdout else "")

    if res.returncode == 0 and out_set.exists():
        print(f"\n[STEP 05] Successfully created cleaned dataset: {out_set.name} ({out_set.stat().st_size / (1024*1024):.1f} MB)")
        return out_set
    else:
        raise RuntimeError(f"MATLAB execution failed with return code {res.returncode}")


if __name__ == "__main__":
    clean_full_dataset()

