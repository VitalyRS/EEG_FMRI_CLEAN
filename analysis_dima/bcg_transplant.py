"""
BCG "transplant" experiment: which factor makes my BCG correction worse than the
reference (Dima) one?

Input  : my 02_resampled250 data (Bergen-corrected, 250 Hz, band-passed 1-45 Hz)
Factors: QRS source    {mine (fMRIB detect) , Dima's events (time-aligned)}
         OBS npc       {3 (current pipeline), Dima's CB_corr_N}
Metrics (all vs Dima's final file, same 1-45Hz + average-ref treatment):
         QRS-locked residual RMS (uV) using Dima's QRS as the cardiac clock,
         RMS ratio, 1-8 Hz power ratio, occipital alpha ratio, mean r.

Usage: python analysis_dima/bcg_transplant.py --seg drone --sid 1924 1925
"""
from __future__ import annotations
import argparse, glob, re, shutil, subprocess, sys, warnings
from pathlib import Path
import numpy as np
import scipy.io as sio
import pandas as pd
from scipy.signal import welch

warnings.filterwarnings("ignore")
import mne
mne.set_log_level("ERROR")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import step08_bcg as s8                      # noqa: E402

OCC = ["O1", "Oz", "O2", "POz", "Pz", "P3", "P4", "P1", "P2"]
PILOT = s8.PILOT_CHANNELS


def load_dima(sid, seg):
    p = glob.glob(str(ROOT / "data_dima" / seg / sid / "*.set"))[0]
    raw = mne.io.read_raw_eeglab(p, preload=True)
    m = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
    qrs = np.array([float(e.latency) - 1 for e in np.atleast_1d(m["event"]) if str(e.type) == "qrs"])
    cb = int(re.search(r"CB_corr_(\d+)", p).group(1))
    return raw, qrs, cb


def lock_rms(x, sf, qrs, tmin=-0.2, tmax=0.8):
    a, b = int(tmin * sf), int(tmax * sf)
    ep = [x[:, int(q) + a:int(q) + b] for q in qrs if int(q) + a >= 0 and int(q) + b < x.shape[1]]
    if len(ep) < 20:
        return np.nan
    ep = np.stack(ep)
    ep = ep - ep.mean(axis=2, keepdims=True)
    return float(np.sqrt(np.mean(ep.mean(axis=0) ** 2)) * 1e6)


def calibrate_shift(x_pre, ch_idx, dq, sf, rng=250):
    """shift s (my_index = dima_index + s) that maximises beat-locked average of BCG-contaminated data."""
    best = (-1, 0)
    for s in range(-rng, rng + 1):
        v = lock_rms(x_pre[ch_idx], sf, dq + s, -0.1, 0.6)
        if not np.isnan(v) and v > best[0]:
            best = (v, s)
    return best[1], best[0]


def obs_with_qrs(all_data, ch_names, sfreq, qrs0, npcs, work):
    """fMRIB OBS in ONE MATLAB session for several npc using given 0-based QRS samples."""
    work.mkdir(parents=True, exist_ok=True)
    in_mat, q_mat = work / "in.mat", work / "dq.mat"
    sio.savemat(in_mat, {"data": all_data.astype(np.float32), "srate": float(sfreq),
                         "labels": np.array(ch_names, dtype=object)}, do_compression=True)
    sio.savemat(q_mat, {"qrs_lat": (np.asarray(qrs0, float) + 1).reshape(-1, 1)})
    ml = [f"addpath('{s8.EEGLAB_DIR.resolve()}');", f"addpath('{s8.FMRIB_DIR.resolve()}');", "eeglab nogui;",
          f"S = load('{in_mat.resolve()}');", "EEG = eeg_emptyset();", "EEG.data = double(S.data);",
          "EEG.srate = double(S.srate);", "EEG.nbchan = size(S.data,1);", "EEG.pnts = size(S.data,2);",
          "EEG.trials = 1;", "EEG.xmin = 0;", "EEG.xmax = (EEG.pnts-1)/EEG.srate;", "EEG.chanlocs = struct([]);",
          "for i = 1:EEG.nbchan", "  EEG.chanlocs(i).labels = char(S.labels{i});", "end",
          "EEG = eeg_checkset(EEG);", f"load('{q_mat.resolve()}', 'qrs_lat');",
          "for i = 1:length(qrs_lat)", "  EEG.event(end+1).type = 'qrs';", "  EEG.event(end).latency = qrs_lat(i);", "end",
          "EEG = eeg_checkset(EEG);"]
    for n in npcs:
        ml += [f"EEGc = pop_fmrib_pas(EEG, 'qrs', 'obs', {n});", "clean = single(EEGc.data);",
               f"save('{(work / f'out_npc{n}.mat').resolve()}', 'clean', '-v7');"]
    ml += ["exit(0);"]
    (work / "run.m").write_text("\n".join(ml) + "\n", encoding="ascii")
    r = subprocess.run([str(s8.MATLAB_BIN), "-nodesktop", "-nosplash", "-batch", f"run('{(work / 'run.m').resolve()}')"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = {}
    for n in npcs:
        p = work / f"out_npc{n}.mat"
        if p.exists():
            out[n] = np.array(sio.loadmat(p)["clean"], dtype=np.float64)
    if not out:
        print(r.stdout[-1500:])
    return out


def prep_eeg(x_uV, sf, chans_all, keep, filt=False):
    """(n_ch,n) uV -> V, pick keep, band 1-45 (idempotent), average ref."""
    idx = [chans_all.index(c) for c in keep]
    info = mne.create_info(keep, sf, "eeg")
    r = mne.io.RawArray(x_uV[idx] * 1e-6, info)
    r.filter(1, 45, verbose=False)
    r.set_eeg_reference("average", verbose=False)
    return r.get_data()


def metrics(x, d, sf, qrs_dima_in_x, keep, tag):
    occ = [keep.index(c) for c in OCC if c in keep]
    f, px = welch(x, sf, nperseg=int(4 * sf), axis=-1)
    _, pd_ = welch(d, sf, nperseg=int(4 * sf), axis=-1)
    lo = (f >= 1) & (f < 8)
    al = (f >= 8) & (f <= 13)
    return dict(variant=tag,
                rms_uV=float(np.sqrt(np.mean(x ** 2)) * 1e6),
                rms_ratio=float(np.sqrt(np.mean(x ** 2) / np.mean(d ** 2))),
                low1_8_ratio=float(px[:, lo].mean() / pd_[:, lo].mean()),
                occ_alpha_ratio=float(px[occ][:, al].mean() / pd_[occ][:, al].mean()),
                qrs_locked_uV=lock_rms(x, sf, qrs_dima_in_x),
                r_vs_dima=float(np.mean([np.corrcoef(x[i], d[i])[0, 1] for i in range(x.shape[0])])))


def run(sid, seg, outdir):
    dr, dq, cb = load_dima(sid, seg)
    raw = mne.io.read_raw_fif(ROOT / "data" / sid / "derivatives" / "02_resampled250" / seg / f"{seg}_250hz.fif", preload=True)
    sf = float(raw.info["sfreq"])
    names = list(raw.ch_names)
    all_uV = raw.get_data() * 1e6
    ecg_idx = names.index("ECG")
    keep = [c for c in dr.ch_names if c in names]
    pilot = [names.index(c) for c in PILOT if c in names]

    shift, pk = calibrate_shift(all_uV, pilot, dq, sf)
    print(f"[{sid}/{seg}] Dima QRS n={len(dq)} (HR~{60/np.median(np.diff(dq)/sf):.0f}); calibrated shift={shift} samples; "
          f"pre-BCG beat-locked RMS={pk:.2f} uV", flush=True)

    work = ROOT / "analysis_dima" / "_tmp" / f"{sid}_{seg}"
    if work.exists():
        shutil.rmtree(work)

    # MY qrs (fMRIB detect) + OBS for npc in {3, CB}
    npcs = sorted({3, cb})
    my_qrs, my_paths = s8.run_fmrib_bcg_matlab(all_uV, names, ecg_idx, sf, npcs, work / "mine")
    dq_my = dq + shift
    # How well do the two QRS sets agree (after alignment)?
    d_near = np.min(np.abs(my_qrs[None, :] - dq_my[:, None]), axis=1)
    agree = float(np.mean(d_near <= 0.04 * sf))
    print(f"   my fMRIB QRS n={len(my_qrs)} (HR~{60/np.median(np.diff(my_qrs)/sf):.0f}); "
          f"Dima beats found by my detector within 40ms: {agree*100:.0f}%; "
          f"my extra detections: {len(my_qrs) - agree * len(dq):.0f}", flush=True)

    # Dima QRS (shifted into my time) + OBS
    dres = obs_with_qrs(all_uV, names, sf, dq_my, npcs, work / "dima")

    n = min(dr.n_times, raw.n_times) - 2 * abs(shift) - 80
    s0 = 40 + max(shift, 0)
    D = dr.copy().pick(keep).filter(1, 45, verbose=False)
    D.set_eeg_reference("average", verbose=False)
    Dd = D.get_data()[:, 40:40 + n]
    qrs_in_x = dq - 40

    def al(x):   # align my array to Dima's clock
        return x[:, 40 + shift:40 + shift + n]

    rows = []
    rows.append(metrics(al(prep_eeg(all_uV, sf, names, keep)), Dd, sf, qrs_in_x, keep, "pre-BCG (input)"))
    for q_src, paths in (("myQRS", my_paths), ("dimaQRS", {k: None for k in dres})):
        for npc in npcs:
            if q_src == "myQRS":
                if npc not in paths:
                    continue
                clean = np.array(sio.loadmat(paths[npc])["clean"], dtype=np.float64)
            else:
                if npc not in dres:
                    continue
                clean = dres[npc]
            tag = f"{q_src} npc={npc}" + ("  [current pipeline]" if (q_src == "myQRS" and npc == 3) else "") + \
                  ("  [Dima setting]" if (q_src == "dimaQRS" and npc == cb) else "")
            rows.append(metrics(al(prep_eeg(clean, sf, names, keep)), Dd, sf, qrs_in_x, keep, tag))
    # my final stored outputs for reference
    for k, rel in (("stored 03_bcg", f"03_bcg/{seg}/{seg}_bcg_clean.fif"), ("stored 05_ica", f"05_ica/{seg}/{seg}_ica_clean.fif")):
        p = ROOT / "data" / sid / "derivatives" / rel
        if p.exists():
            r = mne.io.read_raw_fif(p, preload=True)
            rows.append(metrics(al(prep_eeg(r.get_data() * 1e6, sf, list(r.ch_names), keep)), Dd, sf, qrs_in_x, keep, k))
    df = pd.DataFrame(rows)
    df.insert(0, "seg", seg); df.insert(0, "sid", sid); df["dima_CB"] = cb
    df["shift"] = shift; df["qrs_agree_pct"] = agree * 100
    df["dima_rms_uV"] = float(np.sqrt(np.mean(Dd ** 2)) * 1e6)
    df["dima_locked_uV"] = lock_rms(Dd, sf, qrs_in_x)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"transplant_{sid}_{seg}.csv", index=False)
    shutil.rmtree(work, ignore_errors=True)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print(df[["variant", "rms_uV", "rms_ratio", "low1_8_ratio", "occ_alpha_ratio", "qrs_locked_uV", "r_vs_dima"]]
          .round(2).to_string(index=False))
    print(f"   Dima reference: rms={df.dima_rms_uV.iloc[0]:.2f} uV, beat-locked={df.dima_locked_uV.iloc[0]:.2f} uV\n", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="drone")
    ap.add_argument("--sid", nargs="+", required=True)
    a = ap.parse_args()
    for s in a.sid:
        try:
            run(s, a.seg, ROOT / "reports" / "dima_compare")
        except Exception as e:  # noqa
            import traceback; traceback.print_exc()
