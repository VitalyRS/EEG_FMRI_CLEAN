"""
cleanP01.py — Batch Stage P01: gradient (MRI) artifact removal + resample to 250 Hz
====================================================================================
For every requested subject, runs steps 01 -> 07 on ALL its named segments
(ec, eo, drone, lasertag, video), then down-samples the Bergen-cleaned 5000 Hz
data to 250 Hz and deletes the heavy intermediate files.

Per segment:
  01  detect MRI sessions          (once per subject -> segment_info.json)
  02  detect slice triggers
  03  trim dummy scans
  04  Optuna Bergen parameter search
  05  full 96-ch Bergen AAS        (5000 Hz .set, ~1 GB)
  06  spectra
  07  Bergen HTML report
  P01 resample 5000 -> 250 Hz      (polyphase FIR anti-alias, NO band-pass;
                                     filtering is left to step08)

Outputs kept per segment:
  data/<sid>/segments/<seg>/<seg>_bergen_250hz.set          EEGLAB (step08 picks it up)
  data/<sid>/derivatives/01_bergen_250hz/<seg>/<seg>_bergen_250hz_eeg.fif   MNE
  + all small JSON/PNG/HTML/NPZ reports from steps 01-07

Deleted after a subject is done (ONLY for segments whose 250 Hz output was verified):
  <seg>_bergen_optuna_*.set / .fdt     (5000 Hz Bergen output)
  temp_multich_crop.mat, run_full_optuna_clean.m
  trial*/clean_1ch.mat, trial*/bergen_clean.m, opt_trial_* scratch
Original recordings in EEG_ORIGINAL_ROOT are NEVER touched (read-only).

Usage:
  python cleanP01.py --subjects 1916
  python cleanP01.py --subjects 1916 1917 1918
  python cleanP01.py --all
  python cleanP01.py --subjects 1916 --keep-heavy     # do not delete 5000 Hz files
  python cleanP01.py --subjects 1916 --trials 30
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly
import mne

try:
    from .config import DATA_ROOT, DEFAULT_N_TRIALS, TARGET_SFREQ
    from .step01_detect_mri import detect_mri_sessions, discover_all_eeg_vhdrs
    from .step02_detect_slices import run_detect_slices
    from .step03_trim_dummy import trim_dummy_scans
    from .step04_optuna_tune import run_optuna_tuning
    from .step05_bergen_clean import clean_full_dataset
    from .step06_spectra_analysis import compute_spectra
    from .step07_html_report import generate_html_report
except ImportError:
    from config import DATA_ROOT, DEFAULT_N_TRIALS, TARGET_SFREQ
    from step01_detect_mri import detect_mri_sessions, discover_all_eeg_vhdrs
    from step02_detect_slices import run_detect_slices
    from step03_trim_dummy import trim_dummy_scans
    from step04_optuna_tune import run_optuna_tuning
    from step05_bergen_clean import clean_full_dataset
    from step06_spectra_analysis import compute_spectra
    from step07_html_report import generate_html_report

SEGMENT_ORDER = ["ec", "eo", "drone", "lasertag", "video"]
ECG_CH = "ECG"
OUT_SUFFIX = "_bergen_250hz"
DERIV_STAGE = "01_bergen_250hz"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(msg: str, ch: str = "="):
    print("\n" + ch * 80)
    print(f"  {msg}")
    print(ch * 80)


def _segment_dirs(sid: str) -> list[Path]:
    root = DATA_ROOT / sid / "segments"
    if not root.exists():
        return []
    dirs = [d for d in root.iterdir() if d.is_dir() and (d / "segment_info.json").exists()]
    return sorted(dirs, key=lambda d: (SEGMENT_ORDER.index(d.name)
                                       if d.name in SEGMENT_ORDER else 99, d.name))


def _out_paths(seg_dir: Path) -> tuple[Path, Path]:
    sid = seg_dir.parent.parent.name
    seg = seg_dir.name
    set_out = seg_dir / f"{seg}{OUT_SUFFIX}.set"
    fif_out = DATA_ROOT / sid / "derivatives" / DERIV_STAGE / seg / f"{seg}{OUT_SUFFIX}_eeg.fif"
    return set_out, fif_out


def _heavy_bergen_sets(seg_dir: Path) -> list[Path]:
    """5000 Hz Bergen outputs from step05 (never the 250 Hz file)."""
    return [p for p in seg_dir.glob("*bergen*optuna*.set") if OUT_SUFFIX not in p.name]


def _load_eeglab_set(set_path: Path) -> tuple[np.ndarray, float, list[str]]:
    """Load EEGLAB .set (MAT v5/v7, data inline or in .fdt). Returns (data_uV float32, sfreq, labels)."""
    m = sio.loadmat(str(set_path), squeeze_me=True, struct_as_record=True)
    if "EEG" in m:                       # older 'EEG' struct layout
        eeg = m["EEG"]
        get = lambda k: eeg[k].item() if hasattr(eeg[k], "item") else eeg[k]
    else:
        get = lambda k: m[k]
    sfreq = float(get("srate"))
    nbchan = int(get("nbchan"))
    pnts = int(get("pnts"))
    data = get("data")
    if isinstance(data, str):            # data stored in a separate .fdt
        fdt = set_path.with_name(data)
        data = np.fromfile(fdt, dtype="<f4").reshape((nbchan, pnts), order="F")
    data = np.asarray(data, dtype=np.float32).reshape(nbchan, pnts)
    cl = get("chanlocs")
    if getattr(cl, "dtype", None) is not None and cl.dtype.names and "labels" in cl.dtype.names:
        labels = [str(c["labels"]) for c in np.atleast_1d(cl).flat]
    else:
        labels = [f"Ch{i + 1}" for i in range(nbchan)]
    return data, sfreq, labels


def _save_eeglab_set(path: Path, data_uv: np.ndarray, sfreq: float, labels: list[str], setname: str):
    """Minimal EEGLAB-compatible .set (fields at top level, data inline, µV)."""
    nbchan, pnts = data_uv.shape
    chanlocs = np.zeros((nbchan,), dtype=[("labels", object), ("type", object)])
    for i, lab in enumerate(labels):
        chanlocs[i]["labels"] = lab
        chanlocs[i]["type"] = "ECG" if lab == ECG_CH else "EEG"
    empty = np.zeros((0, 0))
    sio.savemat(str(path), {
        "setname": setname, "filename": path.name, "filepath": str(path.parent),
        "nbchan": float(nbchan), "trials": 1.0, "pnts": float(pnts),
        "srate": float(sfreq), "xmin": 0.0, "xmax": (pnts - 1) / sfreq,
        "times": (np.arange(pnts) / sfreq * 1000.0),
        "data": data_uv.astype(np.float32),
        "chanlocs": chanlocs, "ref": "common",
        "event": empty, "urevent": empty, "epoch": empty,
        "icawinv": empty, "icasphere": empty, "icaweights": empty, "icaact": empty,
        "comments": "Bergen AAS (5000 Hz) -> resample_poly to 250 Hz (cleanP01.py)",
    }, do_compression=False, oned_as="row")


# ─────────────────────────────────────────────────────────────────────────────
# Resample stage
# ─────────────────────────────────────────────────────────────────────────────

def resample_segment(seg_dir: Path, target_sfreq: float = TARGET_SFREQ, force: bool = False) -> Path:
    set_out, fif_out = _out_paths(seg_dir)
    if not force and set_out.exists() and fif_out.exists():
        print(f"  [SKIP] 250 Hz output already exists: {set_out.name}")
        return set_out

    heavy = _heavy_bergen_sets(seg_dir)
    if not heavy:
        raise FileNotFoundError(f"No 5000 Hz Bergen .set in {seg_dir} (step05 output missing).")
    src = max(heavy, key=lambda p: p.stat().st_mtime)
    print(f"  Loading Bergen 5000 Hz: {src.name} ({src.stat().st_size / 2**20:.0f} MB)")
    data, sfreq, labels = _load_eeglab_set(src)

    frac = Fraction(target_sfreq / sfreq).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    print(f"  Resampling {sfreq:.0f} -> {target_sfreq:.0f} Hz (polyphase up={up}, down={down}) "
          f"for {data.shape[0]} ch x {data.shape[1]} samples ...")
    n_out = int(np.ceil(data.shape[1] * up / down))
    out = np.empty((data.shape[0], n_out), dtype=np.float32)
    for i in range(data.shape[0]):   # channel-wise to keep RAM low
        out[i] = resample_poly(data[i].astype(np.float64), up, down)[:n_out]
    del data
    gc.collect()

    # EEGLAB .set (in segment dir -> found by step08 _find_bergen_set)
    _save_eeglab_set(set_out, out, target_sfreq, labels, setname=set_out.stem)
    # MNE .fif (Volts)
    fif_out.parent.mkdir(parents=True, exist_ok=True)
    ch_types = ["ecg" if n == ECG_CH else "eeg" for n in labels]
    info = mne.create_info(labels, target_sfreq, ch_types)
    raw = mne.io.RawArray(out.astype(np.float64) * 1e-6, info, verbose=False)
    raw.save(fif_out, overwrite=True, verbose=False)

    meta = {"source_5000hz_set": src.name, "sfreq_in": sfreq, "sfreq_out": target_sfreq,
            "method": f"scipy.signal.resample_poly(up={up}, down={down})",
            "n_channels": len(labels), "n_samples": int(n_out),
            "duration_sec": n_out / target_sfreq, "set": str(set_out), "fif": str(fif_out)}
    (seg_dir / "resample250_info.json").write_text(json.dumps(meta, indent=2))
    print(f"  Saved: {set_out.name} ({set_out.stat().st_size / 2**20:.1f} MB)")
    print(f"  Saved: {fif_out} ({fif_out.stat().st_size / 2**20:.1f} MB)")
    del out, raw
    gc.collect()
    return set_out


def _verify_250(seg_dir: Path) -> bool:
    set_out, fif_out = _out_paths(seg_dir)
    if not (set_out.exists() and fif_out.exists()):
        return False
    try:
        raw = mne.io.read_raw_fif(fif_out, preload=False, verbose=False)
        ok = abs(raw.info["sfreq"] - TARGET_SFREQ) < 1e-6 and raw.n_times > 1000
        m = sio.loadmat(str(set_out), variable_names=["srate", "pnts"], squeeze_me=True)
        ok = ok and abs(float(m["srate"]) - TARGET_SFREQ) < 1e-6 and int(m["pnts"]) == raw.n_times
        return bool(ok)
    except Exception as ex:
        print(f"  [WARN] verification failed for {seg_dir.name}: {ex}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────

def cleanup_heavy(seg_dir: Path) -> float:
    """Delete heavy 5000 Hz intermediates. Returns freed MB. Only inside DATA_ROOT."""
    seg_dir = seg_dir.resolve()
    data_root = DATA_ROOT.resolve()
    assert data_root in seg_dir.parents, f"Refusing to delete outside {data_root}: {seg_dir}"

    targets: list[Path] = []
    for s in _heavy_bergen_sets(seg_dir):
        targets += [s, s.with_suffix(".fdt")]
    targets += [seg_dir / "temp_multich_crop.mat", seg_dir / "run_full_optuna_clean.m"]
    targets += list(seg_dir.glob("opt_trial_*"))
    for tdir in seg_dir.glob("trial*"):
        if tdir.is_dir():
            targets += [tdir / "clean_1ch.mat", tdir / "bergen_clean.m"]
    targets += list(seg_dir.glob("*_1ch*.mat"))   # Optuna single-channel extract, if any

    freed = 0
    for p in targets:
        if not p.exists():
            continue
        if OUT_SUFFIX in p.name:          # never delete the 250 Hz result
            continue
        size = p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
        freed += size
        print(f"    - deleted {p.relative_to(data_root)} ({size / 2**20:.1f} MB)")
    return freed / 2**20


# ─────────────────────────────────────────────────────────────────────────────
# Per-segment / per-subject drivers
# ─────────────────────────────────────────────────────────────────────────────

def process_segment(seg_dir: Path, n_trials: int, force_resample: bool = False,
                    force: bool = False):
    seg = seg_dir.name
    set_out, fif_out = _out_paths(seg_dir)

    # --force: recompute from Bergen (step05) so a changed step05 (e.g. edge
    # padding) is applied even when the 250 Hz output already exists. We drop the
    # stale 250 Hz files so nothing downstream reuses them.
    if force:
        for p in (set_out, fif_out):
            if p.exists():
                p.unlink()
        force_resample = True

    # If the 250 Hz result already exists and the 5000 Hz set was already deleted,
    # steps 05-07 must NOT re-run (they would recompute the heavy file).
    if set_out.exists() and fif_out.exists() and not _heavy_bergen_sets(seg_dir) and not force_resample:
        print(f"  [SKIP] {seg}: 250 Hz output present, heavy files already removed.")
        return

    _banner(f"[{seg}] step02 detect slices", "-")
    run_detect_slices(seg_dir)
    _banner(f"[{seg}] step03 trim dummy scans", "-")
    trim_dummy_scans(seg_dir)
    _banner(f"[{seg}] step04 Optuna Bergen ({n_trials} trials)", "-")
    if (seg_dir / "optuna_best_params.json").exists():
        print("  [SKIP] optuna_best_params.json already present.")
    else:
        run_optuna_tuning(seg_dir, n_trials=n_trials)
    _banner(f"[{seg}] step05 Bergen AAS (5000 Hz)", "-")
    clean_full_dataset(seg_dir, force=force)
    _banner(f"[{seg}] step06 spectra", "-")
    compute_spectra(seg_dir)
    _banner(f"[{seg}] step07 HTML report", "-")
    html = generate_html_report(seg_dir)
    print(f"  Report: {html}")
    _banner(f"[{seg}] P01 resample -> {TARGET_SFREQ:.0f} Hz", "-")
    resample_segment(seg_dir, force=force_resample)


def process_subject(sid: str, vhdrs: dict[str, Path], n_trials: int,
                    keep_heavy: bool, force_resample: bool,
                    segments_filter: list[str] | None = None, force: bool = False) -> dict:
    t0 = time.time()
    _banner(f"SUBJECT {sid}", "#")
    summary = {"subject": sid, "ok": [], "failed": [], "freed_mb": 0.0}

    # step01 (once per subject)
    if not _segment_dirs(sid):
        vhdr = vhdrs.get(sid)
        if vhdr is None:
            summary["failed"].append(("step01", "no valid raw .vhdr/.eeg found"))
            print(f"  [ERROR] No valid raw recording for {sid}; skipping subject.")
            return summary
        detect_mri_sessions(vhdr_path=vhdr, subject_id=sid, write_report=True)
    else:
        print("  [SKIP] step01: segment_info.json already present.")

    seg_dirs = _segment_dirs(sid)
    if segments_filter:
        seg_dirs = [d for d in seg_dirs if d.name in segments_filter]
    print(f"  Segments: {[d.name for d in seg_dirs]}")
    for seg_dir in seg_dirs:
        try:
            process_segment(seg_dir, n_trials, force_resample, force=force)
            summary["ok"].append(seg_dir.name)
        except Exception as ex:
            summary["failed"].append((seg_dir.name, str(ex)))
            print(f"\n  [FAILED] {sid}/{seg_dir.name}: {ex}")
            traceback.print_exc()
        gc.collect()

    # Cleanup only after all segments of the subject were attempted,
    # and only for segments whose 250 Hz output was verified.
    if not keep_heavy:
        _banner(f"CLEANUP heavy files for subject {sid}", "-")
        for seg_dir in seg_dirs:
            if _verify_250(seg_dir):
                summary["freed_mb"] += cleanup_heavy(seg_dir)
            else:
                print(f"  [KEEP] {seg_dir.name}: 250 Hz output not verified -> heavy files kept.")
        print(f"  Freed {summary['freed_mb'] / 1024:.2f} GB for subject {sid}")

    summary["minutes"] = (time.time() - t0) / 60
    return summary


def main():
    ap = argparse.ArgumentParser(description="P01: steps 01-07 (gradient artifact) + resample to 250 Hz, then cleanup")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--subjects", nargs="+", help="Subject IDs, e.g. 1916 1917")
    g.add_argument("--all", action="store_true", help="All subjects found in EEG_ORIGINAL_ROOT")
    ap.add_argument("--trials", type=int, default=DEFAULT_N_TRIALS, help="Optuna Bergen trials per segment")
    ap.add_argument("--keep-heavy", action="store_true", help="Do NOT delete 5000 Hz intermediates")
    ap.add_argument("--force-resample", action="store_true", help="Recompute the 250 Hz files")
    ap.add_argument("--segments", nargs="+", default=None,
                    help="Only process these segments (e.g. drone lasertag video)")
    ap.add_argument("--force", action="store_true",
                    help="Recompute from Bergen (step05) even if 250 Hz output exists (applies step05 changes)")
    args = ap.parse_args()

    vhdrs = discover_all_eeg_vhdrs()
    subjects = sorted(vhdrs) if args.all else args.subjects
    print(f"[cleanP01] Subjects to process: {subjects}  segments={args.segments or 'all'}  force={args.force}")

    results = []
    for sid in subjects:
        try:
            results.append(process_subject(sid, vhdrs, args.trials, args.keep_heavy, args.force_resample,
                                           segments_filter=args.segments, force=args.force))
        except Exception as ex:
            traceback.print_exc()
            results.append({"subject": sid, "ok": [], "failed": [("subject", str(ex))], "freed_mb": 0.0})

    _banner("cleanP01 SUMMARY", "#")
    for r in results:
        print(f"  {r['subject']}: ok={r['ok']}  failed={[f[0] for f in r['failed']]}  "
              f"freed={r['freed_mb'] / 1024:.2f} GB  time={r.get('minutes', 0):.1f} min")
        for name, err in r["failed"]:
            print(f"      ! {name}: {err}")


if __name__ == "__main__":
    main()
