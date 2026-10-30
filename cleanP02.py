"""
cleanP02.py — Batch Stage P02: BCG removal + ICA cleaning (steps 08 -> 10 -> 11)
================================================================================
Continues after cleanP01.py. For every requested subject / segment that has the
250 Hz Bergen-cleaned file produced by cleanP01:

  data/<sid>/segments/<seg>/<seg>_bergen_250hz.set

runs:
  08  band-pass (FILTER_HP..FILTER_LP) + BCG removal (EEGLAB fMRIB OBS, npc sweep)
        -> data/<sid>/derivatives/03_bcg/<seg>/<seg>_bcg_clean.fif (+ metrics/html)
  10  ICA-stage parameter search (clean_rawdata grid, ASR BurstCriterion, ICLabel thr)
        -> data/<sid>/segments/<seg>/optuna_ica_best_params.json
  11  clean_rawdata + ASR + runica + ICLabel + IC rejection + interp + avg-ref
        -> data/<sid>/derivatives/05_ica/<seg>/<seg>_ica_clean.fif (+ .set, metrics, html)

All cleaning parameters live in config/cleanP02.yaml (see comments there). They
are injected into the step08/10/11 modules before each segment, so per-segment /
per-subject overrides are possible.

Incremental behaviour (per segment, per step):
  * a step runs if its outputs are missing, older than its input, forced, or
    (run.rerun_on_config_change) its config section changed since the last run;
  * a re-run step automatically invalidates the downstream steps (mtime cascade);
  * state + effective config is stored in segments/<seg>/cleanP02_state.json.

Usage:
  python cleanP02.py --subjects 1916
  python cleanP02.py --subjects 1916 1917 --segments ec eo
  python cleanP02.py --all
  python cleanP02.py --all --dry-run                       # show what would run
  python cleanP02.py --subjects 1916 --force bcg           # recompute BCG (+ downstream)
  python cleanP02.py --subjects 1916 --set bcg.fixed_npc=4 --set ica_opt.duration_sec=300
  python cleanP02.py --all --report-only                   # rebuild summary CSV only
  python cleanP02.py --subjects 1916 --config config/my_variant.yaml
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio
import yaml
import mne

try:
    from .config import DATA_ROOT, PROJECT_ROOT, MATLAB_BIN
    from . import step08_bcg as s08
    from . import step10_optuna_ica as s10
    from . import step11_ica_final as s11
except ImportError:
    from config import DATA_ROOT, PROJECT_ROOT, MATLAB_BIN
    import step08_bcg as s08
    import step10_optuna_ica as s10
    import step11_ica_final as s11

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "cleanP02.yaml"
IN_SUFFIX = "_bergen_250hz"
STEPS = ["bcg", "ica_opt", "ica_final"]
SEGMENT_ORDER = ["ec", "eo", "drone", "lasertag", "video"]
ICA_PARAM_KEYS = ["flatline_crit", "channel_crit", "line_crit", "iclabel_thresh", "burst_crit", "ica_hp_hz",
                  "thresh_eye", "thresh_heart", "thresh_muscle", "thresh_line_chan", "thresh_other", "other_brain_max", "r_ecg_thresh",
                  "max_brain_prob", "max_rej_fraction", "manual_reject_ics", "rejection_mode"]


# ─────────────────────────────────────────────────────────────────────────────
# Value converters (YAML -> what the step modules expect)
# ─────────────────────────────────────────────────────────────────────────────

def _burst(v):
    """'off' / off(False) / None -> 'off', otherwise float SD cutoff."""
    if v is None or v is False or (isinstance(v, str) and v.strip().lower() == "off"):
        return "off"
    return float(v)


def _crit(v):
    """'off' / off(False) / None -> 'off', otherwise float criterion."""
    if v is None or v is False or (isinstance(v, str) and v.strip().lower() == "off"):
        return "off"
    return float(v)


def _band(v):
    lo, hi = float(v[0]), float(v[1])
    if not lo < hi:
        raise ValueError(f"band must be [lo, hi] with lo < hi, got {v}")
    return (lo, hi)


_floats = lambda v: [float(x) for x in v]
_ints = lambda v: [int(x) for x in v]
_strs = lambda v: [str(x) for x in v]
_opt_int = lambda v: None if v in (None, 0, False) else int(v)

# config path -> [(module, attribute, converter)]
PARAM_MAP: dict[tuple, list] = {
    ("bands", "alpha"): [(s08, "ALPHA_BAND", _band), (s10, "ALPHA_BAND", _band), (s11, "ALPHA_BAND", _band)],
    ("filter", "hp"): [(s08, "FILTER_HP", float)],
    ("filter", "lp"): [(s08, "FILTER_LP", float)],
    ("filter", "notch"): [(s08, "FILTER_NOTCH", lambda v: float(v) if (v and str(v).lower() != "off") else None)],
    ("trim", "enabled"): [(s08, "TRIM_ENABLED", bool)],
    ("trim", "start_sec"): [(s08, "TRIM_START_SEC", float)],
    ("trim", "end_sec"): [(s08, "TRIM_END_SEC", float)],
    ("bcg", "npc_grid"): [(s08, "NPC_GRID", _ints)],
    ("bcg", "alpha_gate_min"): [(s08, "ALPHA_GATE_MIN", float)],
    ("bcg", "min_cardiac_suppr"): [(s08, "MIN_CARDIAC_SUPPR", float)],
    ("bcg", "alpha_warn_min"): [(s08, "ALPHA_WARN_MIN", float)],
    ("bcg", "cardiac_band"): [(s08, "CARDIAC_BAND", _band)],
    ("bcg", "pilot_channels"): [(s08, "PILOT_CHANNELS", _strs)],
    ("bcg", "eval_channels"): [(s08, "EVAL_CHANNELS", _strs)],
    ("bcg", "residual_aas"): [(s08, "RESIDUAL_AAS", bool)],
    ("bcg", "aas_neighbors"): [(s08, "AAS_NEIGHBORS", int)],
    ("bcg", "residual_obs_npc_grid"): [(s08, "RESIDUAL_OBS_NPC_GRID", _ints)],
    ("bcg", "residual_alpha_gate"): [(s08, "RESIDUAL_ALPHA_GATE", float)],
    ("ica_opt", "duration_sec"): [(s10, "OPTUNA_DURATION_SEC", float)],
    ("ica_opt", "max_channel_removal_fraction"): [(s10, "MAX_CHANNEL_REMOVAL_FRACTION", float)],
    ("ica_opt", "flatline_grid"): [(s10, "FLATLINE_GRID", lambda v: [_crit(x) for x in v])],
    ("ica_opt", "channel_grid"): [(s10, "CHANNEL_GRID", lambda v: [_crit(x) for x in v])],
    ("ica_opt", "line_grid"): [(s10, "LINE_GRID", lambda v: [_crit(x) for x in v])],
    ("ica_opt", "burst_grid"): [(s10, "BURST_GRID", lambda v: [_burst(x) for x in v])],
    ("ica_opt", "hf_artifact_band"): [(s10, "HF_ARTIFACT_BAND", _band)],
    ("ica_opt", "asr_min_hf_gain"): [(s10, "ASR_MIN_HF_GAIN", float)],
    ("ica_opt", "asr_min_alpha_ret"): [(s10, "ASR_MIN_ALPHA_RET", float)],
    ("ica_opt", "threshold_grid"): [(s10, "THRESHOLD_GRID", _floats)],
    ("ica_opt", "protected_channels"): [(s10, "PROTECTED_CHANNELS", lambda v: set(_strs(v)))],
    ("iclabel", "other_brain_max"): [(s10, "OTHER_BRAIN_MAX", float), (s11, "OTHER_BRAIN_MAX", float)],
    ("iclabel", "ica_pca_max"): [(s10, "ICA_PCA_MAX", _opt_int), (s11, "ICA_PCA_MAX", _opt_int)],
    ("iclabel", "backend"): [(s10, "ICLABEL_BACKEND", str), (s11, "ICLABEL_BACKEND", str)],
    ("ica_final", "defaults", "flatline_crit"): [(s11, "FLATLINE_CRIT", _crit)],
    ("ica_final", "defaults", "channel_crit"): [(s11, "CHANNEL_CRIT", _crit)],
    ("ica_final", "defaults", "line_crit"): [(s11, "LINE_NOISE_CRIT", _crit)],
    ("ica_final", "defaults", "iclabel_thresh"): [(s11, "ICLABEL_THRESH", float)],
    ("ica_final", "defaults", "burst_crit"): [(s11, "ASR_BURST_CRIT", _burst)],
    ("ica_final", "defaults", "ica_hp_hz"): [(s11, "ICA_HP_HZ", float), (s10, "ICA_HP_HZ", float)],
    ("ica_final", "guards", "min_iclabel_thresh"): [(s11, "GUARD_MIN_ICLABEL_THRESH", float)],
    ("ica_final", "guards", "max_channel_crit"): [(s11, "GUARD_MAX_CHANNEL_CRIT", _crit)],
    ("ica_final", "guards", "min_channel_crit"): [(s11, "GUARD_MAX_CHANNEL_CRIT", _crit)],
    ("ica_final", "guards", "min_burst_crit"): [(s11, "GUARD_MIN_BURST_CRIT", float)],
    ("ica_final", "cardiac_qrs", "ratio_thresh"): [(s11, "CARDIAC_QRS_RATIO_THRESH", float)],
    ("ica_final", "cardiac_qrs", "max_ics"): [(s11, "CARDIAC_QRS_MAX_ICS", int)],
    ("ica_final", "cardiac_qrs", "max_brain"): [(s11, "CARDIAC_QRS_MAX_BRAIN", float)],
}

# Keys handled by this orchestrator (not injected into step modules).
ORCH_KEYS = {
    ("run", "segments"), ("run", "steps", "bcg"), ("run", "steps", "ica_opt"),
    ("run", "steps", "ica_final"), ("run", "force"), ("run", "rerun_on_config_change"),
    ("run", "continue_on_error"), ("bcg", "fixed_npc"),
    ("ica_final", "use_optimized_params"), ("ica_final", "force_params"),
    ("ica_final", "rejection_mode"),
    ("output", "export_set"), ("output", "report_dir"),
    ("cleanup", "delete_prebcg_fif"), ("cleanup", "delete_ica_opt_workdir"),
    ("qc", "min_bcg_cardiac_suppression"), ("qc", "min_bcg_alpha_retention"),
    ("qc", "min_final_alpha_retention"), ("qc", "max_channels_removed_frac"),
    ("qc", "max_ic_rejected_frac"), ("qc", "max_variance_drop"),
}
DICT_LEAVES = {("ica_final", "force_params"), ("overrides",)}

# Config sections each step depends on (hashed for rerun_on_config_change).
STEP_SECTIONS = {
    "bcg": ["bands", "filter", "trim", "bcg"],
    "ica_opt": ["bands", "ica_opt", "iclabel"],
    "ica_final": ["bands", "iclabel", "ica_final"],
}

# Snapshot of the original module values so per-segment overrides never leak.
_ORIGINALS = {(mod.__name__, attr): (mod, copy.deepcopy(getattr(mod, attr)))
              for targets in PARAM_MAP.values() for mod, attr, _ in targets}


# ─────────────────────────────────────────────────────────────────────────────
# Config handling
# ─────────────────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _get(cfg: dict, path: tuple, default=None):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _leaf_paths(d: dict, prefix=()):
    for k, v in d.items():
        p = prefix + (str(k),)
        if p in DICT_LEAVES or not isinstance(v, dict):
            yield p
        else:
            yield from _leaf_paths(v, p)


def validate_config(cfg: dict, where: str = "config") -> list[str]:
    """Warn about unknown keys (typos) and bad values. Returns list of warnings."""
    known = set(PARAM_MAP) | ORCH_KEYS | DICT_LEAVES
    warns = [f"{where}: unknown key '{'.'.join(p)}' (ignored)"
             for p in _leaf_paths(cfg) if p not in known]
    for k in _get(cfg, ("ica_final", "force_params"), {}) or {}:
        if k not in ICA_PARAM_KEYS:
            warns.append(f"{where}: ica_final.force_params.{k} unknown (allowed: {ICA_PARAM_KEYS})")
    for s in _get(cfg, ("run", "force"), []) or []:
        if s not in STEPS:
            warns.append(f"{where}: run.force contains unknown step '{s}' (allowed: {STEPS})")
    # dry conversion of every mapped value -> catches type errors early
    for path, targets in PARAM_MAP.items():
        v = _get(cfg, path)
        if v is not None:
            for _, attr, conv in targets:
                try:
                    conv(v)
                except Exception as ex:
                    warns.append(f"{where}: {'.'.join(path)}={v!r} invalid for {attr}: {ex}")
    for scope in ("segments", "subjects", "subject_segments"):
        for key, block in (_get(cfg, ("overrides", scope), {}) or {}).items():
            if isinstance(block, dict):
                warns += validate_config(block, f"overrides.{scope}.{key}")
    return warns


def load_config(path: Path, sets: list[str]) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for s in sets or []:
        if "=" not in s:
            raise SystemExit(f"--set expects key.path=value, got: {s}")
        key, val = s.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        node[parts[-1]] = yaml.safe_load(val)
    # normalise override keys (YAML turns 1921 into int)
    for scope in ("segments", "subjects", "subject_segments"):
        blk = _get(cfg, ("overrides", scope))
        if isinstance(blk, dict):
            cfg["overrides"][scope] = {str(k): (v or {}) for k, v in blk.items()}
    return cfg


def effective_config(cfg: dict, sid: str, seg: str) -> dict:
    base = {k: v for k, v in cfg.items() if k != "overrides"}
    ov = cfg.get("overrides") or {}
    for blk in ((ov.get("segments") or {}).get(seg),
                (ov.get("subjects") or {}).get(sid),
                (ov.get("subject_segments") or {}).get(f"{sid}/{seg}")):
        if blk:
            base = _deep_merge(base, blk)
    return base


def apply_to_modules(cfg: dict):
    """Reset step-module globals to their originals, then inject config values."""
    for (_, attr), (mod, val) in _ORIGINALS.items():
        setattr(mod, attr, copy.deepcopy(val))
    for path, targets in PARAM_MAP.items():
        v = _get(cfg, path)
        if v is None and path != ("iclabel", "ica_pca_max"):
            continue
        for mod, attr, conv in targets:
            setattr(mod, attr, conv(v))


def section_hash(cfg: dict, step: str) -> str:
    sub = {k: cfg.get(k) for k in STEP_SECTIONS[step]}
    return hashlib.sha1(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Paths / state
# ─────────────────────────────────────────────────────────────────────────────

class SegPaths:
    def __init__(self, sid: str, seg: str):
        self.sid, self.seg = sid, seg
        self.seg_dir = DATA_ROOT / sid / "segments" / seg
        deriv = DATA_ROOT / sid / "derivatives"
        self.in_set = self.seg_dir / f"{seg}{IN_SUFFIX}.set"
        self.prebcg_dir = deriv / "02_resampled250" / seg
        self.resamp_fif = self.prebcg_dir / f"{seg}_250hz.fif"
        self.bergen_fif = deriv / "01_bergen_250hz" / seg / f"{seg}_bergen_250hz_eeg.fif"
        self.bcg_dir = deriv / "03_bcg" / seg
        self.bcg_fif = self.bcg_dir / f"{seg}_bcg_clean.fif"
        self.bcg_html = self.bcg_dir / f"{seg}_bcg_report.html"
        self.bcg_metrics = self.bcg_dir / f"{seg}_bcg_metrics.json"
        self.opt_json = self.seg_dir / "optuna_ica_best_params.json"
        self.opt_png = self.seg_dir / "optuna_ica_result.png"
        self.opt_workdir = self.seg_dir / "optuna_ica_trials"
        self.ica_dir = deriv / "05_ica" / seg
        self.ica_fif = self.ica_dir / f"{seg}_ica_clean.fif"
        self.ica_set = self.ica_dir / f"{seg}_ica_clean.set"
        self.ica_html = self.ica_dir / f"{seg}_ica_report.html"
        self.ica_metrics = self.ica_dir / f"{seg}_ica_metrics.json"
        self.state = self.seg_dir / "cleanP02_state.json"

    def outputs(self, step: str) -> list[Path]:
        return {"bcg": [self.bcg_fif, self.bcg_html],
                "ica_opt": [self.opt_json, self.opt_png],
                "ica_final": [self.ica_fif, self.ica_html]}[step]


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


def _load_state(sp: SegPaths) -> dict:
    try:
        return json.loads(sp.state.read_text())
    except Exception:
        return {}


def _save_state(sp: SegPaths, state: dict):
    sp.state.parent.mkdir(parents=True, exist_ok=True)
    sp.state.write_text(json.dumps(state, indent=2, default=str))


def decide(sp: SegPaths, step: str, upstream_mtime: float, forced: bool,
           cfg_hash: str, state: dict, rerun_on_cfg: bool) -> tuple[bool, str]:
    outs = sp.outputs(step)
    if forced:
        return True, "forced"
    if not all(p.exists() for p in outs):
        return True, "output missing"
    if upstream_mtime and min(_mtime(p) for p in outs) < upstream_mtime:
        return True, "input is newer"
    old = (state.get(step) or {}).get("hash")
    if rerun_on_cfg and old is not None and old != cfg_hash:
        return True, f"config changed ({old} -> {cfg_hash})"
    return False, "up to date"


# ─────────────────────────────────────────────────────────────────────────────
# Step wrappers
# ─────────────────────────────────────────────────────────────────────────────

def run_bcg(sp: SegPaths, cfg: dict):
    fixed = _get(cfg, ("bcg", "fixed_npc"))
    npc_grid = [int(fixed)] if fixed else list(s08.NPC_GRID)
    in_set_path = sp.in_set if sp.in_set.exists() else None
    s08.run_bcg_pipeline(sp.seg_dir, npc_grid=npc_grid, force=True, input_set=in_set_path)


def run_ica_opt(sp: SegPaths, cfg: dict):
    s10.run_optuna_ica(sp.seg_dir, force=True)


def ica_final_override(cfg: dict) -> dict:
    fin = cfg.get("ica_final") or {}
    override = {}
    if "rejection_mode" in fin:
        override["rejection_mode"] = str(fin["rejection_mode"]).strip().lower()
    if not fin.get("use_optimized_params", True):
        override.update(fin.get("defaults") or {})
    override.update(fin.get("force_params") or {})
    for k in ("burst_crit", "channel_crit", "flatline_crit", "line_crit"):
        if k in override:
            override[k] = _crit(override[k])
    return {k: (override[k] if k in ("burst_crit", "channel_crit", "flatline_crit", "line_crit", "manual_reject_ics", "rejection_mode") else float(override[k]))
            for k in override if k in ICA_PARAM_KEYS}


def run_ica_final(sp: SegPaths, cfg: dict):
    s11.apply_optimized_ica(sp.seg_dir, force=True, params_override=ica_final_override(cfg) or None, cfg_snapshot=cfg)
    if _get(cfg, ("output", "export_set"), True):
        export_set(sp)
    s11.save_pipeline_info(
        sp.sid, sp.seg, sp.ica_dir, in_fif=sp.bcg_fif, out_fif=sp.ica_fif,
        metrics=_read_json(sp.ica_metrics),
        params_used=_read_json(sp.ica_metrics).get("params_used", {}),
        cfg=cfg
    )


def export_set(sp: SegPaths):
    """Write the final ICA-clean data as a minimal EEGLAB .set (µV, data inline)."""
    sp.ica_set.parent.mkdir(parents=True, exist_ok=True)
    raw = mne.io.read_raw_fif(sp.ica_fif, preload=True, verbose=False)
    data_uv = (raw.get_data() * 1e6).astype(np.float32)
    nbchan, pnts = data_uv.shape
    sfreq = float(raw.info["sfreq"])
    chanlocs = np.zeros((nbchan,), dtype=[("labels", object), ("type", object)])
    for i, lab in enumerate(raw.ch_names):
        chanlocs[i]["labels"] = lab
        chanlocs[i]["type"] = "EEG"
    empty = np.zeros((0, 0))
    sio.savemat(str(sp.ica_set), {
        "setname": sp.ica_set.stem, "filename": sp.ica_set.name, "filepath": str(sp.ica_set.parent),
        "nbchan": float(nbchan), "trials": 1.0, "pnts": float(pnts), "srate": sfreq,
        "xmin": 0.0, "xmax": (pnts - 1) / sfreq, "times": np.arange(pnts) / sfreq * 1000.0,
        "data": data_uv, "chanlocs": chanlocs, "ref": "average",
        "event": empty, "urevent": empty, "epoch": empty,
        "icawinv": empty, "icasphere": empty, "icaweights": empty, "icaact": empty,
        "comments": "Bergen AAS -> 250 Hz -> fMRIB OBS -> clean_rawdata/ASR -> ICA/ICLabel (cleanP02.py)",
    }, do_compression=False, oned_as="row")
    print(f"  Saved EEGLAB: {sp.ica_set} ({sp.ica_set.stat().st_size / 2**20:.1f} MB)")


STEP_FUNCS = {"bcg": run_bcg, "ica_opt": run_ica_opt, "ica_final": run_ica_final}


def cleanup_segment(sp: SegPaths, cfg: dict):
    data_root = DATA_ROOT.resolve()
    targets = []
    if _get(cfg, ("cleanup", "delete_prebcg_fif"), False):
        targets.append(sp.prebcg_dir)
    if _get(cfg, ("cleanup", "delete_ica_opt_workdir"), False):
        targets.append(sp.opt_workdir)
    for t in targets:
        t = t.resolve()
        if t.exists() and data_root in t.parents:
            shutil.rmtree(t, ignore_errors=True)
            print(f"    - deleted {t.relative_to(data_root)}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary / QC
# ─────────────────────────────────────────────────────────────────────────────

def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def collect_row(sp: SegPaths, cfg: dict) -> dict:
    b = _read_json(sp.bcg_metrics)
    m = _read_json(sp.ica_metrics)
    pu = m.get("params_used", {})
    rc = _read_json(DATA_ROOT / sp.sid / "derivatives" / "04_channels" / sp.seg / "removed_channels.json")
    n_ch = rc.get("n_total") or 95  # EEG channels without ECG
    n_ic = m.get("n_ic") or 0
    row = {
        "subject": sp.sid, "segment": sp.seg,
        "bcg_npc": b.get("best_npc"), "bcg_cardiac_suppr": b.get("cardiac_suppression"),
        # brain-alpha (QRS-locked component removed) if available, else raw — the raw
        # ratio counts BCG pseudo-alpha removal as loss and false-flags low-alpha subjects
        "bcg_alpha_ret": (b.get("alpha_retention_brain") if b.get("alpha_retention_brain") is not None
                          else b.get("alpha_retention")),
        "bcg_alpha_raw": b.get("alpha_retention"), "bcg_alpha_flag": b.get("alpha_flag"),
        "n_rpeaks": b.get("n_rpeaks"),
        **{f"ica_{k}": pu.get(k) for k in ICA_PARAM_KEYS},
        "n_ch_removed": m.get("n_channels_removed"),
        "removed_channels": " ".join(m.get("removed_channels", []) or []),
        "n_ic": m.get("n_ic"), "n_ic_rejected": m.get("n_ic_rejected"),
        "ic_rej_frac": (m.get("n_ic_rejected", 0) / n_ic) if n_ic else None,
        "final_alpha_ret": m.get("alpha_retention"), "final_var_drop": m.get("variance_drop"),
    }
    q = cfg.get("qc") or {}
    reasons = []

    def chk(val, thr, cmp, label):
        if val is not None and thr is not None and cmp(val, thr):
            reasons.append(label)
    chk(row["bcg_cardiac_suppr"], q.get("min_bcg_cardiac_suppression"), lambda a, t: a < t, "low_cardiac_suppr")
    chk(row["bcg_alpha_ret"], q.get("min_bcg_alpha_retention"), lambda a, t: a < t, "bcg_alpha_loss")
    chk(row["final_alpha_ret"], q.get("min_final_alpha_retention"), lambda a, t: a < t, "ica_alpha_loss")
    chk((row["n_ch_removed"] / n_ch) if row["n_ch_removed"] is not None else None,
        q.get("max_channels_removed_frac"), lambda a, t: a > t, "many_channels_removed")
    chk(row["ic_rej_frac"], q.get("max_ic_rejected_frac"), lambda a, t: a > t, "many_ics_rejected")
    chk(row["final_var_drop"], q.get("max_variance_drop"), lambda a, t: a > t, "high_variance_drop")
    row["qc"] = ("REVIEW: " + ",".join(reasons)) if reasons else ("OK" if m else "INCOMPLETE")
    return row


def write_summary(rows: list[dict], report_dir: Path, ts: str):
    if not rows:
        return
    report_dir.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    for name in (f"summary_{ts}.csv", "summary_latest.csv"):
        with open(report_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\n  Summary CSV: {report_dir / f'summary_{ts}.csv'}")


def _fmt(v, pct=False):
    if v is None:
        return "  -  "
    return f"{v * 100:5.1f}%" if pct else str(v)


def print_summary(rows: list[dict]):
    print(f"\n  {'subj':<5} {'seg':<9} {'status':<9} {'npc':>3} {'card':>6} {'aBCG':>6} "
          f"{'ch-':>3} {'ICrej':>7} {'aICA':>6} {'min':>5}  qc")
    for r in rows:
        ic = f"{r.get('n_ic_rejected')}/{r.get('n_ic')}" if r.get("n_ic") else "-"
        print(f"  {r['subject']:<5} {r['segment']:<9} {r.get('status', ''):<9} "
              f"{_fmt(r.get('bcg_npc')):>3} {_fmt(r.get('bcg_cardiac_suppr'), True):>6} "
              f"{_fmt(r.get('bcg_alpha_ret'), True):>6} {_fmt(r.get('n_ch_removed')):>3} {ic:>7} "
              f"{_fmt(r.get('final_alpha_ret'), True):>6} {r.get('minutes', 0):5.1f}  {r.get('qc', '')}")
        if r.get("error"):
            print(f"        ! {r['error']}")


# ─────────────────────────────────────────────────────────────────────────────
# Drivers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(msg: str, ch: str = "="):
    print("\n" + ch * 80 + f"\n  {msg}\n" + ch * 80)


def process_segment(sid: str, seg: str, cfg_all: dict, cli_force: set, dry_run: bool) -> dict:
    sp = SegPaths(sid, seg)
    cfg = effective_config(cfg_all, sid, seg)
    t0 = time.time()
    row = {"subject": sid, "segment": seg, "status": "", "minutes": 0.0}
    has_input = sp.in_set.exists() or sp.resamp_fif.exists() or sp.bergen_fif.exists() or sp.bcg_fif.exists()
    if not has_input:
        row.update(status="no_input", qc="INCOMPLETE", error=f"Input data missing for {sid}/{seg} (run cleanP01)")
        print(f"  [SKIP] {sid}/{seg}: input not found.")
        return row

    sp.seg_dir.mkdir(parents=True, exist_ok=True)

    apply_to_modules(cfg)
    steps_on = {s: bool(_get(cfg, ("run", "steps", s), True)) for s in STEPS}
    force = set(_get(cfg, ("run", "force"), []) or []) | cli_force
    rerun_cfg = bool(_get(cfg, ("run", "rerun_on_config_change"), True))
    use_opt = bool(_get(cfg, ("ica_final", "use_optimized_params"), True))
    state = _load_state(sp)

    in_mtime = max(_mtime(sp.in_set), _mtime(sp.resamp_fif), _mtime(sp.bergen_fif))
    upstream = {"bcg": in_mtime}
    ran = []
    for step in STEPS:
        if step == "ica_opt":
            upstream[step] = _mtime(sp.bcg_fif) if "bcg" not in ran else float("inf")
        elif step == "ica_final":
            u = _mtime(sp.bcg_fif) if "bcg" not in ran else float("inf")
            if use_opt:
                u = max(u, _mtime(sp.opt_json) if "ica_opt" not in ran else float("inf"))
            upstream[step] = u
        h = section_hash(cfg, step)
        if not steps_on[step]:
            print(f"  [{step:9s}] disabled in config")
            continue
        need, why = decide(sp, step, upstream[step] if upstream[step] != float("inf") else 0,
                           step in force, h, state, rerun_cfg)
        if upstream[step] == float("inf"):
            need, why = True, "upstream step re-run"
        if not need:
            print(f"  [{step:9s}] SKIP ({why})")
            continue
        print(f"  [{step:9s}] RUN  ({why})" + ("   [dry-run]" if dry_run else ""))
        if dry_run:
            ran.append(step)
            continue
        _banner(f"{sid}/{seg}: {step}", "-")
        ts = time.time()
        STEP_FUNCS[step](sp, cfg)
        ran.append(step)
        state[step] = {"hash": h, "finished": datetime.now().isoformat(timespec="seconds"),
                       "minutes": round((time.time() - ts) / 60, 2)}
        state["effective_config"] = cfg
        _save_state(sp, state)
        gc.collect()

    if not dry_run:
        cleanup_segment(sp, cfg)
    row.update(collect_row(sp, cfg))
    row["status"] = "dry-run" if dry_run else ("done" if ran else "up-to-date")
    row["steps_run"] = "+".join(ran)
    row["minutes"] = (time.time() - t0) / 60
    return row


def discover_subjects() -> list[str]:
    return sorted(d.name for d in DATA_ROOT.iterdir()
                  if d.is_dir() and d.name not in ("logs", "copy.sh") and (
                      any((d / "segments").glob(f"*/*{IN_SUFFIX}.set")) or
                      (d / "derivatives" / "02_resampled250").exists() or
                      (d / "derivatives" / "01_bergen_250hz").exists() or
                      (d / "derivatives" / "03_bcg").exists()
                  ))


class _Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.out.flush()
        self.f.flush()


def main():
    ap = argparse.ArgumentParser(description="P02: BCG (step08) + ICA optimisation (step10) + final ICA (step11)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--subjects", nargs="+", help="Subject IDs, e.g. 1916 1917")
    g.add_argument("--all", action="store_true", help="All subjects that have *_bergen_250hz.set")
    ap.add_argument("--segments", nargs="+", help="Override run.segments (e.g. ec eo)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config file")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="Override a config value, e.g. --set bcg.fixed_npc=4 (repeatable)")
    ap.add_argument("--force", nargs="*", choices=STEPS, default=None,
                    help="Force steps (no names = all steps)")
    ap.add_argument("--recalc", action="store_true", help="Recalculate all steps (alias for --force)")
    ap.add_argument("--dry-run", action="store_true", help="Only show what would be run")
    ap.add_argument("--report-only", action="store_true", help="Only rebuild the summary from existing outputs")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if args.segments:
        cfg.setdefault("run", {})["segments"] = args.segments
    if args.recalc:
        cli_force = set(STEPS)
    else:
        cli_force = set(STEPS) if args.force == [] else set(args.force or [])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = PROJECT_ROOT / (_get(cfg, ("output", "report_dir")) or "reports/cleanP02")
    if not (args.dry_run or args.report_only):
        sys.stdout = _Tee(report_dir / f"cleanP02_{ts}.log")

    warns = validate_config(cfg)
    for w in warns:
        print(f"[CONFIG WARN] {w}")
    if any("invalid" in w for w in warns):
        raise SystemExit("Config has invalid values, aborting.")

    if args.all or (args.subjects and len(args.subjects) == 1 and args.subjects[0].lower() == "all"):
        subjects = discover_subjects()
    else:
        subjects = [str(s) for s in args.subjects]
    segments = _get(cfg, ("run", "segments")) or SEGMENT_ORDER
    print(f"[cleanP02] config={args.config}  subjects={subjects}  segments={segments}  "
          f"force={sorted(cli_force) or '-'}")

    if not (args.dry_run or args.report_only) and not Path(MATLAB_BIN).exists():
        raise SystemExit(f"MATLAB not found: {MATLAB_BIN}")

    rows = []
    t_all = time.time()
    try:
        for sid in subjects:
            _banner(f"SUBJECT {sid}", "#")
            for seg in segments:
                if args.report_only:
                    sp = SegPaths(sid, seg)
                    if sp.seg_dir.exists():
                        rows.append({"status": "report", **collect_row(sp, effective_config(cfg, sid, seg))})
                    continue
                print(f"\n>>> {sid}/{seg}")
                try:
                    rows.append(process_segment(sid, seg, cfg, cli_force, args.dry_run))
                except Exception as ex:
                    traceback.print_exc()
                    sp = SegPaths(sid, seg)
                    rows.append({**collect_row(sp, effective_config(cfg, sid, seg)),
                                 "status": "FAILED", "error": f"{type(ex).__name__}: {ex}"})
                    if not _get(cfg, ("run", "continue_on_error"), True):
                        raise
                gc.collect()
    except KeyboardInterrupt:
        print("\n[cleanP02] Interrupted by user - writing partial summary.")

    _banner(f"cleanP02 SUMMARY ({(time.time() - t_all) / 60:.1f} min)", "#")
    print_summary(rows)
    if not args.dry_run:
        write_summary(rows, report_dir, ts)


if __name__ == "__main__":
    main()
