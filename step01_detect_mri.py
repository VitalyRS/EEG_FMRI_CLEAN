"""
STEP 01: Detect MRI Scanning Sessions and Assign Session Names for All Subjects
================================================================================
Strategy:
  1. Automatically discovers subjects from EEG_ORIGINAL_ROOT (.../eeg-fmri/original/XXXXEEG/)
     or local data/XXXX/raw/eeg96/.
  2. Computes RMS envelope over control channels (Fp1, Fp2, Fz, Cz, Pz)
     on the continuous recording and detects all MRI-active intervals
     above a threshold (each interval = one fMRI session).
  3. Skips the FIRST detected segment (calibration / localizer run with no rp file).
  4. Assigns semantic session names in order:
       ec (301) → eo (401) → drone (501) → lasertag (601) → video (701)
  5. Matches with SPM rp_*.txt files from FMRI_ROOT/<subject>FMRI/.
     Verifies duration: rp_volumes × TR ≈ RMS_duration − dummy_dur (30 s).
  6. Saves lightweight metadata (segment_info.json) under:
       data/<subject>/segments/<name>/segment_info.json
  7. Generates individual HTML reports under:
       reports/step01_segment_detection/<subject>_report.html
     and a global summary report:
       reports/step01_segment_detection/all_subjects_report.html

Usage:
  python step01_detect_mri.py               # Process all available subjects
  python step01_detect_mri.py --all         # Process all available subjects
  python step01_detect_mri.py --subject 1916 # Process single subject
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import mne

# ---------------------------------------------------------------------------
# Paths and Environment Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_ROOT    = PROJECT_ROOT / "data"
REPORT_DIR   = PROJECT_ROOT / "reports" / "step01_segment_detection"

EEG_ORIGINAL_ROOT = Path(os.getenv(
    "EEG_ORIGINAL_ROOT",
    "/media/vitaly/48DEA853CCEBFBF0/0DATDA_2026_eeg_fnri/fmri/eeg-fmri/original"
))
FMRI_ROOT = Path(os.getenv(
    "FMRI_ROOT",
    "/media/vitaly/48DEA853CCEBFBF0/0DATDA_2026_eeg_fnri/fmri/fmri"
))

# ---------------------------------------------------------------------------
# Detection parameters
# ---------------------------------------------------------------------------
DETECTION_CHANNELS = ["Fp1", "Fp2", "Fz", "Cz", "Pz"]
WINDOW_SEC         = 0.20     # RMS window width (seconds)
MIN_SEGMENT_SEC    = 25.0     # discard segments shorter than this (raw detection)
MIN_NAMED_SEC      = 250.0    # skip named-segment assignment for segments shorter than this
MERGE_GAP_SEC      = 1.0      # merge gaps smaller than this
AUTO_THRESHOLD     = False    # True → auto percentile, False → manual
MANUAL_THRESHOLD   = 300.0    # µV  (used when AUTO_THRESHOLD is False)

# ---------------------------------------------------------------------------
# Session / fMRI parameters
# ---------------------------------------------------------------------------
TR_SEC        = 2.5
DUMMY_VOLUMES = 12
DUMMY_DUR_SEC = DUMMY_VOLUMES * TR_SEC   # 30 s

# Session names assigned to segments 2..6 (segment 1 is skipped)
SESSION_ORDER  = ["ec", "eo", "drone", "lasertag", "video"]
SESSION_SUFFIX = {"ec": "301", "eo": "401", "drone": "501",
                  "lasertag": "601", "video": "701"}

# Colours used in the plot and report
SEG_COLORS: dict[str, str] = {
    "ec":       "#4e79a7",
    "eo":       "#f28e2b",
    "drone":    "#59a14f",
    "lasertag": "#e15759",
    "video":    "#b07aa1",
    "_skip":    "#555577",   # first (calibration) segment
}

_RP_SUFFIX_RE = re.compile(r"_(\d{3})_")
_SUBJ_RE      = re.compile(r"(\d{4})")


# ---------------------------------------------------------------------------
# Subject Discovery
# ---------------------------------------------------------------------------

def discover_all_eeg_vhdrs() -> dict[str, Path]:
    """
    Find all continuous raw .vhdr files.
    Priority 1: data/<subject>/raw/eeg96/<subject>.vhdr (if exists locally)
    Priority 2: EEG_ORIGINAL_ROOT/<subject>EEG/<subject>_inside.vhdr
    Returns dict mapping subject_id -> vhdr_path.
    """
    subjects: dict[str, Path] = {}

    # Check external drive
    if EEG_ORIGINAL_ROOT.exists():
        for d in sorted(EEG_ORIGINAL_ROOT.iterdir()):
            if not d.is_dir():
                continue
            m = _SUBJ_RE.search(d.name)
            if not m:
                continue
            sid = m.group(1)
            # Find inside vhdr or any vhdr
            vhdrs = list(d.glob("*_inside.vhdr")) or list(d.glob("*.vhdr"))
            if vhdrs:
                subjects[sid] = vhdrs[0]

    # Check local data folder (overrides if local raw exists)
    if DATA_ROOT.exists():
        for d in sorted(DATA_ROOT.iterdir()):
            if not d.is_dir():
                continue
            m = _SUBJ_RE.search(d.name)
            if not m:
                continue
            sid = m.group(1)
            local_vhdr = d / "raw" / "eeg96" / f"{sid}.vhdr"
            if local_vhdr.exists():
                subjects[sid] = local_vhdr

    return dict(sorted(subjects.items()))


# ---------------------------------------------------------------------------
# rp helpers
# ---------------------------------------------------------------------------

def find_all_rp_files(subject_id: str) -> list[Path]:
    """
    Return all rp_*.txt files for the subject, sorted by their numeric series
    suffix (the first 3-digit number found in the filename) in ascending order.
    Checks FMRI_ROOT/<subject_id>FMRI/ first, then local data/<subject_id>/raw/rp_spm/.
    """
    candidates: list[tuple[int, Path]] = []

    fmri_dir = FMRI_ROOT / f"{subject_id}FMRI"
    if fmri_dir.exists():
        for f in fmri_dir.glob("rp_*.txt"):
            m = _RP_SUFFIX_RE.search(f.name)
            if m:
                candidates.append((int(m.group(1)), f))

    # Fall back to local rp_spm directory
    if not candidates:
        local_rp = DATA_ROOT / subject_id / "raw" / "rp_spm"
        if local_rp.exists():
            for f in local_rp.glob("*.txt"):
                m = _RP_SUFFIX_RE.search(f.name)
                if m:
                    candidates.append((int(m.group(1)), f))

    # Sort by numeric suffix ascending → positional assignment
    candidates.sort(key=lambda x: x[0])
    return [p for _, p in candidates]


def get_rp_n_volumes(rp_path: Path) -> int:
    """Return number of rows (= volumes) in an SPM rp_*.txt file."""
    return int(np.loadtxt(rp_path).shape[0])


# ---------------------------------------------------------------------------
# Visualization & Report helpers
# ---------------------------------------------------------------------------

def _build_plot(times: np.ndarray,
                rms: np.ndarray,
                rms_smooth: np.ndarray,
                threshold: float,
                merged: list[list[float]],
                result: list[dict],
                subject_id: str) -> str:
    """
    Build an RMS-envelope figure and return it as a base-64 PNG string.
    """
    fig, ax = plt.subplots(figsize=(18, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    ax.plot(times, rms, color="#4a6fa5", lw=0.5, alpha=0.6, label="RMS raw")
    ax.plot(times, rms_smooth, color="#a8dadc", lw=1.0, alpha=0.9, label="RMS smooth")

    # Threshold line
    ax.axhline(threshold, color="#e63946", lw=1.2, ls="--", alpha=0.8,
               label=f"Threshold {threshold:.0f} µV")

    ylim_top = max(rms_smooth) * 1.15 if len(rms_smooth) else 600
    ax.set_ylim(0, ylim_top)

    legend_handles: list = []

    # Skipped segment (first calibration)
    if merged:
        t0, t1 = merged[0]
        ax.axvspan(t0, t1, alpha=0.18, color=SEG_COLORS["_skip"], linewidth=0)
        ax.axvline(t0, color=SEG_COLORS["_skip"], lw=1.0, alpha=0.6, ls="--")
        ax.axvline(t1, color=SEG_COLORS["_skip"], lw=1.0, alpha=0.6, ls="--")
        mid = (t0 + t1) / 2
        ax.text(mid, ylim_top * 0.95, "SKIP\n(calib)",
                ha="center", va="top", fontsize=7, color=SEG_COLORS["_skip"],
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a2e",
                          ec=SEG_COLORS["_skip"], lw=0.8, alpha=0.9))
        legend_handles.append(
            mpatches.Patch(color=SEG_COLORS["_skip"], alpha=0.5,
                           label="Skipped / short"))

    # Named and other segments
    seen: set[str] = set()
    for idx, (t0, t1) in enumerate(merged[1:], 2):
        matched_r = next((r for r in result if abs(r["t_start_sec"] - t0) < 1.0), None)
        if matched_r:
            name  = matched_r["name"]
            color = SEG_COLORS.get(name, "#888888")
            ax.axvspan(t0, t1, alpha=0.22, color=color, linewidth=0)
            ax.axvline(t0, color=color, lw=1.1, alpha=0.8, ls="--")
            ax.axvline(t1, color=color, lw=1.1, alpha=0.8, ls="--")
            mid = (t0 + t1) / 2
            ax.text(mid, ylim_top * 0.95, name,
                    ha="center", va="top", fontsize=8.5, color=color,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#1a1a2e",
                              ec=color, lw=0.9, alpha=0.9))
            if name not in seen:
                legend_handles.append(
                    mpatches.Patch(color=color, alpha=0.6, label=name))
                seen.add(name)
        else:
            # Segment was skipped (e.g. dur < MIN_NAMED_SEC)
            ax.axvspan(t0, t1, alpha=0.15, color=SEG_COLORS["_skip"], linewidth=0)
            ax.axvline(t0, color=SEG_COLORS["_skip"], lw=0.8, alpha=0.5, ls=":")
            ax.axvline(t1, color=SEG_COLORS["_skip"], lw=0.8, alpha=0.5, ls=":")
            mid = (t0 + t1) / 2
            ax.text(mid, ylim_top * 0.95, "SKIP\n(short)",
                    ha="center", va="top", fontsize=6.5, color=SEG_COLORS["_skip"],
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a2e",
                              ec=SEG_COLORS["_skip"], lw=0.8, alpha=0.9))

    ax.set_xlabel("Time (s)", color="#cccccc", fontsize=10)
    ax.set_ylabel("RMS  [µV]", color="#cccccc", fontsize=10)
    ax.set_title(
        f"Subject {subject_id}  —  MRI segment detection  "
        f"(total {times[-1]:.0f}s / {times[-1]/60:.1f} min)  "
        f"| channels: {', '.join(DETECTION_CHANNELS)}",
        color="#ffffff", fontsize=11, pad=8)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")

    legend_handles += [
        plt.Line2D([0], [0], color="#a8dadc", lw=1.5, label="RMS smooth"),
        plt.Line2D([0], [0], color="#e63946", lw=1.2, ls="--",
                   label=f"Threshold {threshold:.0f} µV"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=7.5, ncol=4, framealpha=0.45,
              facecolor="#1a1a2e", edgecolor="#444466", labelcolor="#cccccc")

    fig.tight_layout(pad=0.8)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#0d0d1a;
     color:#d0d0e0;padding:28px 40px;line-height:1.6}
h1{color:#a8dadc;font-size:1.85rem;margin-bottom:4px}
.sub{color:#777799;font-size:.85rem;margin-bottom:28px}
h2{color:#a8dadc;font-size:1.15rem;margin:26px 0 10px;
   border-bottom:1px solid #1e1e3a;padding-bottom:5px}
table{border-collapse:collapse;width:100%;font-size:.82rem;margin-bottom:20px}
th{background:#131328;color:#a8dadc;padding:7px 12px;text-align:left;
   border-bottom:2px solid #222244;white-space:nowrap}
td{padding:6px 12px;border-bottom:1px solid #181830;vertical-align:middle}
tr:hover td{background:#141428}
.ctr{text-align:center}
.ok{color:#59a14f;font-weight:700}
.warn{color:#e15759;font-weight:700}
.skip{color:#888;font-style:italic}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;
       border:1px solid;font-size:.77rem;font-weight:700;letter-spacing:.03em}
.rms-img{width:100%;display:block;border-radius:6px;margin-top:16px}
.settings td:first-child{color:#9090cc;font-weight:600;white-space:nowrap;
                          padding-right:20px;width:200px}
a{color:#a8dadc;text-decoration:none}
a:hover{text-decoration:underline}
footer{margin-top:34px;font-size:.72rem;color:#444466}
"""


def _build_html(subject_id: str,
                vhdr_path: Path,
                total_dur: float,
                sfreq: float,
                avail_chs: list[str],
                threshold: float,
                merged: list[list[float]],
                result: list[dict],
                plot_b64: str,
                generated_at: str) -> str:

    # Settings table
    settings = [
        ("Subject ID",          subject_id),
        ("Recording file",      str(vhdr_path)),
        ("Sampling frequency",  f"{sfreq:.0f} Hz"),
        ("Total duration",      f"{total_dur:.2f} s  ({total_dur/60:.1f} min)"),
        ("Detection channels",  ", ".join(avail_chs)),
        ("RMS window",          f"{WINDOW_SEC} s"),
        ("Detection threshold", f"{threshold:.1f} µV"),
        ("Min segment length",  f"{MIN_SEGMENT_SEC} s"),
        ("Merge gap",           f"{MERGE_GAP_SEC} s"),
        ("TR",                  f"{TR_SEC} s"),
        ("Dummy volumes",       f"{DUMMY_VOLUMES}  ({DUMMY_DUR_SEC:.0f} s)"),
        ("fMRI root",           str(FMRI_ROOT)),
    ]
    set_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in settings)

    # All RMS segments (including skipped)
    all_rows = []
    for idx, (t0, t1) in enumerate(merged, 1):
        dur = t1 - t0
        if idx == 1:
            all_rows.append(
                f"<tr><td>{idx}</td><td class='skip'>SKIP (calibration)</td>"
                f"<td>{t0:.2f}</td><td>{t1:.2f}</td><td>{dur:.1f}</td>"
                f"<td class='skip'>—</td><td class='skip'>—</td>"
                f"<td class='skip'>—</td></tr>")
        else:
            matched_r = next((r for r in result if abs(r["t_start_sec"] - t0) < 1.0), None)
            if matched_r:
                name   = matched_r["name"]
                color  = SEG_COLORS.get(name, "#888")
                badge  = (f"<span class='badge' "
                          f"style='background:{color}25;border-color:{color}'>"
                          f"{name}</span>")
                rp_v   = matched_r["rp_n_volumes"] or "—"
                rp_d   = f"{matched_r['rp_duration_sec']:.0f}" if matched_r["rp_duration_sec"] else "—"
                diff   = (abs(dur - (matched_r["rp_duration_sec"] + DUMMY_DUR_SEC))
                          if matched_r["rp_duration_sec"] else None)
                match_cls = "ok" if (diff is not None and diff < 5) else "warn"
                match_str = (f"Δ{diff:.0f}s" if diff is not None else "n/a")
                all_rows.append(
                    f"<tr><td>{idx}</td><td>{badge}</td>"
                    f"<td>{t0:.2f}</td><td>{t1:.2f}</td><td>{dur:.1f}</td>"
                    f"<td class='ctr'>{rp_v}</td><td class='ctr'>{rp_d}</td>"
                    f"<td class='ctr {match_cls}'>{match_str}</td></tr>")
            else:
                skip_reason = f"SKIP (dur {dur:.1f}s < {MIN_NAMED_SEC:.0f}s)" if dur < MIN_NAMED_SEC else "SKIP (extra)"
                all_rows.append(
                    f"<tr><td>{idx}</td><td class='skip'>{skip_reason}</td>"
                    f"<td>{t0:.2f}</td><td>{t1:.2f}</td><td>{dur:.1f}</td>"
                    f"<td class='skip'>—</td><td class='skip'>—</td>"
                    f"<td class='skip'>—</td></tr>")

    seg_table = "\n".join(all_rows)

    img_tag = (f"<img src='data:image/png;base64,{plot_b64}' "
               f"class='rms-img' alt='RMS envelope'>") if plot_b64 else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>Step 01 — Segment Detection — {subject_id}</title>
<style>{_CSS}</style>
</head>
<body>
<div style="display:flex; justify-content:space-between; align-items:baseline;">
  <h1>Step 01 · MRI Segment Detection</h1>
  <a href="all_subjects_report.html">← All Subjects Summary</a>
</div>
<p class="sub">Subject: <b>{subject_id}</b> &nbsp;|&nbsp;
Generated: {generated_at} &nbsp;|&nbsp;
{len(merged)} raw segments detected &nbsp;|&nbsp;
{len(result)} named sessions saved</p>

<h2>⚙ Settings</h2>
<table class="settings"><tbody>{set_rows}</tbody></table>

<h2>📋 Segment Table</h2>
<table>
  <thead><tr>
    <th>#</th><th>Session</th><th>Start (s)</th><th>Stop (s)</th>
    <th>RMS dur (s)</th><th>rp vols</th><th>rp dur (s)</th><th>Match</th>
  </tr></thead>
  <tbody>{seg_table}</tbody>
</table>

<h2>📈 RMS Envelope</h2>
{img_tag}

<footer>step01_detect_mri.py — EEG-fMRI Clean Pipeline</footer>
</body>
</html>"""


def build_summary_html(all_results: dict[str, list[dict]], generated_at: str) -> str:
    """Build multi-subject overview HTML."""
    rows = []
    total_sessions = 0
    for sid, segs in sorted(all_results.items()):
        total_sessions += len(segs)
        badges = []
        for s in segs:
            c = SEG_COLORS.get(s["name"], "#888")
            badges.append(
                f"<span class='badge' style='background:{c}25;border-color:{c}'>"
                f"{s['name']} ({s['duration_sec']:.0f}s)</span>"
            )
        badges_html = " ".join(badges)
        all_ok = all(s.get("rp_path") is not None for s in segs)
        status_html = "<span class='ok'>✓ Matched rp</span>" if all_ok else "<span class='warn'>Partial rp</span>"
        report_link = f"<a href='{sid}_report.html'><b>{sid}</b> Report ↗</a>"

        rows.append(
            f"<tr>"
            f"<td><b>{sid}</b></td>"
            f"<td class='ctr'>{len(segs)}</td>"
            f"<td>{badges_html}</td>"
            f"<td class='ctr'>{status_html}</td>"
            f"<td class='ctr'>{report_link}</td>"
            f"</tr>"
        )

    table_body = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>Step 01 — Segment Detection — All Subjects</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Step 01 · Multi-Subject Segment Detection Overview</h1>
<p class="sub">Generated: {generated_at} &nbsp;|&nbsp;
Total Subjects: <b>{len(all_results)}</b> &nbsp;|&nbsp;
Total Named Sessions: <b>{total_sessions}</b></p>

<h2>📋 Subjects Summary</h2>
<table>
  <thead><tr>
    <th>Subject</th><th>Sessions</th><th>Detected Segments (ec → eo → drone → lasertag → video)</th>
    <th>rp Status</th><th>Report</th>
  </tr></thead>
  <tbody>{table_body}</tbody>
</table>

<footer>step01_detect_mri.py — EEG-fMRI Clean Pipeline</footer>
</body>
</html>"""


def update_all_subjects_summary_report() -> Path:
    """Scan data/<sid>/segments/ for all subjects and rebuild all_subjects_report.html."""
    all_results: dict[str, list[dict]] = {}
    if DATA_ROOT.exists():
        for sdir in sorted(DATA_ROOT.iterdir()):
            if not sdir.is_dir():
                continue
            sid = sdir.name
            seg_dir = sdir / "segments"
            if not seg_dir.exists():
                continue
            subj_segs = []
            for name in SESSION_ORDER:
                s_info = seg_dir / name / "segment_info.json"
                if s_info.exists():
                    try:
                        with open(s_info, "r", encoding="utf-8") as f:
                            subj_segs.append(json.load(f))
                    except Exception:
                        pass
            if subj_segs:
                all_results[sid] = subj_segs

    if all_results:
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        summary_html = build_summary_html(all_results, generated_at)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = REPORT_DIR / "all_subjects_report.html"
        summary_path.write_text(summary_html, encoding="utf-8")
        print(f"[STEP 01] Multi-Subject Summary Report updated: file://{summary_path.resolve()}")
        return summary_path
    return None



# ---------------------------------------------------------------------------
# Main detection function (single subject)
# ---------------------------------------------------------------------------

def detect_mri_sessions(vhdr_path: Path = None,
                        output_dir: Path = None,
                        subject_id: str = None,
                        write_report: bool = True) -> list[dict]:
    """
    Detect MRI session segments in a continuous BrainVision recording.
    """
    if vhdr_path is None:
        # Default discovery
        all_vhdrs = discover_all_eeg_vhdrs()
        if "1916" in all_vhdrs:
            vhdr_path = all_vhdrs["1916"]
        elif all_vhdrs:
            vhdr_path = next(iter(all_vhdrs.values()))
        else:
            raise FileNotFoundError("No raw EEG .vhdr files found.")

    vhdr_path = Path(vhdr_path).resolve()

    if subject_id is None:
        m = _SUBJ_RE.search(vhdr_path.name) or _SUBJ_RE.search(vhdr_path.parent.name)
        subject_id = m.group(1) if m else vhdr_path.stem

    segments_dir = Path(output_dir) if output_dir is not None else (DATA_ROOT / subject_id / "segments")
    segments_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"[STEP 01] Reading continuous EEG: {vhdr_path}")
    print(f"          Subject: {subject_id} | Output dir: {segments_dir}")
    print("=" * 70)

    raw       = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose=False)
    sfreq     = float(raw.info["sfreq"])
    total_dur = float(raw.times[-1])
    print(f"Sampling frequency : {sfreq} Hz | Channels : {len(raw.ch_names)} "
          f"| Total Duration : {total_dur:.2f} s ({total_dur/60:.1f} min)")

    # ------------------------------------------------------------------ RMS
    avail_chs = [ch for ch in DETECTION_CHANNELS if ch in raw.ch_names]
    if not avail_chs:
        avail_chs = raw.ch_names[:5]
    print(f"Detection channels : {avail_chs}")

    det_raw = raw.copy().pick(avail_chs)
    data    = det_raw.get_data(units="uV")

    window_samples = int(WINDOW_SEC * sfreq)
    n_windows      = data.shape[1] // window_samples
    times_win      = np.zeros(n_windows)
    rms_values     = np.zeros(n_windows)
    for i in range(n_windows):
        s = i * window_samples
        e = s + window_samples
        rms_values[i] = np.median(np.sqrt(np.mean(data[:, s:e] ** 2, axis=1)))
        times_win[i]  = (s + e) / 2 / sfreq

    # Smooth
    smooth_w   = 5
    rms_smooth = np.copy(rms_values)
    for i in range(n_windows):
        lo = max(0, i - smooth_w)
        hi = min(n_windows, i + smooth_w + 1)
        rms_smooth[i] = np.median(rms_values[lo:hi])

    threshold = (MANUAL_THRESHOLD if not AUTO_THRESHOLD
                 else np.percentile(rms_smooth, 20) * 2.0)
    print(f"Detection threshold: {threshold:.1f} µV")

    # ---------------------------------------------------------------- Detect
    is_mri    = rms_smooth > threshold
    segments  = []
    in_seg    = False
    seg_start = 0.0

    for i, active in enumerate(is_mri):
        if active and not in_seg:
            in_seg    = True
            seg_start = i * window_samples / sfreq
        elif not active and in_seg:
            in_seg   = False
            seg_stop = i * window_samples / sfreq
            if (seg_stop - seg_start) >= MIN_SEGMENT_SEC:
                segments.append((seg_start, seg_stop))

    if in_seg:
        seg_stop = n_windows * window_samples / sfreq
        if (seg_stop - seg_start) >= MIN_SEGMENT_SEC:
            segments.append((seg_start, seg_stop))

    # Merge nearby gaps
    merged: list[list[float]] = []
    for t0, t1 in segments:
        if not merged:
            merged.append([t0, t1])
        elif t0 - merged[-1][1] <= MERGE_GAP_SEC:
            merged[-1][1] = t1
        else:
            merged.append([t0, t1])

    print(f"\nFound {len(merged)} RMS segments total (segment 1 will be skipped):")
    for idx, (t0, t1) in enumerate(merged, 1):
        mark = " ← SKIP (calibration)" if idx == 1 else ""
        print(f"  Segment {idx}: [{t0:.2f}s – {t1:.2f}s]  dur={t1-t0:.1f}s{mark}")

    # --------------------------------------------------------- Assign names
    # Drop first calibration segment, then filter out any segment shorter
    # than MIN_NAMED_SEC (e.g. aborted runs, artefact bursts).
    candidates = merged[1:]   # drop first calibration segment
    named_segments = []
    named_indices = []
    for idx, (t0, t1) in enumerate(candidates, 2):
        dur = t1 - t0
        if dur < MIN_NAMED_SEC:
            print(f"  [INFO] Segment [{t0:.2f}s – {t1:.2f}s] dur={dur:.1f}s < "
                  f"{MIN_NAMED_SEC:.0f}s → skipped (too short for a named session)")
        else:
            named_segments.append([t0, t1])
            named_indices.append(idx)

    if len(named_segments) > len(SESSION_ORDER):
        print(f"\n[WARN] {len(named_segments)} segments after skip but only "
              f"{len(SESSION_ORDER)} session names defined — extras ignored.")
        named_segments = named_segments[:len(SESSION_ORDER)]
        named_indices = named_indices[:len(SESSION_ORDER)]
    elif len(named_segments) < len(SESSION_ORDER):
        print(f"\n[WARN] Only {len(named_segments)} segments after skip "
              f"(expected {len(SESSION_ORDER)}).")  

    print(f"\n{'Segment':<10} {'Name':<12} {'Start(s)':>9} {'Stop(s)':>9} "
          f"{'RMS dur':>8} {'rp vols':>8} {'rp dur':>8} {'Match':>6}")
    print("-" * 80)

    result: list[dict] = []
    dur_lines: list[str] = []

    # Collect all rp files sorted by numeric suffix (ascending) and assign positionally
    all_rp_files = find_all_rp_files(subject_id)
    print(f"  [INFO] Found {len(all_rp_files)} rp file(s) for subject {subject_id} "
          f"(sorted by series number):")
    for rp in all_rp_files:
        m = _RP_SUFFIX_RE.search(rp.name)
        print(f"    {m.group(1) if m else '???'}  →  {rp.name}")

    if len(all_rp_files) < len(named_segments):
        print(f"  [WARN] Only {len(all_rp_files)} rp file(s) found, "
              f"but {len(named_segments)} named segment(s) — some sessions will have no rp.")

    for rank, (t0, t1) in enumerate(named_segments):
        name    = SESSION_ORDER[rank]
        dur_rms = t1 - t0
        actual_seg_idx = named_indices[rank]

        rp_path = all_rp_files[rank] if rank < len(all_rp_files) else None
        # Extract actual suffix from matched file (for segment_info.json)
        suffix = SESSION_SUFFIX[name]   # default
        if rp_path:
            m = _RP_SUFFIX_RE.search(rp_path.name)
            if m:
                suffix = m.group(1)

        rp_vols   = None
        rp_dur    = None
        match_str = "n/a"
        if rp_path:
            rp_vols   = get_rp_n_volumes(rp_path)
            rp_dur    = rp_vols * TR_SEC
            expected  = rp_dur + DUMMY_DUR_SEC
            diff      = abs(dur_rms - expected)
            match_str = f"Δ{diff:.0f}s"
        else:
            print(f"  [WARNING] No rp file for session '{name}' (position {rank+1}).")

        seg_folder = segments_dir / name
        seg_folder.mkdir(exist_ok=True)

        print(f"  {actual_seg_idx:<8} {name:<12} {t0:>9.2f} {t1:>9.2f} "
              f"{dur_rms:>8.1f} "
              f"{rp_vols if rp_vols else '—':>8} "
              f"{rp_dur if rp_dur else '—':>8} "
              f"{match_str:>6}")

        seg_info = {
            "segment_idx":     actual_seg_idx,
            "name":            name,
            "session_suffix":  suffix,
            "t_start_sec":     float(t0),
            "t_stop_sec":      float(t1),
            "duration_sec":    float(dur_rms),
            "raw_vhdr":        str(vhdr_path),
            "sfreq":           sfreq,
            "rp_path":         str(rp_path) if rp_path else None,
            "rp_file":         str(rp_path) if rp_path else None,
            "rp_n_volumes":    rp_vols,
            "rp_duration_sec": rp_dur,
            "dummy_volumes":   DUMMY_VOLUMES,
            "tr_sec":          TR_SEC,
        }
        with open(seg_folder / "segment_info.json", "w", encoding="utf-8") as f:
            json.dump(seg_info, f, indent=2)

        dur_lines.append(f"{name} (segment {actual_seg_idx}): "
                         f"[{t0:.2f}s – {t1:.2f}s]  dur={dur_rms:.1f}s")
        result.append(seg_info)

    with open(segments_dir / "segments_duration.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(dur_lines) + "\n")

    print(f"\n[STEP 01] Done for subject {subject_id}. {len(result)} named session(s) saved.")

    # ---------------------------------------------------------------- Report
    if write_report:
        print(f"[STEP 01] Generating HTML report for {subject_id} …", flush=True)
        plot_b64 = _build_plot(
            times_win, rms_values, rms_smooth,
            threshold, merged, result,
            subject_id=subject_id)

        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        html = _build_html(
            subject_id, vhdr_path, total_dur, sfreq, avail_chs,
            threshold, merged, result, plot_b64, generated_at)

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / f"{subject_id}_report.html"
        report_path.write_text(html, encoding="utf-8")
        print(f"[STEP 01] Report saved → {report_path}")

    return result


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Detect MRI session segments in continuous EEG")
    parser.add_argument("--subject", default=None, help="Process single subject ID (e.g. 1916)")
    parser.add_argument("--all", action="store_true", help="Process all available EEG subjects")
    parser.add_argument("--vhdr", default=None, help="Explicit path to raw .vhdr file")
    args = parser.parse_args()

    all_vhdrs = discover_all_eeg_vhdrs()
    if not all_vhdrs and not args.vhdr:
        print("[ERROR] No EEG recordings discovered. Check EEG_ORIGINAL_ROOT.")
        sys.exit(1)

    print(f"[INFO] Discovered {len(all_vhdrs)} EEG recording(s): {', '.join(all_vhdrs.keys())}")

    # Determine subjects to process
    targets: dict[str, Path] = {}
    if args.vhdr:
        vpath = Path(args.vhdr).resolve()
        m = _SUBJ_RE.search(vpath.name) or _SUBJ_RE.search(vpath.parent.name)
        sid = args.subject or (m.group(1) if m else "unknown")
        targets[sid] = vpath
    elif args.subject:
        if args.subject not in all_vhdrs:
            print(f"[ERROR] Subject {args.subject} not found in discovered list: {list(all_vhdrs.keys())}")
            sys.exit(1)
        targets[args.subject] = all_vhdrs[args.subject]
    else:
        # Process all discovered by default
        targets = all_vhdrs

    all_results: dict[str, list[dict]] = {}
    for sid, vhdr in targets.items():
        print("\n" + "=" * 80)
        print(f"PROCESSING SUBJECT {sid}")
        print("=" * 80)
        res = detect_mri_sessions(vhdr_path=vhdr, subject_id=sid, write_report=True)
        all_results[sid] = res

    # Always update the all-subjects summary report
    update_all_subjects_summary_report()


if __name__ == "__main__":
    main()
