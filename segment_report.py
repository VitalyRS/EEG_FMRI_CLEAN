"""
segment_report.py — EEG-fMRI Segment Mapping Visualization Report
==================================================================
For each subject:
  1. Reads inside.eeg → computes RMS envelope (median over detection channels,
     windowed at PLOT_WINDOW_SEC for speed — no full signal in RAM)
  2. Draws per-subject figure:
       • RMS envelope of the continuous recording
       • Vertical dashed lines + labels at every vmrk marker (ec / eo)
       • Coloured shaded spans for each detected segment
  3. Saves PNG to  reports/segment_maps/<SUBJ>_inside.png
  4. Writes HTML  reports/segment_report.html with:
       • Script settings table
       • Full segment table (all subjects)
       • Per-subject panels with embedded PNG

Usage:
  python segment_report.py                    # all subjects
  python segment_report.py --subject 1916     # single subject
  python segment_report.py --no-plots         # table-only (fast)

Requirements:  mne, numpy, matplotlib  (already present in the project env)
"""
from __future__ import annotations

import argparse
import base64
import sys
from datetime import datetime
from pathlib import Path


import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import mne

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
try:
    from segment_mapper import (
        discover_subjects, SubjectInfo, SegmentInfo,
        parse_vmrk, get_sfreq_from_vhdr,
        EEG_ORIGINAL_ROOT, FMRI_ROOT, PROCESSED_ROOT,
        TR_SEC, SLICES_PER_VOLUME, DUMMY_VOLUMES,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from segment_mapper import (
        discover_subjects, SubjectInfo, SegmentInfo,
        parse_vmrk, get_sfreq_from_vhdr,
        EEG_ORIGINAL_ROOT, FMRI_ROOT, PROCESSED_ROOT,
        TR_SEC, SLICES_PER_VOLUME, DUMMY_VOLUMES,
    )

# ---------------------------------------------------------------------------
# Report configuration
# ---------------------------------------------------------------------------
REPORT_DIR  = Path(__file__).parent / "reports"
MAPS_DIR    = REPORT_DIR / "segment_maps"
REPORT_HTML = REPORT_DIR / "segment_report.html"

DETECTION_CHANNELS = ["Fp1", "Fp2", "Fz", "Cz", "Pz"]
PLOT_WINDOW_SEC    = 1.0   # RMS window (s) — keeps memory low

SEG_COLORS: dict[str, str] = {
    "ec":       "#4e79a7",
    "eo":       "#f28e2b",
    "drone":    "#59a14f",
    "lasertag": "#e15759",
    "video":    "#b07aa1",
    "ec_out":   "#76b7b2",
    "eo_out":   "#edc948",
}
SEG_ALPHA = 0.25
MARKER_COLORS = {"ec": "#7eb8d8", "eo": "#ffc87a"}


# ---------------------------------------------------------------------------
# RMS envelope (low-RAM streaming)
# ---------------------------------------------------------------------------

def compute_rms_envelope(vhdr_path: Path,
                          window_sec: float = PLOT_WINDOW_SEC
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Windowed RMS (median over DETECTION_CHANNELS). Returns (times_s, rms_uV)."""
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=False, verbose=False)
    sfreq = float(raw.info["sfreq"])
    avail = [ch for ch in DETECTION_CHANNELS if ch in raw.ch_names]
    if not avail:
        avail = raw.ch_names[:5]

    det      = raw.copy().pick(avail)
    n_samp   = det.n_times
    win_samp = max(1, int(window_sec * sfreq))
    n_wins   = n_samp // win_samp

    times = np.zeros(n_wins)
    rms   = np.zeros(n_wins)

    CHUNK = 500
    for chunk_start in range(0, n_wins, CHUNK):
        chunk_end = min(chunk_start + CHUNK, n_wins)
        s0 = chunk_start * win_samp
        s1 = chunk_end   * win_samp
        data_uv, _ = det[:, s0:s1]
        data_uv = data_uv * 1e6          # V → µV
        for i, wi in enumerate(range(chunk_start, chunk_end)):
            ws = i * win_samp
            we = ws + win_samp
            ch_rms = np.sqrt(np.mean(data_uv[:, ws:we] ** 2, axis=1))
            rms[wi]   = float(np.median(ch_rms))
            times[wi] = (wi * win_samp + win_samp / 2) / sfreq

    return times, rms


# ---------------------------------------------------------------------------
# Per-subject figure
# ---------------------------------------------------------------------------

def plot_subject(subj: SubjectInfo, save_path: Path) -> None:
    inside_vhdr = subj.eeg_dir / f"{subj.subject_id}_inside.vhdr"
    inside_vmrk = subj.eeg_dir / f"{subj.subject_id}_inside.vmrk"

    if not inside_vhdr.exists():
        print(f"  [REPORT] {subj.subject_id}: inside.vhdr not found, skipping plot")
        return

    print(f"  [REPORT] {subj.subject_id}: computing RMS envelope …", flush=True)
    times, rms = compute_rms_envelope(inside_vhdr)
    sfreq = get_sfreq_from_vhdr(inside_vhdr)

    markers = parse_vmrk(inside_vmrk) if inside_vmrk.exists() else []
    ec_marks = [mk for mk in markers if mk["description"].lower() == "ec"]
    eo_marks = [mk for mk in markers if mk["description"].lower() == "eo"]

    fig, ax = plt.subplots(figsize=(16, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    ax.plot(times, rms, color="#a8dadc", lw=0.7, alpha=0.85, label="RMS")

    # Smoothed overlay
    smth = min(30, len(rms))
    if smth > 1:
        kernel = np.ones(smth) / smth
        rms_smooth = np.convolve(
            np.pad(rms, smth // 2, mode="edge"), kernel, mode="valid"
        )[:len(rms)]
        ax.plot(times, rms_smooth, color="#e63946", lw=1.6, alpha=0.9, label="RMS smooth")

    # Segment spans
    inside_segs = [s for s in subj.segments
                   if s.vhdr_path.name == inside_vhdr.name]
    legend_handles = []
    seen: set[str] = set()
    ylim_top = max(rms) * 1.05 if len(rms) else 500

    for seg in inside_segs:
        color = SEG_COLORS.get(seg.name, "#888")
        ax.axvspan(seg.t_start_sec, seg.t_stop_sec,
                   alpha=SEG_ALPHA, color=color, linewidth=0)
        ax.axvline(seg.t_start_sec, color=color, lw=1.1, alpha=0.7, ls="--")
        ax.axvline(seg.t_stop_sec,  color=color, lw=1.1, alpha=0.7, ls="--")
        mid = (seg.t_start_sec + seg.t_stop_sec) / 2
        ax.text(mid, ylim_top * 0.93, seg.name,
                ha="center", va="top", fontsize=7.5, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a2e", ec=color, lw=0.8, alpha=0.9))
        if seg.name not in seen:
            legend_handles.append(mpatches.Patch(color=color, alpha=0.6, label=seg.name))
            seen.add(seg.name)

    # vmrk marker lines
    for i, mk in enumerate(ec_marks):
        t = (mk["position"] - 1) / sfreq
        ax.axvline(t, color=MARKER_COLORS["ec"], lw=1.3, alpha=0.75, ls=":")
        ax.text(t, ylim_top * 0.02, f"ec{i+1}",
                color=MARKER_COLORS["ec"], fontsize=6.5, ha="center", va="bottom", clip_on=True)
    for i, mk in enumerate(eo_marks):
        t = (mk["position"] - 1) / sfreq
        ax.axvline(t, color=MARKER_COLORS["eo"], lw=1.3, alpha=0.75, ls=":")
        ax.text(t, ylim_top * 0.08, f"eo{i+1}",
                color=MARKER_COLORS["eo"], fontsize=6.5, ha="center", va="bottom", clip_on=True)

    # Axes
    ax.set_ylim(0, ylim_top)
    ax.set_xlabel("Time (s)", color="#cccccc", fontsize=10)
    ax.set_ylabel("RMS  [µV]", color="#cccccc", fontsize=10)
    ax.set_title(
        f"Subject {subj.subject_id}  —  inside recording  "
        f"({times[-1]:.0f}s / {times[-1]/60:.1f} min)  "
        f"| channels: {', '.join(DETECTION_CHANNELS)}",
        color="#ffffff", fontsize=11, pad=8)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")

    legend_handles += [
        mpatches.Patch(color=MARKER_COLORS["ec"], alpha=0.8, label="vmrk ec"),
        mpatches.Patch(color=MARKER_COLORS["eo"], alpha=0.8, label="vmrk eo"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=7, ncol=4, framealpha=0.45,
              facecolor="#1a1a2e", edgecolor="#444466", labelcolor="#cccccc")

    fig.tight_layout(pad=0.8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [REPORT] {subj.subject_id}: saved → {save_path.name}")


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _png_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _seg_row(seg: SegmentInfo) -> str:
    color  = SEG_COLORS.get(seg.name, "#888")
    rp     = seg.rp_path.name if seg.rp_path else "<em>n/a</em>"
    bergen = "✓" if seg.needs_bergen else "—"
    return (
        f"<tr>"
        f"<td>{seg.subject_id}</td>"
        f"<td><span class='badge' style='background:{color}25;border-color:{color}'>"
        f"{seg.name}</span></td>"
        f"<td>{seg.t_start_sec:.1f}</td>"
        f"<td>{seg.t_stop_sec:.1f}</td>"
        f"<td>{seg.duration_sec:.1f}</td>"
        f"<td class='ctr'>{bergen}</td>"
        f"<td class='mono'>{rp}</td>"
        f"</tr>"
    )


def _settings_rows() -> str:
    rows = [
        ("EEG original root",   str(EEG_ORIGINAL_ROOT)),
        ("fMRI root",           str(FMRI_ROOT)),
        ("Processed root",      str(PROCESSED_ROOT)),
        ("TR (s)",              str(TR_SEC)),
        ("Slices per volume",   str(SLICES_PER_VOLUME)),
        ("Dummy volumes",       str(DUMMY_VOLUMES)),
        ("RMS window (s)",      str(PLOT_WINDOW_SEC)),
        ("Detection channels",  ", ".join(DETECTION_CHANNELS)),
        ("Min usable duration", "10 s"),
    ]
    return "".join(f"<tr><td class='k'>{k}</td><td class='v mono'>{v}</td></tr>" for k, v in rows)


def _subject_panel(subj: SubjectInfo, img_path: "Path | None") -> str:
    badge_parts = []
    for s in subj.segments:
        c = SEG_COLORS.get(s.name, "#888")
        badge_parts.append(
            f"<span class='badge' style='background:{c}25;border-color:{c}'>{s.name}</span>"
        )
    badges = " ".join(badge_parts)
    img_tag = ""
    if img_path and img_path.exists():
        b64 = _png_b64(img_path)
        img_tag = (
            f"<img src='data:image/png;base64,{b64}' "
            f"class='rms-img' alt='RMS {subj.subject_id}'>"
        )
    n = len(subj.segments)
    sid = subj.subject_id
    return (
        f"<div class='panel'>"
        f"<div class='panel-hdr'>"
        f"<span class='sid'>Subject {sid}</span>"
        f"<span class='scnt'>{n} segment(s)</span>"
        f"<span>{badges}</span>"
        f"</div>"
        f"{img_tag}"
        f"</div>\n"
    )



CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#0d0d1a;color:#d0d0e0;padding:24px 36px;line-height:1.6}
h1{color:#a8dadc;font-size:1.9rem;margin-bottom:4px}
.sub{color:#555;font-size:.82rem;margin-bottom:28px}
h2{color:#a8dadc;font-size:1.1rem;margin:26px 0 10px;border-bottom:1px solid #1e1e3a;padding-bottom:5px}
table.st{border-collapse:collapse;width:auto;margin-bottom:6px}
table.st td{padding:3px 14px;font-size:.82rem}
td.k{color:#9090cc;font-weight:600;white-space:nowrap}
td.v{color:#d0d0e0}
table.seg{border-collapse:collapse;width:100%;font-size:.81rem;margin-bottom:20px}
table.seg th{background:#131328;color:#a8dadc;padding:6px 10px;text-align:left;border-bottom:2px solid #222244;white-space:nowrap}
table.seg td{padding:5px 10px;border-bottom:1px solid #181830;vertical-align:middle}
table.seg tr:hover td{background:#141428}
.ctr{text-align:center}
.mono{font-family:'Courier New',monospace;font-size:.77rem;color:#9090bb}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;border:1px solid;font-size:.77rem;font-weight:700;letter-spacing:.03em}
.panel{background:#10101f;border:1px solid #1e1e3a;border-radius:8px;margin-bottom:18px;overflow:hidden}
.panel-hdr{display:flex;align-items:center;gap:12px;padding:10px 16px;background:#16162e;flex-wrap:wrap}
.sid{font-size:1rem;font-weight:700;color:#e2e2ff}
.scnt{font-size:.78rem;color:#666}
.rms-img{width:100%;display:block}
footer{margin-top:30px;font-size:.72rem;color:#333}
"""

def build_html(subjects: list[SubjectInfo],
               img_paths: dict[str, Path | None],
               generated_at: str) -> str:
    all_rows  = "".join(_seg_row(s) for subj in subjects for s in subj.segments)
    panels    = "".join(_subject_panel(subj, img_paths.get(subj.subject_id)) for subj in subjects)
    total_seg = sum(len(s.segments) for s in subjects)
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>EEG-fMRI Segment Report</title>
<style>{CSS}</style>
</head>
<body>
<h1>EEG-fMRI Segment Mapping Report</h1>
<p class="sub">Generated: {generated_at} &nbsp;|&nbsp; {len(subjects)} subject(s) &nbsp;|&nbsp; {total_seg} segment(s)</p>

<h2>⚙ Settings</h2>
<table class="st"><tbody>{_settings_rows()}</tbody></table>

<h2>📋 Segment Table</h2>
<table class="seg">
  <thead><tr>
    <th>Subject</th><th>Segment</th><th>Start (s)</th><th>Stop (s)</th>
    <th>Dur (s)</th><th>Bergen</th><th>rp file</th>
  </tr></thead>
  <tbody>{all_rows}</tbody>
</table>

<h2>🧠 RMS Envelope + Segments</h2>
{panels}

<footer>segment_report.py — EEG-fMRI Clean Pipeline</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EEG-fMRI segment report")
    parser.add_argument("--subject", default=None, help="Single subject ID (e.g. 1916)")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG generation")
    parser.add_argument("--out", default=str(REPORT_HTML), help="Output HTML path")
    args = parser.parse_args()

    out_html = Path(args.out)

    print("=" * 70)
    print("[REPORT] Discovering subjects …")
    print("=" * 70)
    subjects = discover_subjects()
    if args.subject:
        subjects = [s for s in subjects if s.subject_id == args.subject]
    if not subjects:
        print("No subjects found.  Check EEG_ORIGINAL_ROOT / FMRI_ROOT paths.")
        sys.exit(1)

    total = sum(len(s.segments) for s in subjects)
    print(f"\n[REPORT] {len(subjects)} subject(s), {total} segment(s) total\n")

    img_paths: dict[str, Path | None] = {}

    if not args.no_plots:
        print("[REPORT] Generating RMS plots …")
        MAPS_DIR.mkdir(parents=True, exist_ok=True)
        for subj in subjects:
            img = MAPS_DIR / f"{subj.subject_id}_inside.png"
            try:
                plot_subject(subj, img)
                img_paths[subj.subject_id] = img
            except Exception as exc:
                print(f"  [REPORT] {subj.subject_id}: plot failed — {exc}")
                img_paths[subj.subject_id] = None
    else:
        for subj in subjects:
            img_paths[subj.subject_id] = None

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = build_html(subjects, img_paths, generated_at)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"\n[REPORT] HTML saved → {out_html}")
    print(f"         Open:        file://{out_html.resolve()}")


if __name__ == "__main__":
    main()
