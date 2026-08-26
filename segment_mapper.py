"""
segment_mapper.py — EEG ↔ fMRI Subject & Segment Discovery
===========================================================
Matches subjects (by 4-digit code: 1916, 1917, …) across:
  • EEG folder  (.../eeg-fmri/original/XXXXEEG/)
  • fMRI folder (.../fmri/XXXXFMRI/)

Parses BrainVision .vmrk markers to derive segment boundaries from
the continuous inside.eeg and outside.eeg recordings.

Segment layout (inside.eeg, vmrk pattern: ec→eo→ec→eo→…→ec→eo):
  ec        (301): Mk[ec,1] → Mk[eo,1]          ← first ec/eo pair
  eo        (401): Mk[ec,2] → Mk[eo,2]           ← second ec/eo pair
  drone     (501): Mk[eo,2] → Mk[eo,2]+rp501_dur ← by rp file line count × TR
  lasertag  (601): after drone → +rp601_dur
  video     (701): after lasertag → Mk[ec,3]      ← up to next ec marker
  ec_out    (–):   outside.vmrk Mk[ec,1] → Mk[eo,1]
  eo_out    (–):   outside.vmrk Mk[eo,1] → end-of-file

Usage (standalone):
  python segment_mapper.py               # prints table for all subjects
  python segment_mapper.py --subject 1916
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration: override via environment variables if needed
# ---------------------------------------------------------------------------
EEG_ORIGINAL_ROOT = Path(os.getenv(
    "EEG_ORIGINAL_ROOT",
    "/media/vitaly/48DEA853CCEBFBF0/0DATDA_2026_eeg_fnri/fmri/eeg-fmri/original"
))
FMRI_ROOT = Path(os.getenv(
    "FMRI_ROOT",
    "/media/vitaly/48DEA853CCEBFBF0/0DATDA_2026_eeg_fnri/fmri/fmri"
))
PROCESSED_ROOT = Path(os.getenv(
    "PROCESSED_ROOT",
    str(Path(__file__).parent / "data")
))

# fMRI acquisition parameters (same for all EEG-fMRI sessions)
TR_SEC: float = 2.5
SLICES_PER_VOLUME: int = 25
DUMMY_VOLUMES: int = 12

# rp session-number → segment name mapping
RP_SUFFIX_MAP = {
    "301": "ec",
    "401": "eo",
    "501": "drone",
    "601": "lasertag",
    "701": "video",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SegmentInfo:
    """All information needed to run the pipeline for one EEG segment."""
    subject_id: str          # e.g. "1916"
    name: str                # e.g. "ec", "drone", "ec_out"
    t_start_sec: float       # start in the continuous .eeg file (seconds)
    t_stop_sec: float        # stop  in the continuous .eeg file (seconds)
    vhdr_path: Path          # path to the .vhdr of the recording (inside or outside)
    rp_path: Optional[Path]  # SPM motion parameter file; None for outside segments
    sfreq: float             # sampling frequency of the recording
    segment_dir: Path        # output working directory for this segment
    needs_bergen: bool       # True → inside recording, rp_path is not None
    session_suffix: str      # "301", "401", … or "" for outside segments

    @property
    def duration_sec(self) -> float:
        return self.t_stop_sec - self.t_start_sec


@dataclass
class SubjectInfo:
    """All segments for one subject."""
    subject_id: str
    eeg_dir: Path
    fmri_dir: Optional[Path]
    segments: list[SegmentInfo] = field(default_factory=list)


# ---------------------------------------------------------------------------
# vmrk parsing
# ---------------------------------------------------------------------------

def parse_vmrk(vmrk_path: Path) -> list[dict]:
    """
    Parse a BrainVision .vmrk file and return a list of marker dicts:
      {"type": str, "description": str, "position": int}

    Position is in data points (1-based, as stored in the file).
    """
    markers = []
    with open(vmrk_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip().strip("\r")
            if not line.startswith("Mk"):
                continue
            # Format: MkN=Type,Description,Position,Size,Channel[,Date]
            m = re.match(r"Mk\d+=([^,]*),([^,]*),(\d+)", line)
            if m:
                markers.append({
                    "type":        m.group(1),
                    "description": m.group(2).strip(),
                    "position":    int(m.group(3)),   # 1-based data points
                })
    return markers


def get_sfreq_from_vhdr(vhdr_path: Path) -> float:
    """Read sampling frequency from a BrainVision .vhdr header file."""
    with open(vhdr_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.lower().startswith("samplinginterval"):
                # SamplingInterval is in microseconds
                val = float(line.split("=", 1)[1].strip())
                return 1_000_000.0 / val
    raise ValueError(f"SamplingInterval not found in {vhdr_path}")


def get_eeg_duration_samples(vhdr_path: Path) -> int:
    """
    Return total number of samples in the .eeg binary via the header.
    Reads DataPoints if present, otherwise derives from file size.
    """
    vhdr = vhdr_path
    eeg_file = vhdr.with_suffix(".eeg")
    with open(vhdr, encoding="utf-8", errors="replace") as fh:
        content = fh.read()

    # Try DataPoints field
    m = re.search(r"DataPoints\s*=\s*(\d+)", content, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Count channels from NumberOfChannels
    m_ch = re.search(r"NumberOfChannels\s*=\s*(\d+)", content, re.IGNORECASE)
    n_ch = int(m_ch.group(1)) if m_ch else 96

    # Check binary format: INT_16 (2 bytes) or IEEE_FLOAT_32 (4 bytes)
    bytes_per_sample = 2
    if "IEEE_FLOAT_32" in content or "FLOAT32" in content.upper():
        bytes_per_sample = 4

    if eeg_file.exists():
        return eeg_file.stat().st_size // (n_ch * bytes_per_sample)
    return 0


def get_rp_n_volumes(rp_path: Path) -> int:
    """Return number of volumes (rows) in an SPM rp_*.txt file."""
    return int(np.loadtxt(rp_path).shape[0])


# ---------------------------------------------------------------------------
# rp file discovery
# ---------------------------------------------------------------------------

_RP_SUFFIX_RE = re.compile(r"_(\d{3})_")


def find_rp_files(fmri_dir: Path) -> dict[str, Path]:
    """
    Return a dict mapping session suffix → rp file path.
    E.g. {"301": Path("rp_1916FMRI-EEG_301_...txt"), "401": …}
    Only session suffixes in RP_SUFFIX_MAP are included.
    """
    result: dict[str, Path] = {}
    if not fmri_dir or not fmri_dir.exists():
        return result
    for f in sorted(fmri_dir.glob("rp_*.txt")):
        m = _RP_SUFFIX_RE.search(f.name)
        if m and m.group(1) in RP_SUFFIX_MAP:
            result[m.group(1)] = f
    return result


# ---------------------------------------------------------------------------
# Core mapping logic
# ---------------------------------------------------------------------------

def map_segments_for_subject(subject_id: str,
                              eeg_dir: Path,
                              fmri_dir: Optional[Path],
                              processed_root: Path = PROCESSED_ROOT) -> list[SegmentInfo]:
    """
    Build the full list of SegmentInfo for one subject.
    If segment_info.json files already exist under processed_root/<subject_id>/segments/,
    loads them directly (e.g. created by step01_detect_mri).
    """
    segments: list[SegmentInfo] = []
    subject_processed = processed_root / subject_id
    seg_base = subject_processed / "segments"

    # If segment_info.json files were already generated by step01 or step00, load them directly!
    if seg_base.exists():
        existing_jsons = sorted(seg_base.glob("*/segment_info.json"))
        if existing_jsons:
            for info_file in existing_jsons:
                try:
                    with open(info_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    seg_name = data.get("segment_name", data.get("name", info_file.parent.name))
                    vhdr_p = Path(data.get("raw_vhdr", eeg_dir / f"{subject_id}_inside.vhdr"))
                    rp_p = Path(data["rp_file"]) if data.get("rp_file") else (Path(data["rp_path"]) if data.get("rp_path") else None)
                    segments.append(SegmentInfo(
                        subject_id=subject_id,
                        name=seg_name,
                        t_start_sec=float(data["t_start_sec"]),
                        t_stop_sec=float(data["t_stop_sec"]),
                        vhdr_path=vhdr_p,
                        rp_path=rp_p,
                        sfreq=float(data.get("sfreq", 5000.0)),
                        segment_dir=info_file.parent,
                        needs_bergen=data.get("needs_bergen", rp_p is not None),
                        session_suffix=str(data.get("session_suffix", ""))
                    ))
                except Exception as e:
                    print(f"  [WARN] Failed to load {info_file}: {e}")
            if segments:
                return segments

    # ------------------------------------------------------------------ inside
    inside_vhdr = eeg_dir / f"{subject_id}_inside.vhdr"
    inside_vmrk = eeg_dir / f"{subject_id}_inside.vmrk"

    if inside_vhdr.exists() and inside_vmrk.exists():
        sfreq = get_sfreq_from_vhdr(inside_vhdr)
        total_samples = get_eeg_duration_samples(inside_vhdr)
        total_sec = total_samples / sfreq

        rp_files = find_rp_files(fmri_dir)
        markers = parse_vmrk(inside_vmrk)

        # Collect ec/eo markers in order of appearance
        ec_markers = [mk for mk in markers if mk["description"].lower() == "ec"]
        eo_markers = [mk for mk in markers if mk["description"].lower() == "eo"]

        # ---- ec segment (301): first ec → first eo ---------------------------
        if len(ec_markers) >= 1 and len(eo_markers) >= 1 and "301" in rp_files:
            t0 = (ec_markers[0]["position"] - 1) / sfreq   # convert to 0-based seconds
            t1 = (eo_markers[0]["position"] - 1) / sfreq
            seg = SegmentInfo(
                subject_id=subject_id, name="ec",
                t_start_sec=t0, t_stop_sec=t1,
                vhdr_path=inside_vhdr, rp_path=rp_files["301"],
                sfreq=sfreq,
                segment_dir=seg_base / "ec",
                needs_bergen=True, session_suffix="301"
            )
            segments.append(seg)
        elif "301" not in rp_files:
            print(f"  [{subject_id}] SKIP ec: no rp_301 file in {fmri_dir}")
        else:
            print(f"  [{subject_id}] SKIP ec: insufficient ec/eo markers in vmrk")

        # ---- eo segment (401): second ec → second eo -------------------------
        if len(ec_markers) >= 2 and len(eo_markers) >= 2 and "401" in rp_files:
            t0 = (ec_markers[1]["position"] - 1) / sfreq
            t1 = (eo_markers[1]["position"] - 1) / sfreq
            seg = SegmentInfo(
                subject_id=subject_id, name="eo",
                t_start_sec=t0, t_stop_sec=t1,
                vhdr_path=inside_vhdr, rp_path=rp_files["401"],
                sfreq=sfreq,
                segment_dir=seg_base / "eo",
                needs_bergen=True, session_suffix="401"
            )
            segments.append(seg)
        elif "401" not in rp_files:
            print(f"  [{subject_id}] SKIP eo: no rp_401 file in {fmri_dir}")
        else:
            print(f"  [{subject_id}] SKIP eo: insufficient ec/eo markers")

        # ---- drone / lasertag / video (501/601/701): after second eo ---------
        # These three sessions follow consecutively, separated by their rp durations.
        # cursor_sec advances by rp_duration OR to EEG end (whichever comes first).
        # A segment is SKIPPED only if its t_start is already at/past the EEG end.
        if len(eo_markers) >= 2:
            cursor_sec = (eo_markers[1]["position"] - 1) / sfreq
            for suffix in ("501", "601", "701"):
                seg_name = RP_SUFFIX_MAP[suffix]
                if suffix not in rp_files:
                    print(f"  [{subject_id}] SKIP {seg_name}: no rp_{suffix} file")
                    cursor_sec = None
                    break
                n_vols = get_rp_n_volumes(rp_files[suffix])
                duration = n_vols * TR_SEC
                t0 = cursor_sec
                t1_full = cursor_sec + duration
                # Clamp stop to actual recording length
                t1_clamped = min(t1_full, total_sec)
                actual_dur = t1_clamped - t0

                if t0 >= total_sec:
                    # Segment starts after recording has already ended
                    print(f"  [{subject_id}] SKIP {seg_name}: t_start ({t0:.1f}s) >= "
                          f"recording end ({total_sec:.1f}s)")
                    cursor_sec = t1_full  # advance fictive cursor so next suffix is consistent
                    continue
                if actual_dur < 10.0:
                    print(f"  [{subject_id}] SKIP {seg_name}: only {actual_dur:.1f}s usable "
                          f"(need >10s); rp had {n_vols} vols × {TR_SEC}s = {duration:.0f}s")
                    cursor_sec = t1_clamped
                    continue

                if t1_clamped < t1_full:
                    print(f"  [{subject_id}] NOTE  {seg_name}: recording ended early — "
                          f"using {actual_dur:.1f}s of {duration:.0f}s "
                          f"({n_vols} vols expected, ~{int(actual_dur/TR_SEC)} usable)")

                seg = SegmentInfo(
                    subject_id=subject_id, name=seg_name,
                    t_start_sec=t0, t_stop_sec=t1_clamped,
                    vhdr_path=inside_vhdr, rp_path=rp_files[suffix],
                    sfreq=sfreq,
                    segment_dir=seg_base / seg_name,
                    needs_bergen=True, session_suffix=suffix
                )
                segments.append(seg)
                cursor_sec = t1_clamped  # always clamp cursor to real end
        else:
            print(f"  [{subject_id}] SKIP drone/lasertag/video: need >=2 eo markers in vmrk")

    else:
        if not inside_vhdr.exists():
            print(f"  [{subject_id}] SKIP inside segments: {inside_vhdr.name} not found")

    # ---------------------------------------------------------------- outside
    outside_vhdr = eeg_dir / f"{subject_id}_outside.vhdr"
    outside_vmrk = eeg_dir / f"{subject_id}_outside.vmrk"

    if outside_vhdr.exists() and outside_vmrk.exists():
        sfreq_out = get_sfreq_from_vhdr(outside_vhdr)
        total_samples_out = get_eeg_duration_samples(outside_vhdr)
        total_sec_out = total_samples_out / sfreq_out

        markers_out = parse_vmrk(outside_vmrk)
        ec_out_marks = [mk for mk in markers_out if mk["description"].lower() == "ec"]
        eo_out_marks = [mk for mk in markers_out if mk["description"].lower() == "eo"]

        # ec_out: first ec → first eo
        if ec_out_marks and eo_out_marks:
            t0 = (ec_out_marks[0]["position"] - 1) / sfreq_out
            t1 = (eo_out_marks[0]["position"] - 1) / sfreq_out
            if t1 > t0:
                segments.append(SegmentInfo(
                    subject_id=subject_id, name="ec_out",
                    t_start_sec=t0, t_stop_sec=t1,
                    vhdr_path=outside_vhdr, rp_path=None,
                    sfreq=sfreq_out,
                    segment_dir=seg_base / "ec_out",
                    needs_bergen=False, session_suffix=""
                ))

        # eo_out: first eo → end of recording
        if eo_out_marks:
            t0 = (eo_out_marks[0]["position"] - 1) / sfreq_out
            t1 = total_sec_out
            if t1 > t0:
                segments.append(SegmentInfo(
                    subject_id=subject_id, name="eo_out",
                    t_start_sec=t0, t_stop_sec=t1,
                    vhdr_path=outside_vhdr, rp_path=None,
                    sfreq=sfreq_out,
                    segment_dir=seg_base / "eo_out",
                    needs_bergen=False, session_suffix=""
                ))
            else:
                print(f"  [{subject_id}] SKIP eo_out: eo marker at {t0:.1f}s "
                      f">= file end {t1:.1f}s")
    else:
        print(f"  [{subject_id}] SKIP outside segments: outside.vhdr/.vmrk not found")

    return segments


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

_SUBJ_RE = re.compile(r"^(\d{4})EEG$", re.IGNORECASE)
_FMRI_RE = re.compile(r"^(\d{4})FMRI$", re.IGNORECASE)


def discover_subjects(eeg_root: Path = EEG_ORIGINAL_ROOT,
                      fmri_root: Path = FMRI_ROOT,
                      processed_root: Path = PROCESSED_ROOT) -> list[SubjectInfo]:
    """
    Scan the EEG root for XXXXeeg folders and match to XXXXfmri folders.
    Returns a sorted list of SubjectInfo (only subjects with EEG data included).
    Subjects with fMRI folder missing still get outside segments.
    """
    # Collect EEG subject IDs
    eeg_subjects: dict[str, Path] = {}
    if eeg_root.exists():
        for d in sorted(eeg_root.iterdir()):
            if d.is_dir():
                m = _SUBJ_RE.match(d.name)
                if m:
                    eeg_subjects[m.group(1)] = d

    # Collect fMRI dirs
    fmri_dirs: dict[str, Path] = {}
    if fmri_root.exists():
        for d in sorted(fmri_root.iterdir()):
            if d.is_dir():
                m = _FMRI_RE.match(d.name)
                if m:
                    fmri_dirs[m.group(1)] = d

    subjects: list[SubjectInfo] = []
    for sid in sorted(eeg_subjects):
        fmri_dir = fmri_dirs.get(sid)
        if not fmri_dir:
            print(f"[WARN] Subject {sid}: EEG found but no matching fMRI folder — "
                  "will process outside segments only")
        info = SubjectInfo(
            subject_id=sid,
            eeg_dir=eeg_subjects[sid],
            fmri_dir=fmri_dir,
        )
        info.segments = map_segments_for_subject(
            sid, eeg_subjects[sid], fmri_dir, processed_root
        )
        subjects.append(info)

    return subjects


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def _print_table(subjects: list[SubjectInfo]) -> None:
    print()
    print(f"{'Subject':<10} {'Segment':<12} {'Start(s)':>9} {'Stop(s)':>9} "
          f"{'Dur(s)':>7} {'Bergen':^8} {'rp file'}")
    print("-" * 90)
    for subj in subjects:
        for seg in subj.segments:
            rp_name = seg.rp_path.name if seg.rp_path else "n/a"
            bergen  = "YES" if seg.needs_bergen else "no"
            print(f"{seg.subject_id:<10} {seg.name:<12} "
                  f"{seg.t_start_sec:>9.1f} {seg.t_stop_sec:>9.1f} "
                  f"{seg.duration_sec:>7.1f} {bergen:^8} {rp_name}")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EEG-fMRI segment mapping table")
    parser.add_argument("--subject", default=None, help="Filter to single subject ID")
    args = parser.parse_args()

    all_subjects = discover_subjects()
    if args.subject:
        all_subjects = [s for s in all_subjects if s.subject_id == args.subject]
    if not all_subjects:
        print("No subjects found.")
    else:
        _print_table(all_subjects)
