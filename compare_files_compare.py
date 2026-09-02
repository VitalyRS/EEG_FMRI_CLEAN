#!/usr/bin/env python3
"""
Self-contained fif-vs-set comparison over the whole files_compare/ tree.

Auto-discovers every segment folder that holds BOTH our cleaned *.fif and the
reference EEGLAB *.set (side by side), then for each pair computes:
  - mean Welch PSD over the common EEG channels (ours vs reference)
  - band-power ratios ours/ref (delta..above-80)
  - estimated low-pass cutoff for each
  - simple cleaning-quality metrics (neural preservation, artifact
    suppression, SNR improvement, temporal stability) and a 0-100 score

Writes a single standalone HTML report with an overlaid PSD plot (inline SVG,
no external assets) per segment plus a summary table.

Usage:
    python compare_files_compare.py [files_compare] [out.html]
Defaults: files_compare/ -> reports/files_compare_report.html
"""
import sys
import base64
import io
from pathlib import Path

import numpy as np
import mne
from scipy.signal import welch, detrend

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BANDS = {
    "delta 1-4": (1.0, 4.0),
    "theta 4-8": (4.0, 8.0),
    "alpha 8-13": (8.0, 13.0),
    "beta 13-30": (13.0, 30.0),
    "low-gamma 30-45": (30.0, 45.0),
    "hi 45-80": (45.0, 80.0),
    "above-80": (80.0, 124.0),
}
NEURAL_BANDS = {"alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
ARTIFACT_BANDS = {"cardiac": (0.7, 4.0), "muscle": (30.0, 80.0)}

# time-domain snippets: which channels, which offsets (s), how wide (s)
TD_CHANNELS = ["O1", "O2", "Cz", "C3", "C4", "Fz", "Pz", "T7", "T8"]
TD_TIMES = [1.0, 50.0, 250.0]
TD_WINDOW = 2.0


# ---------------------------------------------------------------- io / dsp
def load(path):
    p = str(path)
    r = (mne.io.read_raw_eeglab(p, preload=True, verbose="ERROR") if p.endswith(".set")
         else mne.io.read_raw_fif(p, preload=True, verbose="ERROR"))
    r.pick("eeg")
    return r


def psd_mean(raw):
    sf = raw.info["sfreq"]
    d = raw.get_data()
    n = int(min(4 * sf, d.shape[-1]))
    f, p = welch(d, sf, nperseg=n, axis=-1)
    return f, p.mean(axis=0)


def band_power(f, p, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(np.trapz(p[m], f[m])) if m.any() else 0.0


def cutoff(f, p):
    """Highest freq whose PSD is still >=1% of the 10-70 Hz in-band max."""
    ref = (f >= 10) & (f <= 70)
    if not ref.any():
        return None
    thr = float(p[ref].max()) * 0.01
    a = f[(p >= thr) & (f >= 10) & (f <= 120)]
    return float(a.max()) if a.size else None


def temporal_std(raw, duration=60.0):
    sf = raw.info["sfreq"]
    n = int(min(duration * sf, raw.n_times))
    d = detrend(raw.get_data()[:, :n], axis=1)
    return float(np.mean(np.std(d, axis=1)) * 1e6)  # µV


def snippet(raw, ch, t_center, window):
    """(times_rel, values_uV) for a `window`-s slice centred at t_center, or None."""
    if ch not in raw.ch_names:
        return None
    sf = raw.info["sfreq"]
    t0 = max(0.0, t_center - window / 2)
    t1 = min(float(raw.times[-1]), t_center + window / 2)
    if t1 <= t0:
        return None
    a, b = int(t0 * sf), int(t1 * sf)
    d, t = raw.copy().pick([ch]).get_data(start=a, stop=b, return_times=True)
    return (t - t[0]), d[0] * 1e6  # µV


# ---------------------------------------------------------------- discovery
def discover(root):
    """Yield (subj, seg, fif, set) for every folder holding both files."""
    pairs = []
    for subj_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for seg_dir in sorted(p for p in subj_dir.iterdir() if p.is_dir()):
            fif = sorted(seg_dir.glob("*.fif"))
            st = sorted(seg_dir.glob("*.set"))
            pairs.append((subj_dir.name, seg_dir.name,
                          fif[0] if fif else None,
                          st[0] if st else None))
    return pairs


# ---------------------------------------------------------------- per pair
def analyse(subj, seg, fif, ref):
    o, r = load(fif), load(ref)
    common = [c for c in o.ch_names if c in r.ch_names]
    o.pick(common)
    r.pick(common)

    fo, po = psd_mean(o)
    fr, pr = psd_mean(r)

    bands = {}
    for name, (lo, hi) in BANDS.items():
        bo, br = band_power(fo, po, lo, hi), band_power(fr, pr, lo, hi)
        bands[name] = (bo / br) if br > 0 else None

    # quality metrics
    pw_o = {n: band_power(fo, po, lo, hi) for n, (lo, hi) in {**NEURAL_BANDS, **ARTIFACT_BANDS}.items()}
    pw_r = {n: band_power(fr, pr, lo, hi) for n, (lo, hi) in {**NEURAL_BANDS, **ARTIFACT_BANDS}.items()}
    neural = [pw_o[b] / pw_r[b] if pw_r[b] > 0 else 1.0 for b in NEURAL_BANDS]
    neural_pres = float(np.mean([min(x, 1 / x) for x in neural]))
    artifact = [pw_o[b] / pw_r[b] if pw_r[b] > 0 else 1.0 for b in ARTIFACT_BANDS]
    artifact_supp = float(np.mean(artifact))
    snr_o = np.mean([pw_o[b] for b in NEURAL_BANDS]) / np.mean([pw_o[b] for b in ARTIFACT_BANDS])
    snr_r = np.mean([pw_r[b] for b in NEURAL_BANDS]) / np.mean([pw_r[b] for b in ARTIFACT_BANDS])
    snr_impr = float(snr_o / snr_r) if snr_r > 0 else 1.0
    ts_o, ts_r = temporal_std(o), temporal_std(r)
    temp_score = float(ts_r / ts_o) if ts_o > 0 else 1.0
    score = float(
        30 * np.clip(neural_pres, 0, 1.0)
        + 30 * np.clip(2.0 - artifact_supp, 0, 1.0)
        + 20 * np.clip(snr_impr, 0, 2.0) / 2.0
        + 20 * np.clip(temp_score, 0, 2.0) / 2.0
    )
    winner = "ours" if score > 55 else ("ref" if score < 45 else "tie")

    png = plot_psd(fo, po, fr, pr, f"{subj}/{seg}")
    td_png = plot_timedomain(o, r, f"{subj}/{seg}")

    return {
        "subj": subj, "seg": seg, "nch": len(common),
        "dur": float(o.times[-1]),
        "muscle": "_ica_musle" in ref.name,
        "cut_ours": cutoff(fo, po), "cut_ref": cutoff(fr, pr),
        "bands": bands,
        "neural_pres": neural_pres, "artifact_supp": artifact_supp,
        "snr_impr": snr_impr, "ts_o": ts_o, "ts_r": ts_r,
        "temp_score": temp_score, "score": score, "winner": winner,
        "png": png, "td_png": td_png,
    }


def plot_psd(fo, po, fr, pr, title):
    fig = plt.figure(figsize=(9, 3.6))
    plt.semilogy(fo, po, label="ours (fif)", lw=1.2)
    plt.semilogy(fr, pr, label="ref (.set)", lw=1.2, alpha=0.8)
    plt.axvspan(8, 13, color="green", alpha=0.08)
    plt.axvline(80, color="red", ls="--", lw=0.8)
    plt.xlim(0, 120)
    plt.xlabel("Hz")
    plt.ylabel("PSD (V²/Hz)")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_timedomain(o, r, title):
    """
    Grid of time snippets: one row per channel, one column per time offset.
    Ours vs ref overlaid; every subplot autoscaled to its own µV range so
    small differences stay visible. Channels absent in either file are skipped.
    """
    chans = [c for c in TD_CHANNELS if c in o.ch_names and c in r.ch_names]
    dur = min(float(o.times[-1]), float(r.times[-1]))
    times = [t for t in TD_TIMES if t + TD_WINDOW / 2 <= dur]
    if not chans or not times:
        return None

    nr, nc = len(chans), len(times)
    fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 1.15 * nr),
                             squeeze=False, sharex="col")
    for i, ch in enumerate(chans):
        for j, t in enumerate(times):
            ax = axes[i][j]
            so = snippet(o, ch, t, TD_WINDOW)
            sr = snippet(r, ch, t, TD_WINDOW)
            if so is not None:
                ax.plot(so[0], so[1], lw=0.8, color="#3b82f6", label="ours")
            if sr is not None:
                ax.plot(sr[0], sr[1], lw=0.8, color="#e0803a", alpha=0.8, label="ref")
            # autoscale y to the union of both traces (robust to spikes: 1-99 pct)
            vals = np.concatenate([s[1] for s in (so, sr) if s is not None])
            if vals.size:
                lo, hi = np.percentile(vals, [1, 99])
                pad = max((hi - lo) * 0.15, 1.0)
                ax.set_ylim(lo - pad, hi + pad)
            ax.tick_params(labelsize=6, length=2)
            ax.margins(x=0)
            if j == 0:
                ax.set_ylabel(ch, fontsize=8, rotation=0, ha="right", va="center")
            if i == 0:
                ax.set_title(f"t={t:.0f}s", fontsize=8)
            if i == nr - 1:
                ax.set_xlabel("s", fontsize=7)
    axes[0][0].legend(fontsize=6, loc="upper right", framealpha=0.6)
    fig.suptitle(f"{title} — time domain (µV, per-panel scale)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=95)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- html
def ratio_class(r, band):
    if r is None:
        return "na"
    if any(x in band for x in ["alpha", "beta", "theta"]):
        return "good" if 0.7 <= r <= 1.5 else ("warn" if 0.5 <= r <= 2.5 else "bad")
    return "good" if 0.5 <= r <= 2.0 else ("warn" if 0.3 <= r <= 4.0 else "bad")


def cutoff_class(o, r):
    if o is None or r is None:
        return "na"
    d = abs(o - r)
    return "good" if d <= 5 else ("warn" if d <= 15 else "bad")


CSS = """
:root{--bg:#0f1419;--panel:#1a2028;--line:#2a333f;--text:#e2e8f0;--dim:#7a8798;
--good:#2f9e5b;--warn:#c9922b;--bad:#c0392b;--accent:#3b82f6;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;padding:24px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 10px;color:var(--dim)}
.sub{color:var(--dim);margin-bottom:20px}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin-bottom:12px}
th,td{border:1px solid var(--line);padding:5px 7px;text-align:center}
th{background:var(--panel);color:var(--dim);font-weight:600;position:sticky;top:0}
td.mono{font-family:ui-monospace,monospace;text-align:left;white-space:nowrap}
.num{font-family:ui-monospace,monospace}
.ratio-cell{position:relative}
.ratio-cell.good{background:rgba(47,158,91,.15)}
.ratio-cell.warn{background:rgba(201,146,43,.15)}
.ratio-cell.bad{background:rgba(192,57,43,.18)}
.ratio-cell.na{color:var(--dim)}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-family:ui-monospace,monospace;font-size:11.5px}
.good{background:rgba(47,158,91,.2);color:#5fd08a}
.warn{background:rgba(201,146,43,.2);color:#e0b45f}
.bad{background:rgba(192,57,43,.25);color:#f08579}
.na{background:rgba(122,135,152,.15);color:var(--dim)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:14px}
.card h3{margin:0 0 8px;font-size:14px;font-family:ui-monospace,monospace}
.card img{width:100%;border-radius:4px;background:#fff}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(460px,1fr));gap:14px}
.stack{display:flex;flex-direction:column;gap:14px}
.card.wide{width:100%}.card.wide img{width:100%}
.legend{font-size:12px;color:var(--dim);margin:6px 0 18px}
"""


def build_html(rows, skipped, root):
    band_keys = list(BANDS.keys())
    head = "".join(f"<th>{k.split()[0]}</th>" for k in band_keys)

    trs = ""
    for s in rows:
        cc = cutoff_class(s["cut_ours"], s["cut_ref"])
        cut = "–/–" if s["cut_ours"] is None or s["cut_ref"] is None \
            else f"{s['cut_ours']:.0f}/{s['cut_ref']:.0f}"
        cells = ""
        for bk in band_keys:
            r = s["bands"].get(bk)
            cls = ratio_class(r, bk)
            val = "–" if r is None else f"{r:.2f}"
            cells += f"<td class='ratio-cell {cls} num'>{val}</td>"
        sc = s["score"]
        scls = "good" if s["winner"] == "ours" else ("bad" if s["winner"] == "ref" else "warn")
        muscle = " <small style='color:var(--dim)'>+musle</small>" if s["muscle"] else ""
        trs += (
            f"<tr><td class='mono'>{s['subj']}/{s['seg']}{muscle}</td>"
            f"<td class='num'>{s['nch']}</td>"
            f"<td><span class='badge {cc}'>{cut}</span></td>"
            f"{cells}"
            f"<td class='num'>{s['neural_pres']:.2f}</td>"
            f"<td class='num'>{s['artifact_supp']:.2f}</td>"
            f"<td class='num'>{s['snr_impr']:.2f}</td>"
            f"<td class='num'>{s['ts_o']:.1f}/{s['ts_r']:.1f}</td>"
            f"<td><span class='badge {scls}'>{sc:.0f}</span></td></tr>\n"
        )

    def badge(s):
        cls = "good" if s["winner"] == "ours" else ("bad" if s["winner"] == "ref" else "warn")
        return f"<span class='badge {cls}'>{s['score']:.0f} {s['winner']}</span>"

    cards = ""
    for s in rows:
        cards += (
            f"<div class='card'><h3>{s['subj']}/{s['seg']} {badge(s)}</h3>"
            f"<img src='data:image/png;base64,{s['png']}'></div>\n"
        )

    td_cards = ""
    for s in rows:
        if not s.get("td_png"):
            continue
        td_cards += (
            f"<div class='card wide'><h3>{s['subj']}/{s['seg']} {badge(s)}</h3>"
            f"<img src='data:image/png;base64,{s['td_png']}'></div>\n"
        )

    n_ours = sum(r["winner"] == "ours" for r in rows)
    n_ref = sum(r["winner"] == "ref" for r in rows)
    n_tie = sum(r["winner"] == "tie" for r in rows)
    avg = np.mean([r["score"] for r in rows]) if rows else 0

    skip_html = ""
    if skipped:
        items = "".join(f"<li class='mono'>{a}/{b} — {why}</li>" for a, b, why in skipped)
        skip_html = f"<h2>Skipped ({len(skipped)})</h2><ul class='legend'>{items}</ul>"

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>files_compare — fif vs set</title><style>{CSS}</style></head><body>
<h1>fif vs set — {root}</h1>
<div class="sub">{len(rows)} comparable pairs &nbsp;·&nbsp; avg score {avg:.1f}/100
&nbsp;·&nbsp; ours {n_ours} / ref {n_ref} / tie {n_tie}</div>

<h2>Summary</h2>
<table><thead><tr>
<th>segment</th><th>nch</th><th>cut o/r</th>{head}
<th>neural<br>pres</th><th>artif<br>supp</th><th>SNR<br>impr</th>
<th>tstd µV<br>o/r</th><th>score</th>
</tr></thead><tbody>{trs}</tbody></table>
<div class="legend">
Band cells = ratio ours/ref (green≈match, red=far). cut o/r = est. low-pass edge (Hz).
neural pres 1.0=perfect · artif supp &lt;1=less artifact than ref · SNR impr &gt;1=better ·
tstd = 60 s detrended std (lower=steadier). Score &gt;55 ours, &lt;45 ref.
</div>

<h2>PSD overlays</h2>
<div class="grid">{cards}</div>

<h2>Time domain — 1 s / 50 s / 250 s snippets across channels</h2>
<div class="legend">Blue = ours (fif), orange = ref (.set). Each panel autoscaled to
its own µV range (robust 1–99 pct) so small shape differences stay visible;
the y-scale therefore differs panel to panel.</div>
<div class="stack">{td_cards}</div>
{skip_html}
</body></html>"""


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("files_compare")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("reports/files_compare_report.html")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, skipped = [], []
    for subj, seg, fif, ref in discover(root):
        if fif is None:
            skipped.append((subj, seg, "no fif"))
            continue
        if ref is None:
            skipped.append((subj, seg, "no reference .set"))
            continue
        try:
            r = analyse(subj, seg, fif, ref)
            rows.append(r)
            print(f"[ok]   {subj}/{seg:<9} score={r['score']:5.1f} {r['winner']:<4} "
                  f"cut {r['cut_ours']}/{r['cut_ref']}")
        except Exception as e:
            skipped.append((subj, seg, f"error: {e}"))
            print(f"[fail] {subj}/{seg}: {e}")

    out.write_text(build_html(rows, skipped, root))
    print(f"\nWrote {out}  ({len(rows)} pairs, {len(skipped)} skipped)")


if __name__ == "__main__":
    main()
