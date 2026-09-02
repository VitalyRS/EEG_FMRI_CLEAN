#!/usr/bin/env python3
"""
Compute cleaning quality metrics comparing our FIF vs reference .set.

Metrics:
1. SNR improvement (signal-to-noise ratio in neural bands vs artifact bands)
2. Spectral distortion (how much neural bands were altered)
3. Artifact suppression (how much artifact bands were reduced)
4. Temporal stability (std of baseline in time domain)
5. Overall cleaning score (weighted composite)
"""
import json
from pathlib import Path
import numpy as np
import mne
from scipy.signal import welch

DATA = Path("data")
CMP = Path("files_compare")

# Frequency bands
NEURAL_BANDS = {
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
ARTIFACT_BANDS = {
    "cardiac": (0.7, 4.0),
    "muscle": (30.0, 80.0),
}

def load_raw(subj, seg, source='ours'):
    try:
        if source == 'ours':
            path = DATA / subj / 'derivatives' / '05_ica' / seg / f'{seg}_ica_clean.fif'
            return mne.io.read_raw_fif(path, preload=True, verbose='ERROR')
        else:
            cands = sorted((CMP / subj / seg).glob('*.set'))
            if not cands: return None
            return mne.io.read_raw_eeglab(cands[0], preload=True, verbose='ERROR')
    except:
        return None

def psd_mean(raw):
    """Compute mean PSD across all EEG channels"""
    raw = raw.copy().pick('eeg')
    sf = raw.info['sfreq']
    d = raw.get_data()
    n = int(min(4 * sf, d.shape[-1]))
    f, p = welch(d, sf, nperseg=n, axis=-1)
    return f, p.mean(axis=0)

def band_power(f, p, lo, hi):
    """Integrate power in frequency band"""
    m = (f >= lo) & (f <= hi)
    return float(np.trapz(p[m], f[m])) if m.any() else 0.0

def temporal_stability(raw, duration=60.0):
    """
    Measure baseline stability in time domain.
    Lower std = more stable (better cleaning).
    Uses 60s window to capture slow drift.
    """
    raw = raw.copy().pick('eeg')
    sf = raw.info['sfreq']
    n_samples = int(min(duration * sf, raw.n_times))
    data = raw.get_data()[:, :n_samples]

    # Detrend each channel then compute std across time
    from scipy.signal import detrend
    data_detrend = detrend(data, axis=1)

    # Mean std across channels (in µV)
    return float(np.mean(np.std(data_detrend, axis=1)) * 1e6)

def compute_metrics(subj, seg):
    """Compute all quality metrics for one segment"""
    print(f"  Computing metrics for {subj}/{seg}...")

    raw_o = load_raw(subj, seg, 'ours')
    raw_r = load_raw(subj, seg, 'ref')

    if raw_o is None or raw_r is None:
        print(f"    Skip: missing data")
        return None

    # Common channels only
    common = [c for c in raw_o.ch_names if c in raw_r.ch_names]
    raw_o.pick(common)
    raw_r.pick(common)

    # Compute PSDs
    f_o, p_o = psd_mean(raw_o)
    f_r, p_r = psd_mean(raw_r)

    # Band powers
    powers_o = {name: band_power(f_o, p_o, lo, hi) for name, (lo, hi) in {**NEURAL_BANDS, **ARTIFACT_BANDS}.items()}
    powers_r = {name: band_power(f_r, p_r, lo, hi) for name, (lo, hi) in {**NEURAL_BANDS, **ARTIFACT_BANDS}.items()}

    # Metric 1: Neural band preservation (closer to 1.0 = better)
    neural_ratios = [powers_o[b] / powers_r[b] if powers_r[b] > 0 else 1.0 for b in NEURAL_BANDS.keys()]
    neural_preservation = np.mean([min(r, 1/r) for r in neural_ratios])  # penalize both over and under

    # Metric 2: Artifact suppression (lower = better cleaning)
    artifact_ratios = [powers_o[b] / powers_r[b] if powers_r[b] > 0 else 1.0 for b in ARTIFACT_BANDS.keys()]
    artifact_suppression = np.mean(artifact_ratios)

    # Metric 3: SNR (neural/artifact ratio) - higher = better
    snr_o = np.mean([powers_o[b] for b in NEURAL_BANDS.keys()]) / np.mean([powers_o[b] for b in ARTIFACT_BANDS.keys()])
    snr_r = np.mean([powers_r[b] for b in NEURAL_BANDS.keys()]) / np.mean([powers_r[b] for b in ARTIFACT_BANDS.keys()])
    snr_improvement = snr_o / snr_r if snr_r > 0 else 1.0

    # Metric 4: Temporal stability (lower std = better)
    temporal_std_o = temporal_stability(raw_o)
    temporal_std_r = temporal_stability(raw_r)
    temporal_score = temporal_std_r / temporal_std_o if temporal_std_o > 0 else 1.0  # >1 means we're more stable

    # Overall score (weighted composite, 0-100 scale)
    # neural_preservation: 1.0=perfect, <1.0=bad
    # artifact_suppression: <1.0=good (we have less artifact), >1.0=bad
    # snr_improvement: >1.0=good, <1.0=bad
    # temporal_score: >1.0=good (more stable), <1.0=bad

    score = (
        30 * np.clip(neural_preservation, 0, 1.0) +           # 30% weight on preserving neural
        30 * np.clip(2.0 - artifact_suppression, 0, 1.0) +    # 30% weight on suppressing artifact (1.0-0.5 maps to 30-15)
        20 * np.clip(snr_improvement, 0, 2.0) / 2.0 +         # 20% weight on SNR improvement (1.0-2.0 maps to 10-20)
        20 * np.clip(temporal_score, 0, 2.0) / 2.0            # 20% weight on temporal stability
    )

    return {
        'subj': subj,
        'seg': seg,
        'neural_preservation': float(neural_preservation),
        'artifact_suppression': float(artifact_suppression),
        'snr_improvement': float(snr_improvement),
        'temporal_stability_ours': float(temporal_std_o),
        'temporal_stability_ref': float(temporal_std_r),
        'temporal_score': float(temporal_score),
        'overall_score': float(score),
        'winner': 'ours' if score > 55 else ('ref' if score < 45 else 'tie')
    }

def main():
    segments = [
        ('1916', 'ec'), ('1916', 'eo'), ('1916', 'drone'), ('1916', 'lasertag'),
        ('1925', 'ec'), ('1925', 'eo'), ('1925', 'drone'), ('1925', 'lasertag'),
    ]

    print("Computing cleaning quality metrics...")
    results = []

    for subj, seg in segments:
        metrics = compute_metrics(subj, seg)
        if metrics:
            results.append(metrics)
            print(f"    Score: {metrics['overall_score']:.1f}/100 ({metrics['winner']})")

    out_path = Path('/tmp/batch_cleaning_metrics.json')
    json.dump({'segments': results}, open(out_path, 'w'), indent=2)
    print(f"\n✅ Wrote {out_path} ({len(results)} segments)")

    # Summary
    avg_score = np.mean([r['overall_score'] for r in results])
    winners = [r['winner'] for r in results]
    print(f"\nSummary:")
    print(f"  Average score: {avg_score:.1f}/100")
    print(f"  Ours better: {winners.count('ours')}")
    print(f"  Reference better: {winners.count('ref')}")
    print(f"  Tie: {winners.count('tie')}")

if __name__ == '__main__':
    main()
