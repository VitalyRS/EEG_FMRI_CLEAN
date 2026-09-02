#!/usr/bin/env python3
"""
Extract time-domain snippets from FIF and .set files for visualization.
Extracts 1-second windows at t=1s, t=50s, t=250s for key channels.
"""
import json
from pathlib import Path
import numpy as np
import mne

CHANNELS = ['O1', 'O2', 'C1', 'C2', 'C3', 'C4', 'F3', 'F4', 'P3', 'P4', 'T7', 'T8']
TIME_POINTS = [1.0, 50.0, 250.0]  # seconds
WINDOW = 1.0  # 1 second window

def extract_snippet(raw, channel, t_center, window=1.0):
    """Extract 1-second window centered at t_center for given channel"""
    if channel not in raw.ch_names:
        return None

    sfreq = raw.info['sfreq']
    t_start = max(0, t_center - window/2)
    t_stop = min(raw.times[-1], t_center + window/2)

    start_idx = int(t_start * sfreq)
    stop_idx = int(t_stop * sfreq)

    data, times = raw.copy().pick([channel]).get_data(
        start=start_idx,
        stop=stop_idx,
        return_times=True
    )

    # Relative times (0 to window duration)
    times_rel = times - times[0]

    return {
        'times': times_rel.tolist(),
        'values': data[0].tolist(),
        'sfreq': float(sfreq)
    }

def load_raw(subj, seg, source='ours'):
    try:
        if source == 'ours':
            path = Path('data') / subj / 'derivatives' / '05_ica' / seg / f'{seg}_ica_clean.fif'
            return mne.io.read_raw_fif(path, preload=True, verbose='ERROR')
        else:  # reference
            cands = sorted((Path('files_compare') / subj / seg).glob('*.set'))
            if not cands:
                return None
            return mne.io.read_raw_eeglab(cands[0], preload=True, verbose='ERROR')
    except Exception as e:
        print(f"  Error loading {source} for {subj}/{seg}: {e}")
        return None

def main():
    segments_tokens = ['1916:ec', '1916:eo', '1916:drone', '1916:lasertag',
                       '1925:ec', '1925:eo', '1925:drone', '1925:lasertag']

    out = {'segments': []}

    for tok in segments_tokens:
        subj, seg = tok.split(':')
        print(f'Processing {tok}...')

        raw_o = load_raw(subj, seg, 'ours')
        raw_r = load_raw(subj, seg, 'ref')

        if raw_o is None or raw_r is None:
            print(f'  Skip {tok}: missing data')
            continue

        seg_data = {'subj': subj, 'seg': seg, 'timepoints': []}

        for t in TIME_POINTS:
            if t > raw_o.times[-1] or t > raw_r.times[-1]:
                print(f'  Skip t={t}s: beyond duration')
                continue

            tp_data = {'t': t, 'channels': []}

            for ch in CHANNELS:
                snip_o = extract_snippet(raw_o, ch, t, WINDOW)
                snip_r = extract_snippet(raw_r, ch, t, WINDOW)

                if snip_o and snip_r:
                    tp_data['channels'].append({
                        'name': ch,
                        'ours': snip_o,
                        'ref': snip_r
                    })

            if tp_data['channels']:
                seg_data['timepoints'].append(tp_data)
                print(f'  t={t}s: {len(tp_data["channels"])} channels')

        if seg_data['timepoints']:
            out['segments'].append(seg_data)

    out_path = Path('/tmp/batch_timeseries.json')
    json.dump(out, open(out_path, 'w'))
    print(f'\n✅ Wrote {out_path} ({len(out["segments"])} segments)')

if __name__ == '__main__':
    main()
