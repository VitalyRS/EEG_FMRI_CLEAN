"""
Probe: can QRS-locked IC selection remove the residual BCG that ICLabel/ECG-corr miss?
Fits ICA on our 03_bcg output, ranks ICs by the fraction of their energy explained by the
QRS-locked average (evoked-energy ratio), removes the top-k and reports the cardiac-locked
residual (same metric as compare_dima.py) + occipital alpha retention.

Usage: python analysis_dima/ica_cardiac_probe.py 1916 drone -23
       (last arg = lag from compare_dima.csv, our_index = dima_index + lag)
"""
import sys, warnings
from pathlib import Path
import numpy as np
import mne
from scipy.signal import welch

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_dima as cd

sid, seg, lag = sys.argv[1], sys.argv[2], int(sys.argv[3])
N_COMP = int(sys.argv[4]) if len(sys.argv) > 4 else 35
dr, qrs, _, _, _ = cd.load_dima(sid, seg)
q = (qrs + lag).astype(int)
raw = mne.io.read_raw_fif(cd.DATA / sid / "derivatives" / "03_bcg" / seg / f"{seg}_bcg_clean.fif", preload=True).pick("eeg")
sf = raw.info["sfreq"]

ica = mne.preprocessing.ICA(n_components=N_COMP, method="fastica", random_state=0, max_iter=500)
ica.fit(raw.copy().filter(1, 40))
src = ica.get_sources(raw).get_data()

a, b = int(-0.2 * sf), int(0.8 * sf)
ratios = []
for c in range(src.shape[0]):
    ep = np.stack([src[c, i + a:i + b] for i in q if i + a >= 0 and i + b < src.shape[1]])
    ep = ep - ep.mean(axis=1, keepdims=True)
    ratios.append(float(np.mean(ep.mean(axis=0) ** 2) / np.mean(ep ** 2)))
order = np.argsort(ratios)[::-1]
print("top IC evoked-energy ratios:", [(int(c), round(ratios[c], 3)) for c in order[:10]])


def occ_alpha(x):
    occ = [raw.ch_names.index(c) for c in cd.OCC if c in raw.ch_names]
    f, p = welch(x[occ], sf, nperseg=int(4 * sf), axis=-1)
    return float(p[:, (f >= 8) & (f <= 13)].mean())


base = cd.to_band_car(raw)
a0 = occ_alpha(base)
print(f"Dima cardiac-locked residual: see compare csv (~0.5-1.9 uV)")
print(f"k=0  cardiac={cd.qrs_locked_rms(base, sf, q):.2f} uV  alpha_ret=1.00")
for k in (1, 2, 3, 4, 6, 8, 10):
    ica.exclude = [int(c) for c in order[:k]]
    cl = ica.apply(raw.copy())
    x = cd.to_band_car(cl)
    print(f"k={k:<2d} cardiac={cd.qrs_locked_rms(x, sf, q):.2f} uV  alpha_ret={occ_alpha(x) / a0:.2f}  removed={ica.exclude}")
