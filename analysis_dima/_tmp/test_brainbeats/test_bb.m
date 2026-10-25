
addpath("/home/vitaly/Downloads/eeglab_current/eeglab2026.0.0/plugins/BrainBeats/functions");
load("/home/vitaly/PycharmProjects/Antigravity/EEG_FMRI_CLEAN/analysis_dima/_tmp/test_brainbeats/ecg_in.mat");
params.fs = sfreq;
params.heart_signal = "ecg";
params.ecg_bandpass = [3 35];
params.ecg_peakthresh = 0.35;
params.ecg_searchback = true;
params.ecg_refperiod = 0.35;

[RR, RR_t, peaks, sig, t, pol, HR] = get_RR(ecg, times_ms, params);
save("/home/vitaly/PycharmProjects/Antigravity/EEG_FMRI_CLEAN/analysis_dima/_tmp/test_brainbeats/bb_peaks.mat", "peaks", "RR", "HR");
exit(0);
