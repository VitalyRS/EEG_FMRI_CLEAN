addpath('/home/vitaly/Downloads/eeglab_current/eeglab2026.0.0');
addpath('/home/vitaly/Downloads/eeglab_current/eeglab2026.0.0/plugins/fMRIb2.1');
eeglab nogui;
S = load('/home/vitaly/PycharmProjects/Antigravity/EEG_FMRI_CLEAN/analysis_dima/_tmp/bb_obs_1924/in.mat');
EEG = eeg_emptyset();
EEG.data = double(S.data);
EEG.srate = double(S.srate);
EEG.nbchan = size(S.data,1);
EEG.pnts = size(S.data,2);
EEG.trials = 1;
EEG.xmin = 0;
EEG.xmax = (EEG.pnts-1)/EEG.srate;
EEG.chanlocs = struct([]);
for i = 1:EEG.nbchan
  EEG.chanlocs(i).labels = char(S.labels{i});
end
EEG = eeg_checkset(EEG);
load('/home/vitaly/PycharmProjects/Antigravity/EEG_FMRI_CLEAN/analysis_dima/_tmp/bb_obs_1924/dq.mat', 'qrs_lat');
for i = 1:length(qrs_lat)
  EEG.event(end+1).type = 'qrs';
  EEG.event(end).latency = qrs_lat(i);
end
EEG = eeg_checkset(EEG);
EEGc = pop_fmrib_pas(EEG, 'qrs', 'obs', 7);
clean = single(EEGc.data);
save('/home/vitaly/PycharmProjects/Antigravity/EEG_FMRI_CLEAN/analysis_dima/_tmp/bb_obs_1924/out_npc7.mat', 'clean', '-v7');
exit(0);
