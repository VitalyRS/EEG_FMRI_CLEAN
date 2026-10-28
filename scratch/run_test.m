
addpath('/home/vitaly/PycharmProjects/Antigravity/EEG_FMRI_CLEAN/eeglab2024.0');
addpath('/home/vitaly/PycharmProjects/Antigravity/EEG_FMRI_CLEAN/eeglab2024.0/plugins/fmrib2.0');
eeglab nogui;
load('scratch/test_ecg_1916.mat');
EEG = eeg_emptyset();
EEG.data = double(data);
EEG.srate = 250.0;
EEG.nbchan = size(EEG.data, 1);
EEG.pnts = size(EEG.data, 2);
EEG.trials = 1;
EEG.xmin = 0;
EEG.xmax = (EEG.pnts-1)/EEG.srate;
EEG = eeg_checkset(EEG);

EEG_fm = pop_fmrib_qrsdetect(EEG, 32, 'qrs', 'no');
qrs_fm = [];
for i = 1:length(EEG_fm.event)
    if strcmp(EEG_fm.event(i).type, 'qrs')
        qrs_fm(end+1) = EEG_fm.event(i).latency;
    end
end
fprintf('FMRIB QRS count: %d
', length(qrs_fm));
save('scratch/qrs_fmrib_1916.mat', 'qrs_fm');
