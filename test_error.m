
addpath("/home/vitaly/Downloads/eeglab_current/eeglab2026.0.0");
addpath("/home/vitaly/Downloads/eeglab_current/eeglab2026.0.0/plugins/clean_rawdata");
addpath("/home/vitaly/Downloads/eeglab_current/eeglab2026.0.0/plugins/ICLabel");
addpath("/home/vitaly/Downloads/eeglab_current/eeglab2026.0.0/plugins/dipfit");
eeglab nogui;
try
    run("/home/vitaly/PycharmProjects/Antigravity/EEG_FMRI_CLEAN/data/1916/segments/segment04/optuna_ica_trials/trial_044.m");
catch ME
    disp("ERROR CAUGHT:");
    disp(ME.identifier);
    disp(ME.message);
    for k = 1:length(ME.stack)
        fprintf("  File %s, line %d, function %s\n", ME.stack(k).file, ME.stack(k).line, ME.stack(k).name);
    end
end
