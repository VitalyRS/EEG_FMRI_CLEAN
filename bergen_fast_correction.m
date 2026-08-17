function EEG = bergen_fast_correction(EEG, W, Peak_references, onset_val, offset_val)
% BERGEN_FAST_CORRECTION
% Vectorized, high-speed, mathematically exact implementation
% of the Bergen EEG-fMRI Toolbox artifact subtraction algorithm.
%
% Bergen formula:
%   A(i, :)        = EEG.data(ch, Peak(i)+onset : Peak(i)+offset)
%   Template(i, :) = (W(i, :) / sum(W(i, :))) * A
%   Cleaned(i, :)  = A(i, :) - Template(i, :)

n_channels = size(EEG.data, 1);
n_slices   = length(Peak_references);
slice_len  = offset_val - onset_val + 1;

% Normalize weight matrix rows
W_sum = sum(W, 2);
W_sum(W_sum == 0) = 1;
W_norm = W ./ W_sum;

% Check slice boundaries
valid_slices = (Peak_references + onset_val >= 1) & (Peak_references + offset_val <= EEG.pnts);
valid_idx    = find(valid_slices);
n_valid      = length(valid_idx);

if n_valid < n_slices
    fprintf('Warning: %d slices are out of bounds and will be skipped.\n', n_slices - n_valid);
end

% Index matrix for all valid slices [slice_len x n_valid]
offsets           = (onset_val : offset_val)';
sample_indices    = offsets + Peak_references(valid_idx); % [slice_len x n_valid]
sample_indices_1d = sample_indices(:);

% Sub-matrix of weights for valid slices only
W_sub     = W_norm(valid_idx, valid_idx);
W_sub_sum = sum(W_sub, 2);
W_sub_sum(W_sub_sum == 0) = 1;
W_sub = W_sub ./ W_sub_sum;

% Process channel by channel
for ch = 1:n_channels
    % Read all slices for this channel into matrix A [n_valid x slice_len]
    ch_data = EEG.data(ch, :);
    A_cols  = ch_data(sample_indices); % [slice_len x n_valid]
    A       = A_cols';                 % [n_valid x slice_len]

    % Motion-weighted moving-average template (Bergen AAS)
    Template     = W_sub * A;          % [n_valid x slice_len]

    % Subtract artifact template
    Cleaned      = A - Template;       % [n_valid x slice_len]
    Cleaned_cols = Cleaned';           % [slice_len x n_valid]

    % Write back into 1D EEG array (column-major order matches MATLAB)
    EEG.data(ch, sample_indices_1d) = Cleaned_cols(:)';
end

end
