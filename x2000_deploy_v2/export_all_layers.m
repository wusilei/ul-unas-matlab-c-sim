% export_all_layers.m — Export golden intermediate outputs from MATLAB inference
% Runs Main_infer.m with save points after each module, exports int32/int16 binary files
% for bit-exact comparison with C implementation.

clear; clc;

script_dir = fileparts(mfilename('fullpath'));
parent_dir = fullfile(script_dir, '..');

% Add all necessary paths (use absolute paths for robustness)
addpath(fullfile(parent_dir, 'para_in_mat_FP'));
addpath(fullfile(parent_dir, 'test_wavs'));
addpath(parent_dir);  % For STFT_func, log_gen, BM_module, etc.

out_dir = script_dir;
golden_dir = fullfile(out_dir, 'golden');
if ~exist(golden_dir, 'dir')
    mkdir(golden_dir);
end

%% Load ERB weights
erbfc_weight = importdata('erb_erb_fc_weight.mat');
ierbfc_weight = importdata('erb_ierb_fc_weight.mat');

%% Load noisy audio and run STFT
[noisy_audio, fs] = audioread('noisy_fileid_1.wav');
fprintf('Audio: %d samples, fs=%d Hz\n', length(noisy_audio), fs);

N_fft = 512;
win_len = 512;
win_inc = 256;
hann_window = importdata('stft_window.mat');

[cmp_real, cmp_imag] = STFT_func(noisy_audio.', N_fft, win_len, win_inc, hann_window);
T = size(cmp_real, 1);
fprintf('Frames: %d\n', T);

%% Process only first frame for golden export (or more if needed)
total_frames_to_export = min(T, 10);  % Export first 10 frames for testing

%% Initialize caches (same as Main_infer.m)
conv_cache_e0 = zeros(2,129);
conv_cache_e1 = zeros(24,65);
conv_cache_e2 = zeros(24,33);
conv_cache_d0 = zeros(24,33);
conv_cache_d1 = zeros(12,33);
conv_cache_d2 = zeros(12,2,65);

tfa_cache_e0 = zeros(1,24);
tfa_cache_e1 = zeros(1,48);
tfa_cache_e2 = zeros(1,48);
tfa_cache_e3 = zeros(1,64);
tfa_cache_e4 = zeros(1,32);
tfa_cache_d0 = zeros(1,64);
tfa_cache_d1 = zeros(1,48);
tfa_cache_d2 = zeros(1,48);
tfa_cache_d3 = zeros(1,24);
tfa_cache_d4 = zeros(1,2);

inter_cache_0 = zeros(33,16);
inter_cache_1 = zeros(33,16);

%% Process frames and export
for t = 1:total_frames_to_export
    fprintf('\n=== Frame %d/%d ===\n', t, total_frames_to_export);
    prefix = sprintf('frame%03d', t);

    %% Log-magnitude Compression
    spec_real = Fix_point(cmp_real(t,:), 's32f20');
    spec_imag = Fix_point(cmp_imag(t,:), 's32f20');

    x = log_gen(spec_real, spec_imag);
    export_binary(fullfile(golden_dir, [prefix '_log_gen.bin']), x, 'int32');

    %% BM
    x_bm = BM_module(x, erbfc_weight);
    export_binary(fullfile(golden_dir, [prefix '_bm.bin']), x_bm, 'int32');

    %% Encoder
    [y_e0, conv_cache_e0, tfa_cache_e0, ...
        y_e1, conv_cache_e1, tfa_cache_e1, ...
        y_e2, conv_cache_e2, tfa_cache_e2, ...
        y_e3, tfa_cache_e3, ...
        y_e4, tfa_cache_e4] = Encoder_module(x_bm, ...
        conv_cache_e0, tfa_cache_e0, ...
        conv_cache_e1, tfa_cache_e1, ...
        conv_cache_e2, tfa_cache_e2, ...
        tfa_cache_e3, ...
        tfa_cache_e4);

    export_binary(fullfile(golden_dir, [prefix '_e0.bin']), y_e0, 'int32');
    export_binary(fullfile(golden_dir, [prefix '_e1.bin']), y_e1, 'int32');
    export_binary(fullfile(golden_dir, [prefix '_e2.bin']), y_e2, 'int32');
    export_binary(fullfile(golden_dir, [prefix '_e3.bin']), y_e3, 'int32');
    export_binary(fullfile(golden_dir, [prefix '_e4.bin']), y_e4, 'int32');

    %% GDPRNN
    [y_rnn1, inter_cache_0] = GDPRNN_module(y_e4, inter_cache_0, 0);
    export_binary(fullfile(golden_dir, [prefix '_rnn1.bin']), y_rnn1, 'int32');

    [y_rnn2, inter_cache_1] = GDPRNN_module(y_rnn1, inter_cache_1, 1);
    export_binary(fullfile(golden_dir, [prefix '_rnn2.bin']), y_rnn2, 'int32');

    %% Decoder
    [y_dec, ...
        tfa_cache_d0, ...
        tfa_cache_d1, ...
        conv_cache_d0, tfa_cache_d2, ...
        conv_cache_d1, tfa_cache_d3, ...
        conv_cache_d2, tfa_cache_d4] = Decoder_module(y_rnn2, y_e4, tfa_cache_d0, ...
        y_e3, tfa_cache_d1, ...
        y_e2, conv_cache_d0, tfa_cache_d2, ...
        y_e1, conv_cache_d1, tfa_cache_d3, ...
        y_e0, conv_cache_d2, tfa_cache_d4);

    export_binary(fullfile(golden_dir, [prefix '_dec.bin']), y_dec, 'int32');

    %% Sigmoid
    y_dec_dq = y_dec*2^(-20);
    y_sig_dq = sigmoid_func(y_dec_dq);
    y_sig = Fix_point(y_sig_dq, 'u16f15');
    export_binary(fullfile(golden_dir, [prefix '_sig.bin']), y_sig, 'uint16');

    %% BS
    y_bs = BS_module(y_sig, ierbfc_weight);
    export_binary(fullfile(golden_dir, [prefix '_bs.bin']), y_bs, 'int32');

    %% MASK
    y_mask = MASK_module(y_bs, spec_real, spec_imag);
    export_binary(fullfile(golden_dir, [prefix '_mask_r.bin']), y_mask(1,:), 'int32');
    export_binary(fullfile(golden_dir, [prefix '_mask_i.bin']), y_mask(2,:), 'int32');

    %% State caches (first frame only — for initial state validation)
    if t == 1
        export_binary(fullfile(golden_dir, 'state_conv_cache_e0.bin'), conv_cache_e0, 'int32');
        export_binary(fullfile(golden_dir, 'state_conv_cache_e1.bin'), conv_cache_e1, 'int32');
        export_binary(fullfile(golden_dir, 'state_conv_cache_e2.bin'), conv_cache_e2, 'int32');
        export_binary(fullfile(golden_dir, 'state_tfa_cache_e0.bin'), tfa_cache_e0, 'int16');
        export_binary(fullfile(golden_dir, 'state_tfa_cache_e1.bin'), tfa_cache_e1, 'int16');
        export_binary(fullfile(golden_dir, 'state_tfa_cache_e2.bin'), tfa_cache_e2, 'int16');
        export_binary(fullfile(golden_dir, 'state_tfa_cache_e3.bin'), tfa_cache_e3, 'int16');
        export_binary(fullfile(golden_dir, 'state_tfa_cache_e4.bin'), tfa_cache_e4, 'int16');
        export_binary(fullfile(golden_dir, 'state_inter_cache_0.bin'), inter_cache_0, 'int16');
        export_binary(fullfile(golden_dir, 'state_inter_cache_1.bin'), inter_cache_1, 'int16');
    end
end

fprintf('\n=== Golden export complete ===\n');
fprintf('Output directory: %s\n', golden_dir);
fprintf('Frames exported: %d\n', total_frames_to_export);

%% Helper: export a matrix as binary int32/int16
function export_binary(filepath, data, dtype)
    data_flat = data(:);
    fid = fopen(filepath, 'wb');
    if strcmp(dtype, 'int16')
        fwrite(fid, data_flat, 'int16');
    elseif strcmp(dtype, 'uint16')
        fwrite(fid, data_flat, 'uint16');
    else  % int32
        fwrite(fid, data_flat, 'int32');
    end
    fclose(fid);
    fprintf('  Wrote %s (%d elements, %s)\n', filepath, numel(data_flat), dtype);
end
