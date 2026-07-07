close all;clear;clc;

addpath 'para_in_mat_FP';
addpath 'test_wavs';

%% Load ERB
erbfc_weight = importdata('erb_erb_fc_weight.mat');
ierbfc_weight = importdata('erb_ierb_fc_weight.mat');

%% Load noisy audio
[noisy_audio, fs] = audioread('noisy_fileid_1.wav');
disp([max(noisy_audio), min(noisy_audio)])

y_mask_refer = squeeze( importdata('ys_fileid_1.mat') );
[audio, ~] = audioread('refer_fileid_1.wav');
refer_audio = audio.';

%% Temporal cache
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

%% STFT ----------------------------------------------------------------- %
N_fft = 512;
win_len = 512;
win_inc = 256;
hann_window = importdata('stft_window.mat');

[cmp_real,cmp_imag] = STFT_func(noisy_audio.', N_fft, win_len, win_inc, hann_window);

%%
T = size(cmp_real,1);

y_mask_infer = zeros(T,2,257);

for t = 1:T
    
    fprintf('Input temporal T=%d ',t);

    y_refer = ( squeeze(y_mask_refer(:,t,:)) ).';

    %% Log-magnitude Compression ---------------------------------------- %
    spec_real = Fix_point(cmp_real(t,:), 's32f20');
    spec_imag = Fix_point(cmp_imag(t,:), 's32f20');

    x = log_gen(spec_real, spec_imag);
    
    %% BM --------------------------------------------------------------- %    
    x_bm = BM_module(x, erbfc_weight);
    
    %% Encoder ---------------------------------------------------------- %
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
    
    %% G-DPRNN ---------------------------------------------------------- %
    [y_rnn1, inter_cache_0] = GDPRNN_module(y_e4, inter_cache_0, 0);
    [y_rnn2, inter_cache_1] = GDPRNN_module(y_rnn1, inter_cache_1, 1);
    
    %% Decoder ---------------------------------------------------------- %   
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

    %% Sigmod ----------------------------------------------------------- %
    y_dec_dq = y_dec*2^(-20);
    y_sig_dq = sigmoid_func(y_dec_dq);
    y_sig = Fix_point(y_sig_dq, 'u16f15');

    %% BS --------------------------------------------------------------- %
    y_bs = BS_module(y_sig, ierbfc_weight);

    %% MASK ------------------------------------------------------------- %   
    y_mask = MASK_module(y_bs, spec_real, spec_imag);

    y_mask_infer(t,:,:) = y_mask;

    mask_err = abs(y_mask - y_refer);
    max_err = max(mask_err(:));
    mean_err = mean(mask_err(:));
    fprintf(' MASK Error = [%.6f,%.6f]\n', max_err, mean_err);
    
end

I_spec = squeeze( y_mask_infer(:,1,:) );
Q_spec = squeeze( y_mask_infer(:,2,:) );
cmp_spec = I_spec + 1i*Q_spec;

% Reconstructed enhance waveform through ISTFT
enh_audio = ISTFT_func(cmp_spec, N_fft, win_len, win_inc, hann_window);
disp([max(enh_audio), min(enh_audio)])

audio_err = abs(enh_audio - refer_audio);
fprintf(' Enhanced Error = [%.6f,%.6f]\n', max(audio_err(:)), mean(audio_err(:)));

% sound(enh_audio,16000);
audiowrite('test_wavs/enh_fileid_1.wav', enh_audio, 16000);

