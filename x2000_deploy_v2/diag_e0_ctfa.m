% diag_e0_ctfa.m — Export E0 cTFA intermediate outputs at each sub-step
% Run from x2000_deploy_v2/ directory (must have run diag_e0.m first for aff_out)

clear; clc;

script_dir = fileparts(mfilename('fullpath'));
parent_dir = fullfile(script_dir, '..');
addpath(fullfile(parent_dir, 'para_in_mat_FP'));
addpath(fullfile(parent_dir, 'test_wavs'));
addpath(parent_dir);

out_dir = fullfile(script_dir, 'diag_e0');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% Load test data (first frame) — same as diag_e0.m
[noisy_audio, fs] = audioread('noisy_fileid_1.wav');
N_fft = 512; win_len = 512; win_inc = 256;
hann_window = importdata('stft_window.mat');
[cmp_real, cmp_imag] = STFT_func(noisy_audio.', N_fft, win_len, win_inc, hann_window);

t = 1;
spec_real = Fix_point(cmp_real(t,:), 's32f20');
spec_imag = Fix_point(cmp_imag(t,:), 's32f20');
x = log_gen(spec_real, spec_imag);
erbfc_weight = importdata('erb_erb_fc_weight.mat');
x_bm = BM_module(x, erbfc_weight);

%% Get y_aff from E0 (same as diag_e0.m produces)
conv_cache = zeros(2,129);
x_c = [conv_cache; x_bm];

conv_weight = importdata('encoder_en_convs_0_ops_1_weight.mat');
conv_bias   = importdata('encoder_en_convs_0_ops_1_bias.mat');
y_conv = zeros(12,65);
Cin=1; Cout=12; Hout=1; Wout=65; ks=[3,3]; stride=[1,2]; Qr=-14;
for nOut = 1:Cout
    y_chan = zeros(Hout,Wout);
    for nIn = 1:Cin
        x_ch = x_c; x_pd = [zeros(3,1) x_ch zeros(3,1)];
        kc = squeeze(conv_weight(nOut,nIn,:,:));
        cr = zeros(Hout,Wout);
        for hi=1:Hout, for wi=1:Wout
            xk = x_pd((hi-1)*stride(1)+1:(hi-1)*stride(1)+ks(1),(wi-1)*stride(2)+1:(wi-1)*stride(2)+ks(2));
            cr(hi,wi) = sum(round(xk.*kc*2^(Qr)),'all');
        end; end
        y_chan = y_chan + cr;
    end
    y_chan = y_chan + conv_bias(nOut); y_conv(nOut,:) = y_chan;
end

bn_w=importdata('encoder_en_convs_0_ops_2_weight.mat');
bn_b=importdata('encoder_en_convs_0_ops_2_bias.mat');
bn_m=importdata('encoder_en_convs_0_ops_2_running_mean.mat');
bn_v=importdata('encoder_en_convs_0_ops_2_running_var.mat');
y_bn = round(round((y_conv-bn_m).*bn_v*2^(-14)).*bn_w*2^(-14)) + bn_b;

af_w=importdata('encoder_en_convs_0_ops_3_affine_weight.mat');
af_b=importdata('encoder_en_convs_0_ops_3_affine_bias.mat');
af_s=importdata('encoder_en_convs_0_ops_3_slope_weight.mat');
x_cp = y_bn; idx = y_bn<0; [rw,~]=find(idx);
y_bn(idx) = round(y_bn(idx).*af_s(rw)*2^(-13));
y_aff = round(x_cp.*af_w*2^(-13)) + af_b + y_bn;  % [12,65]

%% ===== cTFA TA (Time Attention) =====
fprintf('=== cTFA TA ===\n');

ta_ih_w = importdata('encoder_en_convs_0_ops_4_ta_gru_weight_ih_l0.mat');
ta_ih_b = importdata('encoder_en_convs_0_ops_4_ta_gru_bias_ih_l0.mat');
ta_hh_w = importdata('encoder_en_convs_0_ops_4_ta_gru_weight_hh_l0.mat');
ta_hh_b = importdata('encoder_en_convs_0_ops_4_ta_gru_bias_hh_l0.mat');
ta_fc_w = importdata('encoder_en_convs_0_ops_4_ta_fc_weight.mat');
ta_fc_b = importdata('encoder_en_convs_0_ops_4_ta_fc_bias.mat');

% Step TA1: per-channel energy
x_dq = y_aff * 2^(-20);
x_sq = x_dq.^2;
x_agg = mean(x_sq, 2);  % [12, 1]
x_t = x_agg.';           % [1, 12]
x_t = Fix_point(x_t, 'u32f20');  % uint32 Q20
export_mat(fullfile(out_dir, 'e0_ta_energy.txt'), x_t, 'uint32');
fprintf('TA1: energy [1x12], range=[%u,%u]\n', min(x_t(:)), max(x_t(:)));

% Step TA2: GRU
nHidden = 24;
h_cache = zeros(1,nHidden);
[x_gru, h_cache] = GRU_module(x_t, nHidden, h_cache, ta_ih_w, ta_ih_b, ta_hh_w, ta_hh_b, -21, -16);
export_mat(fullfile(out_dir, 'e0_ta_gru_out.txt'), x_gru, 'int16');
fprintf('TA2: GRU [1x24], range=[%d,%d]\n', min(x_gru(:)), max(x_gru(:)));

% Step TA3: FC
x_fc = round(x_gru * ta_fc_w * 2^(-8)) + ta_fc_b;  % [1, 12]
export_mat(fullfile(out_dir, 'e0_ta_fc_out.txt'), x_fc, 'int32');
fprintf('TA3: FC [1x12], range=[%d,%d]\n', min(x_fc(:)), max(x_fc(:)));

% Step TA4: sigmoid
x_fc_dq = x_fc * 2^(-20);
y_ta = Fix_point(sigmoid_func(x_fc_dq), 'u16f15');  % [1, 12] u16f15
export_mat(fullfile(out_dir, 'e0_ta_sig_out.txt'), y_ta, 'uint16');
fprintf('TA4: sigmoid [1x12], range=[%u,%u]\n', min(y_ta(:)), max(y_ta(:)));

%% ===== cTFA FA (Frequency Attention) =====
fprintf('\n=== cTFA FA ===\n');

fa_ih_w  = importdata('encoder_en_convs_0_ops_4_fa_gru_weight_ih_l0.mat');
fa_ih_b  = importdata('encoder_en_convs_0_ops_4_fa_gru_bias_ih_l0.mat');
fa_hh_w  = importdata('encoder_en_convs_0_ops_4_fa_gru_weight_hh_l0.mat');
fa_hh_b  = importdata('encoder_en_convs_0_ops_4_fa_gru_bias_hh_l0.mat');
fa_rih_w = importdata('encoder_en_convs_0_ops_4_fa_gru_weight_ih_l0_reverse.mat');
fa_rih_b = importdata('encoder_en_convs_0_ops_4_fa_gru_bias_ih_l0_reverse.mat');
fa_rhh_w = importdata('encoder_en_convs_0_ops_4_fa_gru_weight_hh_l0_reverse.mat');
fa_rhh_b = importdata('encoder_en_convs_0_ops_4_fa_gru_bias_hh_l0_reverse.mat');
fa_fc_w  = importdata('encoder_en_convs_0_ops_4_fa_fc_weight.mat');
fa_fc_b  = importdata('encoder_en_convs_0_ops_4_fa_fc_bias.mat');

% Step FA1: per-frequency energy (mean over channels)
x_dq2 = y_aff * 2^(-20);
x_sq2 = x_dq2.^2;
x_agg2 = mean(x_sq2, 1);  % [1, 65]
x_agg2 = Fix_point(x_agg2, 'u32f20');
export_mat(fullfile(out_dir, 'e0_fa_energy.txt'), x_agg2, 'uint32');
fprintf('FA1: energy [1x65], range=[%u,%u]\n', min(x_agg2(:)), max(x_agg2(:)));

% Step FA2: pad + reshape
pad_len = 3;
x_pad = [x_agg2 zeros(1,pad_len)];  % [1, 68]
x_rs = reshape(x_pad, [4, 17])';   % [17, 4]
export_mat(fullfile(out_dir, 'e0_fa_reshape.txt'), x_rs, 'uint32');
fprintf('FA2: reshape [17x4]\n');

% Step FA3: BiGRU
nHid = 4;
hc0 = zeros(1,nHid); xg0 = zeros(17,nHid);
for i=1:17, [xg0(i,:),hc0]=GRU_module(x_rs(i,:),nHid,hc0,fa_ih_w,fa_ih_b,fa_hh_w,fa_hh_b,-13,-8); end
x_rev = x_rs(end:-1:1,:);
hc1 = zeros(1,nHid); xg1 = zeros(17,nHid);
for i=1:17, [xg1(17-i+1,:),hc1]=GRU_module(x_rev(i,:),nHid,hc1,fa_rih_w,fa_rih_b,fa_rhh_w,fa_rhh_b,-13,-8); end
x_gru_fa = cat(2, xg0, xg1);  % [17, 8]
export_mat(fullfile(out_dir, 'e0_fa_gru_out.txt'), x_gru_fa, 'int16');
fprintf('FA3: BiGRU [17x8], range=[%d,%d]\n', min(x_gru_fa(:)), max(x_gru_fa(:)));

% Step FA4: FC
x_fc_fa = round(x_gru_fa * fa_fc_w * 2^(-9)) + fa_fc_b;  % [17, 4]
export_mat(fullfile(out_dir, 'e0_fa_fc_out.txt'), x_fc_fa, 'int32');
fprintf('FA4: FC [17x4], range=[%d,%d]\n', min(x_fc_fa(:)), max(x_fc_fa(:)));

% Step FA5: reshape back + depad
x_sh = reshape(x_fc_fa.', 1, []);  % [1, 68]
y_fa_prepad = x_sh(1:end-pad_len);  % [1, 65]
y_fa = Fix_point(sigmoid_func(y_fa_prepad*2^(-20)), 'u16f15');  % [1, 65] u16f15
export_mat(fullfile(out_dir, 'e0_fa_sig_out.txt'), y_fa, 'uint16');
fprintf('FA5: sigmoid depad [1x65], range=[%u,%u]\n', min(y_fa(:)), max(y_fa(:)));

%% ===== cTFA Fusion =====
fprintf('\n=== cTFA Fusion ===\n');

% y_t = round(y_tconv .* y_ta' * 2^(-15))
y_t = round(y_aff .* (y_ta.') * 2^(-15));
export_mat(fullfile(out_dir, 'e0_ctfa_yt.txt'), y_t, 'int32');
fprintf('Fusion TA: [12x65], range=[%d,%d]\n', min(y_t(:)), max(y_t(:)));

% y = round(y_t .* y_fa * 2^(-15))
y_ctfa = round(y_t .* y_fa * 2^(-15));
export_mat(fullfile(out_dir, 'e0_ctfa_out.txt'), y_ctfa, 'int32');
fprintf('Fusion FA: [12x65] FINAL, range=[%d,%d]\n', min(y_ctfa(:)), max(y_ctfa(:)));

%% Also export GRU split weights for debugging
% ih_r_weight = ta_ih_w(:,1:24)
ta_ih_r = ta_ih_w(:,1:24);
ta_ih_z = ta_ih_w(:,25:48);
ta_ih_n = ta_ih_w(:,49:72);
export_mat(fullfile(out_dir, 'e0_ta_ih_r.txt'), ta_ih_r, 'int16');
export_mat(fullfile(out_dir, 'e0_ta_ih_z.txt'), ta_ih_z, 'int16');
export_mat(fullfile(out_dir, 'e0_ta_ih_n.txt'), ta_ih_n, 'int16');

ta_hh_r = ta_hh_w(:,1:24);
ta_hh_z = ta_hh_w(:,25:48);
ta_hh_n = ta_hh_w(:,49:72);
export_mat(fullfile(out_dir, 'e0_ta_hh_r.txt'), ta_hh_r, 'int16');
export_mat(fullfile(out_dir, 'e0_ta_hh_z.txt'), ta_hh_z, 'int16');
export_mat(fullfile(out_dir, 'e0_ta_hh_n.txt'), ta_hh_n, 'int16');
fprintf('\nGRU weight splits exported\n');

fprintf('\n=== cTFA export complete ===\n');
fprintf('Output: %s\n', out_dir);

%% Helper
function export_mat(filepath, data, dtype)
    data_flat = data(:);
    fid = fopen(filepath, 'w');
    if strcmp(dtype, 'int16')
        for i = 1:numel(data_flat), fprintf(fid, '%d\n', int16(data_flat(i))); end
    elseif strcmp(dtype, 'uint16')
        for i = 1:numel(data_flat), fprintf(fid, '%u\n', uint16(data_flat(i))); end
    elseif strcmp(dtype, 'uint32')
        for i = 1:numel(data_flat), fprintf(fid, '%u\n', uint32(data_flat(i))); end
    else
        for i = 1:numel(data_flat), fprintf(fid, '%d\n', int32(data_flat(i))); end
    end
    fclose(fid);
end
