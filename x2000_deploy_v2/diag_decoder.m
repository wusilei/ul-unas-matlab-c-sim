% diag_decoder.m — Export Decoder D0-D4 intermediate outputs at each sub-step
% Run from x2000_deploy_v2/ directory
% Must have run export_all_layers.m first (for golden binaries)
% Also requires para_in_mat_FP/ in parent directory

clear; clc;

script_dir = fileparts(mfilename('fullpath'));
parent_dir = fullfile(script_dir, '..');
addpath(fullfile(parent_dir, 'para_in_mat_FP'));
addpath(fullfile(parent_dir, 'test_wavs'));
addpath(parent_dir);

out_dir = fullfile(script_dir, 'diag_decoder');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% Load test data (first frame) — same as export_all_layers.m
[noisy_audio, fs] = audioread('noisy_fileid_1.wav');
N_fft = 512; win_len = 512; win_inc = 256;
hann_window = importdata('stft_window.mat');
[cmp_real, cmp_imag] = STFT_func(noisy_audio.', N_fft, win_len, win_inc, hann_window);

t = 1;
spec_real = Fix_point(cmp_real(t,:), 's32f20');
spec_imag = Fix_point(cmp_imag(t,:), 's32f20');

%% Run full pipeline up to Decoder input (identical to export_all_layers.m)
x = log_gen(spec_real, spec_imag);
erbfc_weight = importdata('erb_erb_fc_weight.mat');
x_bm = BM_module(x, erbfc_weight);

% Encoder
conv_cache_e0 = zeros(2,129); tfa_cache_e0 = zeros(1,24);
conv_cache_e1 = zeros(24,65); tfa_cache_e1 = zeros(1,48);
conv_cache_e2 = zeros(24,33); tfa_cache_e2 = zeros(1,48);
tfa_cache_e3 = zeros(1,64);
tfa_cache_e4 = zeros(1,32);

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

% GDPRNN
inter_cache_0 = zeros(33,16);
inter_cache_1 = zeros(33,16);
[y_rnn1, inter_cache_0] = GDPRNN_module(y_e4, inter_cache_0, 0);
[y_rnn2, inter_cache_1] = GDPRNN_module(y_rnn1, inter_cache_1, 1);

% Initialize Decoder caches
tfa_cache_d0 = zeros(1,64);
tfa_cache_d1 = zeros(1,48);
conv_cache_d0 = zeros(24,33); tfa_cache_d2 = zeros(1,48);
conv_cache_d1 = zeros(12,33); tfa_cache_d3 = zeros(1,24);
conv_cache_d2 = zeros(12,2,65); tfa_cache_d4 = zeros(1,2);

fprintf('=== Decoder Diagnostic Export ===\n');
fprintf('Output: %s\n\n', out_dir);

%% ========================================================================
%% D0: De_XDWS0  [16,33]+skip_e4 → [32,33]
%% ========================================================================
fprintf('--- D0 De_XDWS0 ---\n');

% Step D0.0: skip_add
x_d0 = y_rnn2 + y_e4;
export_txt(fullfile(out_dir, 'd0_skip_add.txt'), x_d0, 'int32');
fprintf('D0.0 skip_add [16x33], range=[%d,%d]\n', min(x_d0(:)), max(x_d0(:)));

% Step D0.1: PConv_g2_aff
pw_d0 = importdata('decoder_de_convs_0_pconv_0_weight.mat');
pb_d0 = importdata('decoder_de_convs_0_pconv_0_bias.mat');
bw_d0 = importdata('decoder_de_convs_0_pconv_1_weight.mat');
bb_d0 = importdata('decoder_de_convs_0_pconv_1_bias.mat');
bm_d0 = importdata('decoder_de_convs_0_pconv_1_running_mean.mat');
bv_d0 = importdata('decoder_de_convs_0_pconv_1_running_var.mat');
aw_d0 = importdata('decoder_de_convs_0_pconv_2_affine_weight.mat');
ab_d0 = importdata('decoder_de_convs_0_pconv_2_affine_bias.mat');
as_d0 = importdata('decoder_de_convs_0_pconv_2_slope_weight.mat');

% Group 0: channels 1-8, weight rows 1-16
yp0_d0 = pconv2d_func(x_d0(1:8,:), 8, 16, 1, 33, pw_d0(1:16,:), pb_d0(1:16), -14);
% Group 1: channels 9-16, weight rows 17-32
yp1_d0 = pconv2d_func(x_d0(9:16,:), 8, 16, 1, 33, pw_d0(17:32,:), pb_d0(17:32), -14);
yp_d0 = cat(1, yp0_d0, yp1_d0);
yb_d0 = bn_func(yp_d0, bw_d0, bb_d0, bm_d0, bv_d0, -14, -14);
ya_d0 = affineprelu_func(yb_d0, aw_d0, ab_d0, as_d0, -13, -13);
export_txt(fullfile(out_dir, 'd0_pconv_out.txt'), ya_d0, 'int32');
fprintf('D0.1 pconv_aff [32x33], range=[%d,%d]\n', min(ya_d0(:)), max(ya_d0(:)));

% Step D0.2: Shuffle
ys_d0 = zeros(32,33);
ys_d0(1:2:end,:) = ya_d0(1:16,:);
ys_d0(2:2:end,:) = ya_d0(17:32,:);
export_txt(fullfile(out_dir, 'd0_shuffle_out.txt'), ys_d0, 'int32');
fprintf('D0.2 shuffle [32x33], range=[%d,%d]\n', min(ys_d0(:)), max(ys_d0(:)));

% Step D0.3: nonGTConv_aff
ncw_d0 = importdata('decoder_de_convs_0_dconv_1_weight.mat');
ncb_d0 = importdata('decoder_de_convs_0_dconv_1_bias.mat');
nbw_d0 = importdata('decoder_de_convs_0_dconv_2_weight.mat');
nbb_d0 = importdata('decoder_de_convs_0_dconv_2_bias.mat');
nbm_d0 = importdata('decoder_de_convs_0_dconv_2_running_mean.mat');
nbv_d0 = importdata('decoder_de_convs_0_dconv_2_running_var.mat');
naw_d0 = importdata('decoder_de_convs_0_dconv_3_affine_weight.mat');
nab_d0 = importdata('decoder_de_convs_0_dconv_3_affine_bias.mat');
nas_d0 = importdata('decoder_de_convs_0_dconv_3_slope_weight.mat');

ynt_d0 = non_gtconv2d_func(ys_d0, 32, 1, 33, [1,5], [1,1], ncw_d0, ncb_d0, -14);
ynb_d0 = bn_func(ynt_d0, nbw_d0, nbb_d0, nbm_d0, nbv_d0, -14, -14);
yna_d0 = affineprelu_func(ynb_d0, naw_d0, nab_d0, nas_d0, -13, -13);
export_txt(fullfile(out_dir, 'd0_ngtconv_out.txt'), yna_d0, 'int32');
fprintf('D0.3 ngtconv_aff [32x33], range=[%d,%d]\n', min(yna_d0(:)), max(yna_d0(:)));

% Step D0.4: cTFA
ty_d0 = export_ctfa_ta(fullfile(out_dir, 'd0'), yna_d0, 32, 33, tfa_cache_d0, 64, 32, ...
    importdata('decoder_de_convs_0_dconv_4_ta_gru_weight_ih_l0.mat'), ...
    importdata('decoder_de_convs_0_dconv_4_ta_gru_bias_ih_l0.mat'), ...
    importdata('decoder_de_convs_0_dconv_4_ta_gru_weight_hh_l0.mat'), ...
    importdata('decoder_de_convs_0_dconv_4_ta_gru_bias_hh_l0.mat'), ...
    importdata('decoder_de_convs_0_dconv_4_ta_fc_weight.mat'), ...
    importdata('decoder_de_convs_0_dconv_4_ta_fc_bias.mat'));
fy_d0 = export_ctfa_fa(fullfile(out_dir, 'd0'), yna_d0, 32, 33, 3, 4, ...
    importdata('decoder_de_convs_0_dconv_4_fa_gru_weight_ih_l0.mat'), ...
    importdata('decoder_de_convs_0_dconv_4_fa_gru_bias_ih_l0.mat'), ...
    importdata('decoder_de_convs_0_dconv_4_fa_gru_weight_hh_l0.mat'), ...
    importdata('decoder_de_convs_0_dconv_4_fa_gru_bias_hh_l0.mat'), ...
    importdata('decoder_de_convs_0_dconv_4_fa_gru_weight_ih_l0_reverse.mat'), ...
    importdata('decoder_de_convs_0_dconv_4_fa_gru_bias_ih_l0_reverse.mat'), ...
    importdata('decoder_de_convs_0_dconv_4_fa_gru_weight_hh_l0_reverse.mat'), ...
    importdata('decoder_de_convs_0_dconv_4_fa_gru_bias_hh_l0_reverse.mat'), ...
    importdata('decoder_de_convs_0_dconv_4_fa_fc_weight.mat'), ...
    importdata('decoder_de_convs_0_dconv_4_fa_fc_bias.mat'));
y_d0 = export_ctfa_fusion(fullfile(out_dir, 'd0'), yna_d0, ty_d0, fy_d0);
fprintf('D0.4 ctfa [32x33] FINAL, range=[%d,%d]\n', min(y_d0(:)), max(y_d0(:)));

%% ========================================================================
%% D1: De_XMB0  [32,33]+skip_e3 → [24,33]
%% ========================================================================
fprintf('\n--- D1 De_XMB0 ---\n');

% Step D1.0: skip_add
x_d1 = y_d0 + y_e3;
export_txt(fullfile(out_dir, 'd1_skip_add.txt'), x_d1, 'int32');
fprintf('D1.0 skip_add [32x33], range=[%d,%d]\n', min(x_d1(:)), max(x_d1(:)));

% Step D1.1: PConv_g2_aff (PConv_block_0)
pw0_d1 = importdata('decoder_de_convs_1_pconv1_0_weight.mat');
pb0_d1 = importdata('decoder_de_convs_1_pconv1_0_bias.mat');
bw0_d1 = importdata('decoder_de_convs_1_pconv1_1_weight.mat');
bb0_d1 = importdata('decoder_de_convs_1_pconv1_1_bias.mat');
bm0_d1 = importdata('decoder_de_convs_1_pconv1_1_running_mean.mat');
bv0_d1 = importdata('decoder_de_convs_1_pconv1_1_running_var.mat');
aw0_d1 = importdata('decoder_de_convs_1_pconv1_2_affine_weight.mat');
ab0_d1 = importdata('decoder_de_convs_1_pconv1_2_affine_bias.mat');
as0_d1 = importdata('decoder_de_convs_1_pconv1_2_slope_weight.mat');

yp0_d1 = pconv2d_func(x_d1(1:16,:), 16, 12, 1, 33, pw0_d1(1:12,:), pb0_d1(1:12), -13);
yp1_d1 = pconv2d_func(x_d1(17:32,:), 16, 12, 1, 33, pw0_d1(13:24,:), pb0_d1(13:24), -13);
ypc_d1 = cat(1, yp0_d1, yp1_d1);
ypb_d1 = bn_func(ypc_d1, bw0_d1, bb0_d1, bm0_d1, bv0_d1, -11, -14);
ypa_d1 = affineprelu_func(ypb_d1, aw0_d1, ab0_d1, as0_d1, -13, -13);
export_txt(fullfile(out_dir, 'd1_pconv0_out.txt'), ypa_d1, 'int32');
fprintf('D1.1 pconv0_aff [24x33], range=[%d,%d]\n', min(ypa_d1(:)), max(ypa_d1(:)));

% Step D1.2: Shuffle
ys_d1 = zeros(24,33);
ys_d1(1:2:end,:) = ypa_d1(1:12,:);
ys_d1(2:2:end,:) = ypa_d1(13:24,:);
export_txt(fullfile(out_dir, 'd1_shuffle_out.txt'), ys_d1, 'int32');
fprintf('D1.2 shuffle [24x33], range=[%d,%d]\n', min(ys_d1(:)), max(ys_d1(:)));

% Step D1.3: nonGTConv_aff (nonTConv_block)
ncw_d1 = importdata('decoder_de_convs_1_dconv_1_weight.mat');
ncb_d1 = importdata('decoder_de_convs_1_dconv_1_bias.mat');
nbw_d1 = importdata('decoder_de_convs_1_dconv_2_weight.mat');
nbb_d1 = importdata('decoder_de_convs_1_dconv_2_bias.mat');
nbm_d1 = importdata('decoder_de_convs_1_dconv_2_running_mean.mat');
nbv_d1 = importdata('decoder_de_convs_1_dconv_2_running_var.mat');
naw_d1 = importdata('decoder_de_convs_1_dconv_3_affine_weight.mat');
nab_d1 = importdata('decoder_de_convs_1_dconv_3_affine_bias.mat');
nas_d1 = importdata('decoder_de_convs_1_dconv_3_slope_weight.mat');

ynt_d1 = non_gtconv2d_func(ys_d1, 24, 1, 33, [1,5], [1,1], ncw_d1, ncb_d1, -14);
ynb_d1 = bn_func(ynt_d1, nbw_d1, nbb_d1, nbm_d1, nbv_d1, -11, -14);
yna_d1 = affineprelu_func(ynb_d1, naw_d1, nab_d1, nas_d1, -13, -13);
export_txt(fullfile(out_dir, 'd1_ngtconv_out.txt'), yna_d1, 'int32');
fprintf('D1.3 ngtconv_aff [24x33], range=[%d,%d]\n', min(yna_d1(:)), max(yna_d1(:)));

% Step D1.4: PConv_g2_bn (PConv_block_1)
pw1_d1 = importdata('decoder_de_convs_1_pconv2_0_weight.mat');
pb1_d1 = importdata('decoder_de_convs_1_pconv2_0_bias.mat');
bw1_d1 = importdata('decoder_de_convs_1_pconv2_1_weight.mat');
bb1_d1 = importdata('decoder_de_convs_1_pconv2_1_bias.mat');
bm1_d1 = importdata('decoder_de_convs_1_pconv2_1_running_mean.mat');
bv1_d1 = importdata('decoder_de_convs_1_pconv2_1_running_var.mat');

ypb0 = pconv2d_func(yna_d1(1:12,:), 12, 12, 1, 33, pw1_d1(1:12,:), pb1_d1(1:12), -14);
ypb1 = pconv2d_func(yna_d1(13:24,:), 12, 12, 1, 33, pw1_d1(13:24,:), pb1_d1(13:24), -14);
ypb2_d1 = cat(1, ypb0, ypb1);
ypbn_d1 = bn_func(ypb2_d1, bw1_d1, bb1_d1, bm1_d1, bv1_d1, -11, -11);
export_txt(fullfile(out_dir, 'd1_pconv1_out.txt'), ypbn_d1, 'int32');
fprintf('D1.4 pconv1_bn [24x33], range=[%d,%d]\n', min(ypbn_d1(:)), max(ypbn_d1(:)));

% Step D1.5: cTFA
ty_d1 = export_ctfa_ta(fullfile(out_dir, 'd1'), ypbn_d1, 24, 33, tfa_cache_d1, 48, 24, ...
    importdata('decoder_de_convs_1_pconv2_2_ta_gru_weight_ih_l0.mat'), ...
    importdata('decoder_de_convs_1_pconv2_2_ta_gru_bias_ih_l0.mat'), ...
    importdata('decoder_de_convs_1_pconv2_2_ta_gru_weight_hh_l0.mat'), ...
    importdata('decoder_de_convs_1_pconv2_2_ta_gru_bias_hh_l0.mat'), ...
    importdata('decoder_de_convs_1_pconv2_2_ta_fc_weight.mat'), ...
    importdata('decoder_de_convs_1_pconv2_2_ta_fc_bias.mat'));
fy_d1 = export_ctfa_fa(fullfile(out_dir, 'd1'), ypbn_d1, 24, 33, 3, 4, ...
    importdata('decoder_de_convs_1_pconv2_2_fa_gru_weight_ih_l0.mat'), ...
    importdata('decoder_de_convs_1_pconv2_2_fa_gru_bias_ih_l0.mat'), ...
    importdata('decoder_de_convs_1_pconv2_2_fa_gru_weight_hh_l0.mat'), ...
    importdata('decoder_de_convs_1_pconv2_2_fa_gru_bias_hh_l0.mat'), ...
    importdata('decoder_de_convs_1_pconv2_2_fa_gru_weight_ih_l0_reverse.mat'), ...
    importdata('decoder_de_convs_1_pconv2_2_fa_gru_bias_ih_l0_reverse.mat'), ...
    importdata('decoder_de_convs_1_pconv2_2_fa_gru_weight_hh_l0_reverse.mat'), ...
    importdata('decoder_de_convs_1_pconv2_2_fa_gru_bias_hh_l0_reverse.mat'), ...
    importdata('decoder_de_convs_1_pconv2_2_fa_fc_weight.mat'), ...
    importdata('decoder_de_convs_1_pconv2_2_fa_fc_bias.mat'));
yc_d1 = export_ctfa_fusion(fullfile(out_dir, 'd1'), ypbn_d1, ty_d1, fy_d1);

% Step D1.6: Shuffle (final)
y_d1 = zeros(24,33);
y_d1(1:2:end,:) = yc_d1(1:12,:);
y_d1(2:2:end,:) = yc_d1(13:24,:);
export_txt(fullfile(out_dir, 'd1_final.txt'), y_d1, 'int32');
fprintf('D1.6 shuffle FINAL [24x33], range=[%d,%d]\n', min(y_d1(:)), max(y_d1(:)));

%% ========================================================================
%% D2: De_XDWS1  [24,33]+skip_e2 → [24,33]
%% ========================================================================
fprintf('\n--- D2 De_XDWS1 ---\n');

% Step D2.0: skip_add
x_d2 = y_d1 + y_e2;
export_txt(fullfile(out_dir, 'd2_skip_add.txt'), x_d2, 'int32');
fprintf('D2.0 skip_add [24x33], range=[%d,%d]\n', min(x_d2(:)), max(x_d2(:)));

% Step D2.1: PConv_g2_aff
pw_d2 = importdata('decoder_de_convs_2_pconv_0_weight.mat');
pb_d2 = importdata('decoder_de_convs_2_pconv_0_bias.mat');
bw_d2 = importdata('decoder_de_convs_2_pconv_1_weight.mat');
bb_d2 = importdata('decoder_de_convs_2_pconv_1_bias.mat');
bm_d2 = importdata('decoder_de_convs_2_pconv_1_running_mean.mat');
bv_d2 = importdata('decoder_de_convs_2_pconv_1_running_var.mat');
aw_d2 = importdata('decoder_de_convs_2_pconv_2_affine_weight.mat');
ab_d2 = importdata('decoder_de_convs_2_pconv_2_affine_bias.mat');
as_d2 = importdata('decoder_de_convs_2_pconv_2_slope_weight.mat');

yp0_d2 = pconv2d_func(x_d2(1:12,:), 12, 12, 1, 33, pw_d2(1:12,:), pb_d2(1:12), -14);
yp1_d2 = pconv2d_func(x_d2(13:24,:), 12, 12, 1, 33, pw_d2(13:24,:), pb_d2(13:24), -14);
ypc_d2 = cat(1, yp0_d2, yp1_d2);
ypb_d2 = bn_func(ypc_d2, bw_d2, bb_d2, bm_d2, bv_d2, -11, -14);
ypa_d2 = affineprelu_func(ypb_d2, aw_d2, ab_d2, as_d2, -13, -13);
export_txt(fullfile(out_dir, 'd2_pconv_out.txt'), ypa_d2, 'int32');
fprintf('D2.1 pconv_aff [24x33], range=[%d,%d]\n', min(ypa_d2(:)), max(ypa_d2(:)));

% Step D2.2: Shuffle
ys_d2 = zeros(24,33);
ys_d2(1:2:end,:) = ypa_d2(1:12,:);
ys_d2(2:2:end,:) = ypa_d2(13:24,:);
export_txt(fullfile(out_dir, 'd2_shuffle_out.txt'), ys_d2, 'int32');
fprintf('D2.2 shuffle [24x33], range=[%d,%d]\n', min(ys_d2(:)), max(ys_d2(:)));

% Step D2.3: GTConv_aff (with cache)
gcw_d2 = importdata('decoder_de_convs_2_dconv_1_weight.mat');
gcb_d2 = importdata('decoder_de_convs_2_dconv_1_bias.mat');
gbw_d2 = importdata('decoder_de_convs_2_dconv_2_weight.mat');
gbb_d2 = importdata('decoder_de_convs_2_dconv_2_bias.mat');
gbm_d2 = importdata('decoder_de_convs_2_dconv_2_running_mean.mat');
gbv_d2 = importdata('decoder_de_convs_2_dconv_2_running_var.mat');
gaw_d2 = importdata('decoder_de_convs_2_dconv_3_affine_weight.mat');
gab_d2 = importdata('decoder_de_convs_2_dconv_3_affine_bias.mat');
gas_d2 = importdata('decoder_de_convs_2_dconv_3_slope_weight.mat');

% Manual GTConv: [cache; current] → [2,33], zero-insert stride=1, rot180 kernel
% gtconv: stride_w=1 → no zero insertion needed (stride=1 → W_ins=W)
W = 33; stride_w = 1; W_ins = (W-1)*stride_w + 1;  % = 33
ygt_d2 = zeros(24,33);
for nOut = 1:24
    x_ch = [conv_cache_d0(nOut,:); ys_d2(nOut,:)];  % [2,33]
    % zero insertion (stride=1 = identity)
    x_ins = x_ch;  % [2,33]
    % rot90(kernel,2) = 180 deg rotation
    kc = squeeze(gcw_d2(nOut,1,:,:));  % [2,3]
    krot = rot90(kc, 2);  % 180 deg rotation
    % build padded input for valid conv
    x_pad = [zeros(2,1) x_ins zeros(2,1)];  % [2, 35]
    cr = zeros(1,33);
    for wi = 1:33
        xk = x_pad(:, wi:wi+2);
        cr(wi) = sum(round(xk .* krot * 2^(-13)), 'all');
    end
    ygt_d2(nOut,:) = cr + gcb_d2(nOut);
end
conv_cache_d0_out = ys_d2;  % update cache
ygb_d2 = bn_func(ygt_d2, gbw_d2, gbb_d2, gbm_d2, gbv_d2, -11, -14);
yga_d2 = affineprelu_func(ygb_d2, gaw_d2, gab_d2, gas_d2, -13, -13);
export_txt(fullfile(out_dir, 'd2_gtconv_out.txt'), yga_d2, 'int32');
fprintf('D2.3 gtconv_aff [24x33], range=[%d,%d]\n', min(yga_d2(:)), max(yga_d2(:)));

% Step D2.4: cTFA
ty_d2 = export_ctfa_ta(fullfile(out_dir, 'd2'), yga_d2, 24, 33, tfa_cache_d2, 48, 24, ...
    importdata('decoder_de_convs_2_dconv_4_ta_gru_weight_ih_l0.mat'), ...
    importdata('decoder_de_convs_2_dconv_4_ta_gru_bias_ih_l0.mat'), ...
    importdata('decoder_de_convs_2_dconv_4_ta_gru_weight_hh_l0.mat'), ...
    importdata('decoder_de_convs_2_dconv_4_ta_gru_bias_hh_l0.mat'), ...
    importdata('decoder_de_convs_2_dconv_4_ta_fc_weight.mat'), ...
    importdata('decoder_de_convs_2_dconv_4_ta_fc_bias.mat'));
fy_d2 = export_ctfa_fa(fullfile(out_dir, 'd2'), yga_d2, 24, 33, 3, 4, ...
    importdata('decoder_de_convs_2_dconv_4_fa_gru_weight_ih_l0.mat'), ...
    importdata('decoder_de_convs_2_dconv_4_fa_gru_bias_ih_l0.mat'), ...
    importdata('decoder_de_convs_2_dconv_4_fa_gru_weight_hh_l0.mat'), ...
    importdata('decoder_de_convs_2_dconv_4_fa_gru_bias_hh_l0.mat'), ...
    importdata('decoder_de_convs_2_dconv_4_fa_gru_weight_ih_l0_reverse.mat'), ...
    importdata('decoder_de_convs_2_dconv_4_fa_gru_bias_ih_l0_reverse.mat'), ...
    importdata('decoder_de_convs_2_dconv_4_fa_gru_weight_hh_l0_reverse.mat'), ...
    importdata('decoder_de_convs_2_dconv_4_fa_gru_bias_hh_l0_reverse.mat'), ...
    importdata('decoder_de_convs_2_dconv_4_fa_fc_weight.mat'), ...
    importdata('decoder_de_convs_2_dconv_4_fa_fc_bias.mat'));
y_d2 = export_ctfa_fusion(fullfile(out_dir, 'd2'), yga_d2, ty_d2, fy_d2);
fprintf('D2.4 ctfa [24x33] FINAL, range=[%d,%d]\n', min(y_d2(:)), max(y_d2(:)));

%% ========================================================================
%% D3: De_XMB1  [24,33]+skip_e1 → [12,65]
%% ========================================================================
fprintf('\n--- D3 De_XMB1 ---\n');

% Step D3.0: skip_add
x_d3 = y_d2 + y_e1;
export_txt(fullfile(out_dir, 'd3_skip_add.txt'), x_d3, 'int32');
fprintf('D3.0 skip_add [24x33], range=[%d,%d]\n', min(x_d3(:)), max(x_d3(:)));

% Step D3.1: PConv_g2_aff (PConv_block_0)
pw0_d3 = importdata('decoder_de_convs_3_pconv1_0_weight.mat');
pb0_d3 = importdata('decoder_de_convs_3_pconv1_0_bias.mat');
bw0_d3 = importdata('decoder_de_convs_3_pconv1_1_weight.mat');
bb0_d3 = importdata('decoder_de_convs_3_pconv1_1_bias.mat');
bm0_d3 = importdata('decoder_de_convs_3_pconv1_1_running_mean.mat');
bv0_d3 = importdata('decoder_de_convs_3_pconv1_1_running_var.mat');
aw0_d3 = importdata('decoder_de_convs_3_pconv1_2_affine_weight.mat');
ab0_d3 = importdata('decoder_de_convs_3_pconv1_2_affine_bias.mat');
as0_d3 = importdata('decoder_de_convs_3_pconv1_2_slope_weight.mat');

yp0_d3 = pconv2d_func(x_d3(1:12,:), 12, 6, 1, 33, pw0_d3(1:6,:), pb0_d3(1:6), -14);
yp1_d3 = pconv2d_func(x_d3(13:24,:), 12, 6, 1, 33, pw0_d3(7:12,:), pb0_d3(7:12), -14);
ypc_d3 = cat(1, yp0_d3, yp1_d3);
ypb_d3 = bn_func(ypc_d3, bw0_d3, bb0_d3, bm0_d3, bv0_d3, -11, -14);
ypa_d3 = affineprelu_func(ypb_d3, aw0_d3, ab0_d3, as0_d3, -13, -13);
export_txt(fullfile(out_dir, 'd3_pconv0_out.txt'), ypa_d3, 'int32');
fprintf('D3.1 pconv0_aff [12x33], range=[%d,%d]\n', min(ypa_d3(:)), max(ypa_d3(:)));

% Step D3.2: Shuffle
ys_d3 = zeros(12,33);
ys_d3(1:2:end,:) = ypa_d3(1:6,:);
ys_d3(2:2:end,:) = ypa_d3(7:12,:);
export_txt(fullfile(out_dir, 'd3_shuffle_out.txt'), ys_d3, 'int32');
fprintf('D3.2 shuffle [12x33], range=[%d,%d]\n', min(ys_d3(:)), max(ys_d3(:)));

% Step D3.3: GTConv_aff (TConv_block) stride_w=2: 33→65
gcw_d3 = importdata('decoder_de_convs_3_dconv_1_weight.mat');
gcb_d3 = importdata('decoder_de_convs_3_dconv_1_bias.mat');
gbw_d3 = importdata('decoder_de_convs_3_dconv_2_weight.mat');
gbb_d3 = importdata('decoder_de_convs_3_dconv_2_bias.mat');
gbm_d3 = importdata('decoder_de_convs_3_dconv_2_running_mean.mat');
gbv_d3 = importdata('decoder_de_convs_3_dconv_2_running_var.mat');
gaw_d3 = importdata('decoder_de_convs_3_dconv_3_affine_weight.mat');
gab_d3 = importdata('decoder_de_convs_3_dconv_3_affine_bias.mat');
gas_d3 = importdata('decoder_de_convs_3_dconv_3_slope_weight.mat');

% GTConv stride_w=2: W=33, W_ins=(33-1)*2+1=65
W3 = 33; stride_w3 = 2; W_ins3 = (W3-1)*stride_w3 + 1;  % = 65
ygt_d3 = zeros(12,65);
for nOut = 1:12
    x_ch = [conv_cache_d1(nOut,:); ys_d3(nOut,:)];  % [2,33]
    % zero insertion stride_w=2
    x_ins = zeros(2, W_ins3);
    x_ins(:, 1:stride_w3:end) = x_ch;  % [2,65]
    % rot90(kernel,2) = 180 deg rotation
    kc = squeeze(gcw_d3(nOut,1,:,:));  % [2,3]
    krot = rot90(kc, 2);
    % padded input
    x_pad = [zeros(2,1) x_ins zeros(2,1)];  % [2,67]
    cr = zeros(1,65);
    for wi = 1:65
        xk = x_pad(:, wi:wi+2);
        cr(wi) = sum(round(xk .* krot * 2^(-14)), 'all');
    end
    ygt_d3(nOut,:) = cr + gcb_d3(nOut);
end
conv_cache_d1_out = ys_d3;
ygb_d3 = bn_func(ygt_d3, gbw_d3, gbb_d3, gbm_d3, gbv_d3, -11, -11);
yga_d3 = affineprelu_func(ygb_d3, gaw_d3, gab_d3, gas_d3, -13, -13);
export_txt(fullfile(out_dir, 'd3_gtconv_out.txt'), yga_d3, 'int32');
fprintf('D3.3 gtconv_aff [12x65], range=[%d,%d]\n', min(yga_d3(:)), max(yga_d3(:)));

% Step D3.4: PConv_g2_bn (PConv_block_1)
pw1_d3 = importdata('decoder_de_convs_3_pconv2_0_weight.mat');
pb1_d3 = importdata('decoder_de_convs_3_pconv2_0_bias.mat');
bw1_d3 = importdata('decoder_de_convs_3_pconv2_1_weight.mat');
bb1_d3 = importdata('decoder_de_convs_3_pconv2_1_bias.mat');
bm1_d3 = importdata('decoder_de_convs_3_pconv2_1_running_mean.mat');
bv1_d3 = importdata('decoder_de_convs_3_pconv2_1_running_var.mat');

ypb0_d3 = pconv2d_func(yga_d3(1:6,:), 6, 6, 1, 65, pw1_d3(1:6,:), pb1_d3(1:6), -14);
ypb1_d3 = pconv2d_func(yga_d3(7:12,:), 6, 6, 1, 65, pw1_d3(7:12,:), pb1_d3(7:12), -14);
ypbc_d3 = cat(1, ypb0_d3, ypb1_d3);
ypbn_d3 = bn_func(ypbc_d3, bw1_d3, bb1_d3, bm1_d3, bv1_d3, -11, -11);
export_txt(fullfile(out_dir, 'd3_pconv1_out.txt'), ypbn_d3, 'int32');
fprintf('D3.4 pconv1_bn [12x65], range=[%d,%d]\n', min(ypbn_d3(:)), max(ypbn_d3(:)));

% Step D3.5: cTFA
ty_d3 = export_ctfa_ta(fullfile(out_dir, 'd3'), ypbn_d3, 12, 65, tfa_cache_d3, 24, 12, ...
    importdata('decoder_de_convs_3_pconv2_2_ta_gru_weight_ih_l0.mat'), ...
    importdata('decoder_de_convs_3_pconv2_2_ta_gru_bias_ih_l0.mat'), ...
    importdata('decoder_de_convs_3_pconv2_2_ta_gru_weight_hh_l0.mat'), ...
    importdata('decoder_de_convs_3_pconv2_2_ta_gru_bias_hh_l0.mat'), ...
    importdata('decoder_de_convs_3_pconv2_2_ta_fc_weight.mat'), ...
    importdata('decoder_de_convs_3_pconv2_2_ta_fc_bias.mat'));
fy_d3 = export_ctfa_fa(fullfile(out_dir, 'd3'), ypbn_d3, 12, 65, 3, 4, ...
    importdata('decoder_de_convs_3_pconv2_2_fa_gru_weight_ih_l0.mat'), ...
    importdata('decoder_de_convs_3_pconv2_2_fa_gru_bias_ih_l0.mat'), ...
    importdata('decoder_de_convs_3_pconv2_2_fa_gru_weight_hh_l0.mat'), ...
    importdata('decoder_de_convs_3_pconv2_2_fa_gru_bias_hh_l0.mat'), ...
    importdata('decoder_de_convs_3_pconv2_2_fa_gru_weight_ih_l0_reverse.mat'), ...
    importdata('decoder_de_convs_3_pconv2_2_fa_gru_bias_ih_l0_reverse.mat'), ...
    importdata('decoder_de_convs_3_pconv2_2_fa_gru_weight_hh_l0_reverse.mat'), ...
    importdata('decoder_de_convs_3_pconv2_2_fa_gru_bias_hh_l0_reverse.mat'), ...
    importdata('decoder_de_convs_3_pconv2_2_fa_fc_weight.mat'), ...
    importdata('decoder_de_convs_3_pconv2_2_fa_fc_bias.mat'));
yc_d3 = export_ctfa_fusion(fullfile(out_dir, 'd3'), ypbn_d3, ty_d3, fy_d3);

% Step D3.6: Shuffle (final)
y_d3 = zeros(12,65);
y_d3(1:2:end,:) = yc_d3(1:6,:);
y_d3(2:2:end,:) = yc_d3(7:12,:);
export_txt(fullfile(out_dir, 'd3_final.txt'), y_d3, 'int32');
fprintf('D3.6 shuffle FINAL [12x65], range=[%d,%d]\n', min(y_d3(:)), max(y_d3(:)));

%% ========================================================================
%% D4: De_XConv  [12,65]+skip_e0 → [1,129]
%% ========================================================================
fprintf('\n--- D4 De_XConv ---\n');

% Step D4.0: skip_add
x_d4 = y_d3 + y_e0;
export_txt(fullfile(out_dir, 'd4_skip_add.txt'), x_d4, 'int32');
fprintf('D4.0 skip_add [12x65], range=[%d,%d]\n', min(x_d4(:)), max(x_d4(:)));

% Step D4.1: Build 3D cache xc = [conv_cache_d2(:,:); reshape(x_d4,[12,1,65])]
% conv_cache_d2 is [12,2,65]
xc_d4 = zeros(12,3,65);
for c = 1:12
    xc_d4(c,1:2,:) = conv_cache_d2(c,:,:);  % 2 frames from cache
    xc_d4(c,3,:) = x_d4(c,:);                % current frame
end
export_txt(fullfile(out_dir, 'd4_xc.txt'), xc_d4, 'int32');
fprintf('D4.1 xc [12x3x65]\n');

% Step D4.2: TConv [12,3,65] → [1,129], stride_h=1, stride_w=2
tcw_d4 = importdata('decoder_de_convs_4_ops_1_weight.mat');
tcb_d4 = importdata('decoder_de_convs_4_ops_1_bias.mat');

% TConv: zero-insertion + rot180 kernel
% x is [Cin=12, H=3, W=65], stride_w=2, kernel [3,3]
W4_in = 65; stride_w4 = 2; W4_ins = (W4_in-1)*stride_w4 + 1;  % = 129
H4_in = 3; stride_h4 = 1; H4_ins = (H4_in-1)*stride_h4 + 1;  % = 3

yt_d4 = zeros(1,129);  % Cout=1
for nOut = 1:1  % Cout=1
    y_chan = zeros(1,129);
    for nIn = 1:12  % Cin=12
        x_ch = squeeze(xc_d4(nIn,:,:));  % [3,65]
        % zero insertion
        x_ins = zeros(H4_ins, W4_ins);
        x_ins(1:stride_h4:end, 1:stride_w4:end) = x_ch;  % [3,129]
        % rot90(kernel,2)
        kc = squeeze(tcw_d4(nIn,nOut,:,:));  % [3,3]
        krot = rot90(kc, 2);
        % pad left=1
        x_pad = [zeros(3,1) x_ins zeros(3,1)];  % [3,131]
        cr = zeros(1,129);
        for wi = 1:129
            xk = x_pad(:, wi:wi+2);
            cr(wi) = sum(round(xk .* krot * 2^(-14)), 'all');
        end
        y_chan = y_chan + cr;
    end
    yt_d4(nOut,:) = y_chan + tcb_d4(nOut);
end
export_txt(fullfile(out_dir, 'd4_tconv_out.txt'), yt_d4, 'int32');
fprintf('D4.2 tconv [1x129], range=[%d,%d]\n', min(yt_d4(:)), max(yt_d4(:)));

% Update cache
conv_cache_d2_out = xc_d4(:,2:3,:);

% Step D4.3: BN (Qr1=-11, Qr2=-11 — unique to D4)
bnw_d4 = importdata('decoder_de_convs_4_ops_2_weight.mat');
bnb_d4 = importdata('decoder_de_convs_4_ops_2_bias.mat');
bnm_d4 = importdata('decoder_de_convs_4_ops_2_running_mean.mat');
bnv_d4 = importdata('decoder_de_convs_4_ops_2_running_var.mat');

yb_d4 = bn_func(yt_d4, bnw_d4, bnb_d4, bnm_d4, bnv_d4, -11, -11);
export_txt(fullfile(out_dir, 'd4_bn_out.txt'), yb_d4, 'int32');
fprintf('D4.3 bn [1x129], range=[%d,%d]\n', min(yb_d4(:)), max(yb_d4(:)));

% Step D4.4: cTFA (C=1, W=129)
ty_d4 = export_ctfa_ta(fullfile(out_dir, 'd4'), yb_d4, 1, 129, tfa_cache_d4, 2, 1, ...
    importdata('decoder_de_convs_4_ops_4_ta_gru_weight_ih_l0.mat'), ...
    importdata('decoder_de_convs_4_ops_4_ta_gru_bias_ih_l0.mat'), ...
    importdata('decoder_de_convs_4_ops_4_ta_gru_weight_hh_l0.mat'), ...
    importdata('decoder_de_convs_4_ops_4_ta_gru_bias_hh_l0.mat'), ...
    importdata('decoder_de_convs_4_ops_4_ta_fc_weight.mat'), ...
    importdata('decoder_de_convs_4_ops_4_ta_fc_bias.mat'));
fy_d4 = export_ctfa_fa(fullfile(out_dir, 'd4'), yb_d4, 1, 129, 3, 4, ...
    importdata('decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0.mat'), ...
    importdata('decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0.mat'), ...
    importdata('decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0.mat'), ...
    importdata('decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0.mat'), ...
    importdata('decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0_reverse.mat'), ...
    importdata('decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0_reverse.mat'), ...
    importdata('decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0_reverse.mat'), ...
    importdata('decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0_reverse.mat'), ...
    importdata('decoder_de_convs_4_ops_4_fa_fc_weight.mat'), ...
    importdata('decoder_de_convs_4_ops_4_fa_fc_bias.mat'));
y_d4 = export_ctfa_fusion(fullfile(out_dir, 'd4'), yb_d4, ty_d4, fy_d4);
fprintf('D4.4 ctfa [1x129] FINAL, range=[%d,%d]\n', min(y_d4(:)), max(y_d4(:)));

%% ========================================================================
fprintf('\n=== Decoder diagnostic export complete ===\n');
fprintf('Output directory: %s\n', out_dir);
fprintf('\nFiles exported:\n');
ls_out = dir(fullfile(out_dir, '*.txt'));
for i = 1:length(ls_out)
    fprintf('  %s (%d bytes)\n', ls_out(i).name, ls_out(i).bytes);
end

%% ========================================================================
%% Helper functions
%% ========================================================================

function y = export_ctfa_ta(prefix, x, C, W, h_cache, nHidden, input_dim, ...
    ih_w, ih_b, hh_w, hh_b, fc_w, fc_b)
    % TA: per-channel energy → GRU → FC → sigmoid
    x_dq = x * 2^(-20);
    x_sq = x_dq.^2;
    x_agg = mean(x_sq, 2);  % [C, 1]
    x_t = Fix_point(x_agg.', 'u32f20');  % [1, C]
    export_txt([prefix '_ta_energy.txt'], x_t, 'uint32');

    [x_gru, ~] = GRU_module(x_t, nHidden, h_cache, ih_w, ih_b, hh_w, hh_b, -21, -16);
    export_txt([prefix '_ta_gru.txt'], x_gru, 'int16');

    x_fc = round(x_gru * fc_w * 2^(-8)) + fc_b;  % [1, C]
    export_txt([prefix '_ta_fc.txt'], x_fc, 'int32');

    x_dq2 = x_fc * 2^(-20);
    y = Fix_point(sigmoid_func(x_dq2), 'u16f15');  % [1, C]
    export_txt([prefix '_ta_sig.txt'], y, 'uint16');
end

function y = export_ctfa_fa(prefix, x, C, W, pad_len, nHidden, ...
    ih_w, ih_b, hh_w, hh_b, rih_w, rih_b, rhh_w, rhh_b, fc_w, fc_b)
    % FA: per-frequency energy → pad+reshape → BiGRU → FC → depad+reshape → sigmoid
    x_dq = x * 2^(-20);
    x_sq = x_dq.^2;
    x_agg = mean(x_sq, 1);  % [1, W]
    x_agg = Fix_point(x_agg, 'u32f20');
    export_txt([prefix '_fa_energy.txt'], x_agg, 'uint32');

    x_pad = [x_agg zeros(1, pad_len)];  % [1, W+pad]
    ngrp = 4;
    nseg = (W + pad_len) / ngrp;
    x_rs = reshape(x_pad, [ngrp, nseg])';  % [nseg, ngrp]
    export_txt([prefix '_fa_reshape.txt'], x_rs, 'uint32');

    % Forward GRU
    hc0 = zeros(1, nHidden);
    xg0 = zeros(nseg, nHidden);
    for i = 1:nseg
        [xg0(i,:), hc0] = GRU_module(x_rs(i,:), nHidden, hc0, ih_w, ih_b, hh_w, hh_b, -21, -16);
    end
    % Reverse GRU
    x_rev = x_rs(end:-1:1,:);
    hc1 = zeros(1, nHidden);
    xg1 = zeros(nseg, nHidden);
    for i = 1:nseg
        [xg1(nseg-i+1,:), hc1] = GRU_module(x_rev(i,:), nHidden, hc1, rih_w, rih_b, rhh_w, rhh_b, -21, -16);
    end
    x_gru = cat(2, xg0, xg1);  % [nseg, 2*nHidden]
    export_txt([prefix '_fa_gru.txt'], x_gru, 'int16');

    x_fc = round(x_gru * fc_w * 2^(-9)) + fc_b;  % [nseg, ngrp]
    export_txt([prefix '_fa_fc.txt'], x_fc, 'int32');

    % Reshape back + depad
    x_sh = reshape(x_fc.', 1, []);  % [1, W+pad]
    y_pre = x_sh(1:end-pad_len);    % [1, W]
    y = Fix_point(sigmoid_func(y_pre*2^(-20)), 'u16f15');
    export_txt([prefix '_fa_sig.txt'], y, 'uint16');
end

function y = export_ctfa_fusion(prefix, x, ta, fa)
    y_t = round(x .* (ta.') * 2^(-15));
    export_txt([prefix '_fusion_ta.txt'], y_t, 'int32');
    y = round(y_t .* fa * 2^(-15));
    export_txt([prefix '_fusion_final.txt'], y, 'int32');
end

function export_txt(filepath, data, dtype)
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
