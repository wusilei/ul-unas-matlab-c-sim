% diag_e0.m — Export E0 (XConv) intermediate outputs at each sub-step
% Run from x2000_deploy_v2/ directory

clear; clc;

script_dir = fileparts(mfilename('fullpath'));
parent_dir = fullfile(script_dir, '..');
addpath(fullfile(parent_dir, 'para_in_mat_FP'));
addpath(fullfile(parent_dir, 'test_wavs'));
addpath(parent_dir);

out_dir = fullfile(script_dir, 'diag_e0');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% Load test data (first frame)
[noisy_audio, fs] = audioread('noisy_fileid_1.wav');
N_fft = 512; win_len = 512; win_inc = 256;
hann_window = importdata('stft_window.mat');
[cmp_real, cmp_imag] = STFT_func(noisy_audio.', N_fft, win_len, win_inc, hann_window);

%% First frame
t = 1;
spec_real = Fix_point(cmp_real(t,:), 's32f20');
spec_imag = Fix_point(cmp_imag(t,:), 's32f20');

x = log_gen(spec_real, spec_imag);
erbfc_weight = importdata('erb_erb_fc_weight.mat');
x_bm = BM_module(x, erbfc_weight);

%% ===== E0: XConv module, step by step =====

% Step 0: Build x_c = [conv_cache; x]
conv_cache = zeros(2,129);
x_c = [conv_cache; x_bm];
export_mat(fullfile(out_dir, 'e0_x_c.txt'), x_c, 'int32');
fprintf('E0 Step 0: x_c [3x129]\n');

% Step 1: conv2d
conv_weight = importdata('encoder_en_convs_0_ops_1_weight.mat');
conv_bias   = importdata('encoder_en_convs_0_ops_1_bias.mat');

% Export conv weight (flattened in column-major order)
export_mat(fullfile(out_dir, 'e0_conv_weight.txt'), conv_weight, 'int16');
export_mat(fullfile(out_dir, 'e0_conv_bias.txt'), conv_bias, 'int32');

% Do the conv2d manually to export sub-results
Cin = 1; Cout = 12; Hout = 1; Wout = 65;
kernel_size = [3,3]; stride = [1,2]; Qr = -14;

y_conv = zeros(Cout, Wout);
for nOut = 1:Cout
    y_chan = zeros(Hout, Wout);
    for nIn = 1:Cin
        x_chan = x_c;
        x_padd = [zeros(3,1) x_chan zeros(3,1)];
        kernel_chan = squeeze(conv_weight(nOut,nIn,:,:));
        conv_result = zeros(Hout, Wout);
        for h_id = 1:Hout
            for w_id = 1:Wout
                x_kernel = x_padd((h_id-1)*stride(1)+1:(h_id-1)*stride(1)+kernel_size(1), ...
                                  (w_id-1)*stride(2)+1:(w_id-1)*stride(2)+kernel_size(2));
                temp = round(x_kernel.*kernel_chan*2^(Qr));
                conv_result(h_id,w_id) = sum(temp,'all');
            end
        end
        y_chan = y_chan + conv_result;
    end
    y_chan = y_chan + conv_bias(nOut);
    y_conv(nOut,:) = y_chan;
end
export_mat(fullfile(out_dir, 'e0_conv_out.txt'), y_conv, 'int32');
fprintf('E0 Step 1: conv2d_out [12x65], range=[%d,%d]\n', min(y_conv(:)), max(y_conv(:)));

% Step 2: BN
bn_weight = importdata('encoder_en_convs_0_ops_2_weight.mat');
bn_bias   = importdata('encoder_en_convs_0_ops_2_bias.mat');
running_mean = importdata('encoder_en_convs_0_ops_2_running_mean.mat');
running_var  = importdata('encoder_en_convs_0_ops_2_running_var.mat');

% Export BN params
export_mat(fullfile(out_dir, 'e0_bn_weight.txt'), bn_weight, 'int16');
export_mat(fullfile(out_dir, 'e0_bn_bias.txt'), bn_bias, 'int32');
export_mat(fullfile(out_dir, 'e0_bn_mean.txt'), running_mean, 'int32');
export_mat(fullfile(out_dir, 'e0_bn_var.txt'), running_var, 'uint16');

% Export BN intermediate: (x - running_mean) .* running_var
x_norm = round((y_conv - running_mean) .* running_var * 2^(-14));
export_mat(fullfile(out_dir, 'e0_bn_norm.txt'), x_norm, 'int32');
fprintf('E0 Step 2a: bn_norm [12x65], range=[%d,%d]\n', min(x_norm(:)), max(x_norm(:)));

% BN final
y_bn = round(x_norm .* bn_weight * 2^(-14)) + bn_bias;
export_mat(fullfile(out_dir, 'e0_bn_out.txt'), y_bn, 'int32');
fprintf('E0 Step 2b: bn_out [12x65], range=[%d,%d]\n', min(y_bn(:)), max(y_bn(:)));

% Step 3: AffinePReLU
affine_weight = importdata('encoder_en_convs_0_ops_3_affine_weight.mat');
affine_bias   = importdata('encoder_en_convs_0_ops_3_affine_bias.mat');
affine_slope  = importdata('encoder_en_convs_0_ops_3_slope_weight.mat');

export_mat(fullfile(out_dir, 'e0_aff_weight.txt'), affine_weight, 'int16');
export_mat(fullfile(out_dir, 'e0_aff_bias.txt'), affine_bias, 'int32');
export_mat(fullfile(out_dir, 'e0_aff_slope.txt'), affine_slope, 'int16');

% AffinePReLU computation (from affineprelu_func.m)
x_copy = y_bn;
index = y_bn < 0;
[row, ~] = find(index);
y_bn(index) = round(y_bn(index) .* affine_slope(row) * 2^(-13));
y_aff = round(x_copy .* affine_weight * 2^(-13)) + affine_bias + y_bn;

export_mat(fullfile(out_dir, 'e0_aff_out.txt'), y_aff, 'int32');
fprintf('E0 Step 3: aff_out [12x65], range=[%d,%d]\n', min(y_aff(:)), max(y_aff(:)));

fprintf('\nAll E0 intermediates exported to: %s\n', out_dir);
fprintf('Files:\n');
ls_out = dir(fullfile(out_dir, '*.txt'));
for i = 1:length(ls_out)
    fprintf('  %s (%d bytes)\n', ls_out(i).name, ls_out(i).bytes);
end

%% Helper: export matrix as text (one value per line)
function export_mat(filepath, data, dtype)
    data_flat = data(:);
    fid = fopen(filepath, 'w');
    if strcmp(dtype, 'int16')
        for i = 1:numel(data_flat)
            fprintf(fid, '%d\n', int16(data_flat(i)));
        end
    elseif strcmp(dtype, 'uint16')
        for i = 1:numel(data_flat)
            fprintf(fid, '%u\n', uint16(data_flat(i)));
        end
    else  % int32
        for i = 1:numel(data_flat)
            fprintf(fid, '%d\n', int32(data_flat(i)));
        end
    end
    fclose(fid);
end
