function [y, conv_cache] = XMB0_TConv_block(x, conv_cache)

%% Load Para ------------------------------------------------------------ %
conv_weight = importdata('encoder_en_convs_1_dconv_1_weight.mat');
conv_bias = importdata('encoder_en_convs_1_dconv_1_bias.mat');

bn_weight = importdata('encoder_en_convs_1_dconv_2_weight.mat');
bn_bias = importdata('encoder_en_convs_1_dconv_2_bias.mat');
running_mean = importdata('encoder_en_convs_1_dconv_2_running_mean.mat');
running_var = importdata('encoder_en_convs_1_dconv_2_running_var.mat');

affine_weight = importdata('encoder_en_convs_1_dconv_3_affine_weight.mat');
affine_bias = importdata('encoder_en_convs_1_dconv_3_affine_bias.mat');
affine_slope = importdata('encoder_en_convs_1_dconv_3_slope_weight.mat');

%% Conv2D --------------------------------------------------------------- %
Cout = 24;  % Cin = Cout
Hout = 1;
Wout = 33;
kernel_size = [2,3];
stride = [1,2];

y_conv = gconv2d_func(x, Cout, Hout, Wout, kernel_size, stride, conv_weight, conv_bias, conv_cache, -14);

% Update cache
conv_cache = x;

%% BN ------------------------------------------------------------------- %
y_bn = bn_func(y_conv, bn_weight, bn_bias, running_mean, running_var, -11, -14);

%% Affine PReLU --------------------------------------------------------- %
y = affineprelu_func(y_bn, affine_weight, affine_bias, affine_slope, -13, -13);

end

