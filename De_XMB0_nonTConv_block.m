function [y] = De_XMB0_nonTConv_block(x)

%% Load Para ------------------------------------------------------------ %
conv_weight = importdata('decoder_de_convs_1_dconv_1_weight.mat');
conv_bias = importdata('decoder_de_convs_1_dconv_1_bias.mat');

bn_weight = importdata('decoder_de_convs_1_dconv_2_weight.mat');
bn_bias = importdata('decoder_de_convs_1_dconv_2_bias.mat');
running_mean = importdata('decoder_de_convs_1_dconv_2_running_mean.mat');
running_var = importdata('decoder_de_convs_1_dconv_2_running_var.mat');

affine_weight = importdata('decoder_de_convs_1_dconv_3_affine_weight.mat');
affine_bias = importdata('decoder_de_convs_1_dconv_3_affine_bias.mat');
affine_slope = importdata('decoder_de_convs_1_dconv_3_slope_weight.mat');

%% Conv2D --------------------------------------------------------------- %
Cout = 24;  % Cin = Cout
Hout = 1;
Wout = 33;
kernel_size = [1,5];
stride = [1,1];

y_conv = non_gtconv2d_func(x, Cout, Hout, Wout, kernel_size, stride, conv_weight, conv_bias, -14);

%% BN ------------------------------------------------------------------- %
y_bn = bn_func(y_conv, bn_weight, bn_bias, running_mean, running_var, -11, -14);

%% Affine PReLU --------------------------------------------------------- %
y = affineprelu_func(y_bn, affine_weight, affine_bias, affine_slope, -13, -13);

end

