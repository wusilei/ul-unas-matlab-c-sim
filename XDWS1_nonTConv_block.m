function [y] = XDWS1_nonTConv_block(x)

%% Load Para ------------------------------------------------------------ %
conv_weight = importdata('encoder_en_convs_4_dconv_1_weight.mat');
conv_bias = importdata('encoder_en_convs_4_dconv_1_bias.mat');

bn_weight = importdata('encoder_en_convs_4_dconv_2_weight.mat');
bn_bias = importdata('encoder_en_convs_4_dconv_2_bias.mat');
running_mean = importdata('encoder_en_convs_4_dconv_2_running_mean.mat');
running_var = importdata('encoder_en_convs_4_dconv_2_running_var.mat');

affine_weight = importdata('encoder_en_convs_4_dconv_3_affine_weight.mat');
affine_bias = importdata('encoder_en_convs_4_dconv_3_affine_bias.mat');
affine_slope = importdata('encoder_en_convs_4_dconv_3_slope_weight.mat');

%% Conv2D --------------------------------------------------------------- %
Cout = 16;  % Cin = Cout
Hout = 1;
Wout = 33;
kernel_size = [1,5];
stride = [1,1];

y_conv = non_gconv2d_func(x, Cout, Hout, Wout, kernel_size, stride, conv_weight, conv_bias, -14);

%% BN ------------------------------------------------------------------- %
y_bn = bn_func(y_conv, bn_weight, bn_bias, running_mean, running_var, -14, -14);

%% Affine PReLU --------------------------------------------------------- %
y = affineprelu_func(y_bn, affine_weight, affine_bias, affine_slope, -13, -13);

end

