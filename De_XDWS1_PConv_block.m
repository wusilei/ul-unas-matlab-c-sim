function [y] = De_XDWS1_PConv_block(x)

%% Load Para ------------------------------------------------------------ %
conv_weight = importdata('decoder_de_convs_2_pconv_0_weight.mat');
conv_bias = importdata('decoder_de_convs_2_pconv_0_bias.mat');

bn_weight = importdata('decoder_de_convs_2_pconv_1_weight.mat');
bn_bias = importdata('decoder_de_convs_2_pconv_1_bias.mat');
running_mean = importdata('decoder_de_convs_2_pconv_1_running_mean.mat');
running_var = importdata('decoder_de_convs_2_pconv_1_running_var.mat');

affine_weight = importdata('decoder_de_convs_2_pconv_2_affine_weight.mat');
affine_bias = importdata('decoder_de_convs_2_pconv_2_affine_bias.mat');
affine_slope = importdata('decoder_de_convs_2_pconv_2_slope_weight.mat');

%% P-Conv2D ------------------------------------------------------------- %
Cin = 12;
Cout = 12;
Hout = 1;
Wout = 33;

y_conv0 = pconv2d_func(x(1:12,:), Cin, Cout, Hout, Wout, conv_weight(1:12,:), conv_bias(1:12), -14);

y_conv1 = pconv2d_func(x(13:24,:), Cin, Cout, Hout, Wout, conv_weight(13:24,:), conv_bias(13:24), -14);

y_conv = cat(1, y_conv0, y_conv1);

%% BN ------------------------------------------------------------------- %
y_bn = bn_func(y_conv, bn_weight, bn_bias, running_mean, running_var, -11, -14);

%% Affine PReLU --------------------------------------------------------- %
y = affineprelu_func(y_bn, affine_weight, affine_bias, affine_slope, -13, -13);

end

