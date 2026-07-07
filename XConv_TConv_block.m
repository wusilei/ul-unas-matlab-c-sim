function [y, conv_cache] = XConv_TConv_block(x, conv_cache)

%% Cat Cache ------------------------------------------------------------ %
x_c = [ conv_cache; x ];

%% Load Para ------------------------------------------------------------ %
conv_weight = importdata('encoder_en_convs_0_ops_1_weight.mat');
conv_bias = importdata('encoder_en_convs_0_ops_1_bias.mat');

bn_weight = importdata('encoder_en_convs_0_ops_2_weight.mat');
bn_bias = importdata('encoder_en_convs_0_ops_2_bias.mat');
running_mean = importdata('encoder_en_convs_0_ops_2_running_mean.mat');
running_var = importdata('encoder_en_convs_0_ops_2_running_var.mat');

affine_weight = importdata('encoder_en_convs_0_ops_3_affine_weight.mat');
affine_bias = importdata('encoder_en_convs_0_ops_3_affine_bias.mat');
affine_slope = importdata('encoder_en_convs_0_ops_3_slope_weight.mat');

%% Conv2D --------------------------------------------------------------- %
Cin = 1;
Cout = 12;
Hout = 1;
Wout = 65;
kernel_size = [3,3];
stride = [1,2];

y_conv = conv2d_func(x_c, Cin, Cout, Hout, Wout, kernel_size, stride, conv_weight, conv_bias, -14);

% Update cache
conv_cache = x_c(2:3,:);

%% BN ------------------------------------------------------------------- %
y_bn = bn_func(y_conv, bn_weight, bn_bias, running_mean, running_var, -14, -14);

%% Affine PReLU --------------------------------------------------------- %
y = affineprelu_func(y_bn, affine_weight, affine_bias, affine_slope, -13, -13);

end

