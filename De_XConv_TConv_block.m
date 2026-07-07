function [y, conv_cache] = De_XConv_TConv_block(x, conv_cache)

%% Load Para ------------------------------------------------------------ %
conv_weight = importdata('decoder_de_convs_4_ops_1_weight.mat');
conv_bias = importdata('decoder_de_convs_4_ops_1_bias.mat');

bn_weight = importdata('decoder_de_convs_4_ops_2_weight.mat');
bn_bias = importdata('decoder_de_convs_4_ops_2_bias.mat');
running_mean = importdata('decoder_de_convs_4_ops_2_running_mean.mat');
running_var = importdata('decoder_de_convs_4_ops_2_running_var.mat');

%%
x_cache = cat(2, conv_cache, reshape(x, [12,1,65]));

%% Conv2D --------------------------------------------------------------- %
Cin = 12;
Cout = 1;
Hout = 1;
Wout = 129;
kernel_size = [3,3];
stride = [1,2];

y_conv = tconv2d_func(x_cache, Cin, Cout, Hout, Wout, kernel_size, stride, conv_weight, conv_bias, -14);

% Update cache
conv_cache = x_cache(:, 2:3, :);

%% BN ------------------------------------------------------------------- %
y = bn_func(y_conv, bn_weight, bn_bias, running_mean, running_var, -11, -11);

end

