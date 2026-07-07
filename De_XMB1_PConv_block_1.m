function [y] = De_XMB1_PConv_block_1(x)

%% Load Para ------------------------------------------------------------ %
conv_weight = importdata('decoder_de_convs_3_pconv2_0_weight.mat');
conv_bias = importdata('decoder_de_convs_3_pconv2_0_bias.mat');

bn_weight = importdata('decoder_de_convs_3_pconv2_1_weight.mat');
bn_bias = importdata('decoder_de_convs_3_pconv2_1_bias.mat');
running_mean = importdata('decoder_de_convs_3_pconv2_1_running_mean.mat');
running_var = importdata('decoder_de_convs_3_pconv2_1_running_var.mat');

%% P-Conv2D ------------------------------------------------------------- %
Cin = 6;
Cout = 6;
Hout = 1;
Wout = 65;

y_conv0 = pconv2d_func(x(1:6,:), Cin, Cout, Hout, Wout, conv_weight(1:6,:), conv_bias(1:6), -14);

y_conv1 = pconv2d_func(x(7:12,:), Cin, Cout, Hout, Wout, conv_weight(7:12,:), conv_bias(7:12), -14);

y_conv = cat(1, y_conv0, y_conv1);

%% BN ------------------------------------------------------------------- %
y = bn_func(y_conv, bn_weight, bn_bias, running_mean, running_var, -11, -11);

end

