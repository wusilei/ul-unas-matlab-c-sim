function [y] = XMB1_PConv_block_1(x)

%% Load Para ------------------------------------------------------------ %
conv_weight = importdata('encoder_en_convs_3_pconv2_0_weight.mat');
conv_bias = importdata('encoder_en_convs_3_pconv2_0_bias.mat');

bn_weight = importdata('encoder_en_convs_3_pconv2_1_weight.mat');
bn_bias = importdata('encoder_en_convs_3_pconv2_1_bias.mat');
running_mean = importdata('encoder_en_convs_3_pconv2_1_running_mean.mat');
running_var = importdata('encoder_en_convs_3_pconv2_1_running_var.mat');

%% P-Conv2D ------------------------------------------------------------- %
Cin = 16;
Cout = 16;
Hout = 1;
Wout = 33;

y_conv0 = pconv2d_func(x(1:16,:), Cin, Cout, Hout, Wout, conv_weight(1:16,:), conv_bias(1:16), -14);

y_conv1 = pconv2d_func(x(17:32,:), Cin, Cout, Hout, Wout, conv_weight(17:32,:), conv_bias(17:32), -14);

y_conv = cat(1, y_conv0, y_conv1);

%% BN ------------------------------------------------------------------- %
y = bn_func(y_conv, bn_weight, bn_bias, running_mean, running_var, -14, -14);

end

