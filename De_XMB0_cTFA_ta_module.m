function [y, h_cache] = De_XMB0_cTFA_ta_module(x, h_cache)

%% Load Para ---------------------------------------------------- %
ta_ih_weight = importdata('decoder_de_convs_1_pconv2_2_ta_gru_weight_ih_l0.mat');
ta_ih_bias = importdata('decoder_de_convs_1_pconv2_2_ta_gru_bias_ih_l0.mat');
ta_hh_weight = importdata('decoder_de_convs_1_pconv2_2_ta_gru_weight_hh_l0.mat');
ta_hh_bias = importdata('decoder_de_convs_1_pconv2_2_ta_gru_bias_hh_l0.mat');
ta_fc_weight = importdata('decoder_de_convs_1_pconv2_2_ta_fc_weight.mat');
ta_fc_bias = importdata('decoder_de_convs_1_pconv2_2_ta_fc_bias.mat');

%% Aggregation ---------------------------------------------------------- %
% Square + avg pool
x_dq = x*2^(-20);

x_squared = x_dq.^2;
x_agg = mean(x_squared,2);
x_t = x_agg.';

x_t = Fix_point(x_t, 'u32f20');

%% TA GRU --------------------------------------------------------------- %
nHidden = 48;
[x_gru, h_cache] = GRU_module(x_t, nHidden, h_cache, ta_ih_weight, ta_ih_bias, ta_hh_weight, ta_hh_bias, -21, -16);

%% TA FC ---------------------------------------------------------------- %
x_fc = round( x_gru*ta_fc_weight*2^(-9) ) + ta_fc_bias;

%% Sigmoid -------------------------------------------------------------- %
x_fc_dq = x_fc*2^(-20);
y_dq = sigmoid_func(x_fc_dq);

y = Fix_point(y_dq, 'u16f15');

end

