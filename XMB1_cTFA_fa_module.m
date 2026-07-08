function [y] = XMB1_cTFA_fa_module(x)

%% Load Para ------------------------------------------------------------ %
fa_ih_weight = importdata('encoder_en_convs_3_pconv2_2_fa_gru_weight_ih_l0.mat');
fa_ih_bias = importdata('encoder_en_convs_3_pconv2_2_fa_gru_bias_ih_l0.mat');
fa_hh_weight = importdata('encoder_en_convs_3_pconv2_2_fa_gru_weight_hh_l0.mat');
fa_hh_bias = importdata('encoder_en_convs_3_pconv2_2_fa_gru_bias_hh_l0.mat');

fa_re_ih_weight = importdata('encoder_en_convs_3_pconv2_2_fa_gru_weight_ih_l0_reverse.mat');
fa_re_ih_bias = importdata('encoder_en_convs_3_pconv2_2_fa_gru_bias_ih_l0_reverse.mat');
fa_re_hh_weight = importdata('encoder_en_convs_3_pconv2_2_fa_gru_weight_hh_l0_reverse.mat');
fa_re_hh_bias = importdata('encoder_en_convs_3_pconv2_2_fa_gru_bias_hh_l0_reverse.mat');

fa_fc_weight = importdata('encoder_en_convs_3_pconv2_2_fa_fc_weight.mat');
fa_fc_bias = importdata('encoder_en_convs_3_pconv2_2_fa_fc_bias.mat');

%% Aggregation ---------------------------------------------------------- %
% Square + avg pool
x_dq = x*2^(-20);

x_squared = x_dq.^2;
x_agg = mean(x_squared,1);

x_agg = Fix_point(x_agg, 'u32f20');

%% Attention Generation ------------------------------------------------- %
pad_len = 3;
x_pad = [x_agg zeros(1,pad_len)];
x_t = reshape(x_pad,[4,9])';

%% FA Bi-GRU ------------------------------------------------------------ %
nHidden = 4;
h_cache0 = zeros(1,nHidden);
x_gru0 = zeros(9,nHidden);
for i = 1:9
    [x_gru0(i,:), h_cache0] = GRU_module(x_t(i,:), nHidden, h_cache0, fa_ih_weight, fa_ih_bias, fa_hh_weight, fa_hh_bias, -21, -16);
end

x_t_re = x_t(end:-1:1,:);
h_cache1 = zeros(1,nHidden);
x_gru1 = zeros(9,nHidden);
for i = 1:9
    [x_gru1(9-i+1,:), h_cache1] = GRU_module(x_t_re(i,:), nHidden, h_cache1, fa_re_ih_weight, fa_re_ih_bias, fa_re_hh_weight, fa_re_hh_bias, -21, -16);
end

x_gru = cat(2, x_gru0, x_gru1);

%% TA FC ---------------------------------------------------------------- %
x_fc = round( x_gru*fa_fc_weight*2^(-9) ) + fa_fc_bias;

x_shape = reshape(x_fc.', 1, []); 

%% Sigmoid -------------------------------------------------------------- %
x_shape_dq = x_shape*2^(-20);
y_dq = sigmoid_func(x_shape_dq(1:end-pad_len));

y = Fix_point(y_dq, 'u16f15');

end

