function [y] = Intra_RNN_module(x, gdprnn_idx)

%% Load Para ------------------------------------------------------------ %
switch gdprnn_idx
    case 0
        rnn1_ih_weight = importdata('dpgrnn_0_intra_rnn_rnn1_weight_ih_l0.mat');
        rnn1_ih_bias = importdata('dpgrnn_0_intra_rnn_rnn1_bias_ih_l0.mat');
        rnn1_hh_weight = importdata('dpgrnn_0_intra_rnn_rnn1_weight_hh_l0.mat');
        rnn1_hh_bias = importdata('dpgrnn_0_intra_rnn_rnn1_bias_hh_l0.mat');
        rnn1_re_ih_weight = importdata('dpgrnn_0_intra_rnn_rnn1_weight_ih_l0_reverse.mat');
        rnn1_re_ih_bias = importdata('dpgrnn_0_intra_rnn_rnn1_bias_ih_l0_reverse.mat');
        rnn1_re_hh_weight = importdata('dpgrnn_0_intra_rnn_rnn1_weight_hh_l0_reverse.mat');
        rnn1_re_hh_bias = importdata('dpgrnn_0_intra_rnn_rnn1_bias_hh_l0_reverse.mat');
        rnn2_ih_weight = importdata('dpgrnn_0_intra_rnn_rnn2_weight_ih_l0.mat');
        rnn2_ih_bias = importdata('dpgrnn_0_intra_rnn_rnn2_bias_ih_l0.mat');
        rnn2_hh_weight = importdata('dpgrnn_0_intra_rnn_rnn2_weight_hh_l0.mat');
        rnn2_hh_bias = importdata('dpgrnn_0_intra_rnn_rnn2_bias_hh_l0.mat');
        rnn2_re_ih_weight = importdata('dpgrnn_0_intra_rnn_rnn2_weight_ih_l0_reverse.mat');
        rnn2_re_ih_bias = importdata('dpgrnn_0_intra_rnn_rnn2_bias_ih_l0_reverse.mat');
        rnn2_re_hh_weight = importdata('dpgrnn_0_intra_rnn_rnn2_weight_hh_l0_reverse.mat');
        rnn2_re_hh_bias = importdata('dpgrnn_0_intra_rnn_rnn2_bias_hh_l0_reverse.mat');
        fc_weight = importdata('dpgrnn_0_intra_fc_weight.mat');
        fc_bias = importdata('dpgrnn_0_intra_fc_bias.mat');
        ln_weight = importdata('dpgrnn_0_intra_ln_weight.mat');
        ln_bias = importdata('dpgrnn_0_intra_ln_bias.mat');
    case 1
        rnn1_ih_weight = importdata('dpgrnn_1_intra_rnn_rnn1_weight_ih_l0.mat');
        rnn1_ih_bias = importdata('dpgrnn_1_intra_rnn_rnn1_bias_ih_l0.mat');
        rnn1_hh_weight = importdata('dpgrnn_1_intra_rnn_rnn1_weight_hh_l0.mat');
        rnn1_hh_bias = importdata('dpgrnn_1_intra_rnn_rnn1_bias_hh_l0.mat');
        rnn1_re_ih_weight = importdata('dpgrnn_1_intra_rnn_rnn1_weight_ih_l0_reverse.mat');
        rnn1_re_ih_bias = importdata('dpgrnn_1_intra_rnn_rnn1_bias_ih_l0_reverse.mat');
        rnn1_re_hh_weight = importdata('dpgrnn_1_intra_rnn_rnn1_weight_hh_l0_reverse.mat');
        rnn1_re_hh_bias = importdata('dpgrnn_1_intra_rnn_rnn1_bias_hh_l0_reverse.mat');
        rnn2_ih_weight = importdata('dpgrnn_1_intra_rnn_rnn2_weight_ih_l0.mat');
        rnn2_ih_bias = importdata('dpgrnn_1_intra_rnn_rnn2_bias_ih_l0.mat');
        rnn2_hh_weight = importdata('dpgrnn_1_intra_rnn_rnn2_weight_hh_l0.mat');
        rnn2_hh_bias = importdata('dpgrnn_1_intra_rnn_rnn2_bias_hh_l0.mat');
        rnn2_re_ih_weight = importdata('dpgrnn_1_intra_rnn_rnn2_weight_ih_l0_reverse.mat');
        rnn2_re_ih_bias = importdata('dpgrnn_1_intra_rnn_rnn2_bias_ih_l0_reverse.mat');
        rnn2_re_hh_weight = importdata('dpgrnn_1_intra_rnn_rnn2_weight_hh_l0_reverse.mat');
        rnn2_re_hh_bias = importdata('dpgrnn_1_intra_rnn_rnn2_bias_hh_l0_reverse.mat');
        fc_weight = importdata('dpgrnn_1_intra_fc_weight.mat');
        fc_bias = importdata('dpgrnn_1_intra_fc_bias.mat');
        ln_weight = importdata('dpgrnn_1_intra_ln_weight.mat');
        ln_bias = importdata('dpgrnn_1_intra_ln_bias.mat');
end

%% GRNN(Grouped + Bidirectional=True) ----------------------------------- %
% Grouped
x0 = x(:,1:8);
x1 = x(:,9:16);

nHidden = 4;

x0_gru = BiGRU_module(x0, nHidden, rnn1_ih_weight, rnn1_ih_bias, rnn1_hh_weight, rnn1_hh_bias, rnn1_re_ih_weight, rnn1_re_ih_bias, rnn1_re_hh_weight, rnn1_re_hh_bias, -21, -16);

x1_gru = BiGRU_module(x1, nHidden, rnn2_ih_weight, rnn2_ih_bias, rnn2_hh_weight, rnn2_hh_bias, rnn2_re_ih_weight, rnn2_re_ih_bias, rnn2_re_hh_weight, rnn2_re_hh_bias, -21, -16);

x_gru = cat(2, x0_gru, x1_gru);

%% FC ------------------------------------------------------------------- %
x_fc = round( x_gru*fc_weight*2^(-9) ) + fc_bias;

%% LN ------------------------------------------------------------------- %
x_ln = ln_func(x_fc, ln_weight, ln_bias, -14);

%% Residual Connection -------------------------------------------------- %
y = x + x_ln;

end

