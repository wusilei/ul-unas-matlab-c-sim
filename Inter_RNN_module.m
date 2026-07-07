function [y, h_cache] = Inter_RNN_module(x, h_cache, gdprnn_idx)

%% Load Para ------------------------------------------------------------ %
switch gdprnn_idx
    case 0
        rnn1_ih_weight = importdata('dpgrnn_0_inter_rnn_rnn1_weight_ih_l0.mat');
        rnn1_ih_bias = importdata('dpgrnn_0_inter_rnn_rnn1_bias_ih_l0.mat');
        rnn1_hh_weight = importdata('dpgrnn_0_inter_rnn_rnn1_weight_hh_l0.mat');
        rnn1_hh_bias = importdata('dpgrnn_0_inter_rnn_rnn1_bias_hh_l0.mat');
        rnn2_ih_weight = importdata('dpgrnn_0_inter_rnn_rnn2_weight_ih_l0.mat');
        rnn2_ih_bias = importdata('dpgrnn_0_inter_rnn_rnn2_bias_ih_l0.mat');
        rnn2_hh_weight = importdata('dpgrnn_0_inter_rnn_rnn2_weight_hh_l0.mat');
        rnn2_hh_bias = importdata('dpgrnn_0_inter_rnn_rnn2_bias_hh_l0.mat');
        fc_weight = importdata('dpgrnn_0_inter_fc_weight.mat');
        fc_bias = importdata('dpgrnn_0_inter_fc_bias.mat');
        ln_weight = importdata('dpgrnn_0_inter_ln_weight.mat');
        ln_bias = importdata('dpgrnn_0_inter_ln_bias.mat');
    case 1
        rnn1_ih_weight = importdata('dpgrnn_1_inter_rnn_rnn1_weight_ih_l0.mat');
        rnn1_ih_bias = importdata('dpgrnn_1_inter_rnn_rnn1_bias_ih_l0.mat');
        rnn1_hh_weight = importdata('dpgrnn_1_inter_rnn_rnn1_weight_hh_l0.mat');
        rnn1_hh_bias = importdata('dpgrnn_1_inter_rnn_rnn1_bias_hh_l0.mat');
        rnn2_ih_weight = importdata('dpgrnn_1_inter_rnn_rnn2_weight_ih_l0.mat');
        rnn2_ih_bias = importdata('dpgrnn_1_inter_rnn_rnn2_bias_ih_l0.mat');
        rnn2_hh_weight = importdata('dpgrnn_1_inter_rnn_rnn2_weight_hh_l0.mat');
        rnn2_hh_bias = importdata('dpgrnn_1_inter_rnn_rnn2_bias_hh_l0.mat');
        fc_weight = importdata('dpgrnn_1_inter_fc_weight.mat');
        fc_bias = importdata('dpgrnn_1_inter_fc_bias.mat');
        ln_weight = importdata('dpgrnn_1_inter_ln_weight.mat');
        ln_bias = importdata('dpgrnn_1_inter_ln_bias.mat');
end

%% GRNN(Grouped + Bidirectional=False) ---------------------------------- %
% Grouped
x0 = x(:,1:8);
x1 = x(:,9:16);

nHidden = 8;

[x0_gru, h_cache(:,1:8)] = GRU_module(x0, nHidden, h_cache(:,1:8), rnn1_ih_weight, rnn1_ih_bias, rnn1_hh_weight, rnn1_hh_bias, -13, -8);

[x1_gru, h_cache(:,9:16)] = GRU_module(x1, nHidden, h_cache(:,9:16), rnn2_ih_weight, rnn2_ih_bias, rnn2_hh_weight, rnn2_hh_bias, -13, -8);

x_gru = cat(2, x0_gru, x1_gru);

%% FC ------------------------------------------------------------------- %
x_fc = round( x_gru*fc_weight*2^(-9) ) + fc_bias;

%% LN ------------------------------------------------------------------- %
x_ln = ln_func(x_fc, ln_weight, ln_bias, -13);

%% Residual Connection -------------------------------------------------- %
y = x + x_ln;

end

