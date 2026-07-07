function [y, inter_cache] = GDPRNN_module(x, inter_cache, gdprnn_idx)
% *********************************************************************** %
% Grouped Dual-path RNN Module
% *********************************************************************** %

%% Intra-RNN ------------------------------------------------------------ %
y_intra = Intra_RNN_module(x.', gdprnn_idx);

%% Inter-RNN ------------------------------------------------------------ %
[y_inter, inter_cache] = Inter_RNN_module(y_intra, inter_cache, gdprnn_idx);

y = y_inter.';

end

