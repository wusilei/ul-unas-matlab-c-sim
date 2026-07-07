function [y_e0, conv_cache_e0, tfa_cache_0, ...
    y_e1, conv_cache_e1, tfa_cache_1, ...
    y_e2, conv_cache_e2, tfa_cache_2, ...
    y_e3, tfa_cache_3, ...
    y_e4, tfa_cache_4] = Encoder_module(x, ...
    conv_cache_e0, tfa_cache_0, ...
    conv_cache_e1, tfa_cache_1, ...
    conv_cache_e2, tfa_cache_2, ...
    tfa_cache_3, ...
    tfa_cache_4)
% *********************************************************************** %
% Encoder module
% *********************************************************************** %

%% XConv module --------------------------------------------------------- %
[y_e0, conv_cache_e0, tfa_cache_0] = XConv_module(x, conv_cache_e0, tfa_cache_0);

%% XMB module 0 --------------------------------------------------------- %
[y_e1, conv_cache_e1, tfa_cache_1] = XMB0_module(y_e0, conv_cache_e1, tfa_cache_1);

%% XDWS module 0 -------------------------------------------------------- %
[y_e2, conv_cache_e2, tfa_cache_2] = XDWS0_module(y_e1, conv_cache_e2, tfa_cache_2);

%% XMB module 1 --------------------------------------------------------- %
[y_e3, tfa_cache_3] = XMB1_module(y_e2, tfa_cache_3);

%% XDWS module 1 -------------------------------------------------------- %
[y_e4, tfa_cache_4] = XDWS1_module(y_e3, tfa_cache_4);

end

