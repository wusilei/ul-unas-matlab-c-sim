function [y, ...
    tfa_cache_d0, ...
    tfa_cache_d1, ...
    conv_cache_d0, tfa_cache_d2, ...
    conv_cache_d1, tfa_cache_d3, ...
    conv_cache_d2, tfa_cache_d4] = Decoder_module(x, x_e4, tfa_cache_d0, ...
    x_e3, tfa_cache_d1, ...
    x_e2, conv_cache_d0, tfa_cache_d2, ...
    x_e1, conv_cache_d1, tfa_cache_d3, ...
    x_e0, conv_cache_d2, tfa_cache_d4)
% *********************************************************************** %
% Decoder module
% *********************************************************************** %

%% De-XDWS module 0 ----------------------------------------------------- %
[y_d0, tfa_cache_d0] = De_XDWS0_module(x, x_e4, tfa_cache_d0);

%% De-XMB module 0 ------------------------------------------------------ %
[y_d1, tfa_cache_d1] = De_XMB0_module(y_d0, x_e3, tfa_cache_d1);

%% De-XDWS module 1 ----------------------------------------------------- %
[y_d2, conv_cache_d0, tfa_cache_d2] = De_XDWS1_module(y_d1, x_e2, conv_cache_d0, tfa_cache_d2);

%% De-XMB module 1 ------------------------------------------------------ %
[y_d3, conv_cache_d1, tfa_cache_d3] = De_XMB1_module(y_d2, x_e1, conv_cache_d1, tfa_cache_d3);

%% DeXConv module ------------------------------------------------------- %
[y, conv_cache_d2, tfa_cache_d4] = De_XConv_module(y_d3, x_e0, conv_cache_d2, tfa_cache_d4);

end









