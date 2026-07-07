function [y, conv_cache, h_cache] = XConv_module(x, conv_cache, h_cache)

%% Temporal Conv Block -------------------------------------------------- %
[y_tconv, conv_cache] = XConv_TConv_block(x, conv_cache);

%% cTFA_ta -------------------------------------------------------------- %
[y_ta, h_cache] = XConv_cTFA_ta_module(y_tconv, h_cache);

%% cTFA_fa -------------------------------------------------------------- %
y_fa = XConv_cTFA_fa_module(y_tconv);

%%
y_t = round( y_tconv.*(y_ta.')*2^(-15) );

y = round( y_t.*y_fa*2^(-15) );

end

