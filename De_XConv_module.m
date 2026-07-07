function [y, conv_cache, h_cache] = De_XConv_module(x, x_enc, conv_cache, h_cache)

%%
x_con = x + x_enc;

%% Temporal DeConv Block ------------------------------------------------ %
[y_tconv, conv_cache] = De_XConv_TConv_block(x_con, conv_cache);

%% cTFA_ta -------------------------------------------------------------- %
[y_ta, h_cache] = De_XConv_cTFA_ta_module(y_tconv, h_cache);

%% cTFA_fa -------------------------------------------------------------- %
y_fa = De_XConv_cTFA_fa_module(y_tconv);

%%
y_t = round( y_tconv.*(y_ta.')*2^(-15) );

y = round( y_t.*y_fa*2^(-15) );

end

