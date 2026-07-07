function [y, conv_cache, h_cache] = De_XDWS1_module(x, x_enc, conv_cache, h_cache)

%%
x_con = x + x_enc;

%% Point-wise Conv Block ------------------------------------------------ %
y_pconv = De_XDWS1_PConv_block(x_con);

%% Shuffle -------------------------------------------------------------- %
y_s = zeros(24,33);
y_s(1:2:end,:) = y_pconv(1:12,:);
y_s(2:2:end,:) = y_pconv(13:24,:);

%% Temporal DeConv Block ------------------------------------------------ %
[y_tconv, conv_cache] = De_XDWS1_TConv_block(y_s, conv_cache);

%% cTFA_ta -------------------------------------------------------------- %
[y_ta, h_cache] = De_XDWS1_cTFA_ta_module(y_tconv, h_cache);

%% cTFA_fa -------------------------------------------------------------- %
y_fa = De_XDWS1_cTFA_fa_module(y_tconv);

%%
y_t = round( y_tconv.*(y_ta.')*2^(-15) );

y = round( y_t.*y_fa*2^(-15) );

end

