function [y, conv_cache, h_cache] = XDWS0_module(x, conv_cache, h_cache)

%% Point-wise Conv Block ------------------------------------------------ %
y_pconv = XDWS0_PConv_block(x);

%% Shuffle -------------------------------------------------------------- %
y_s = zeros(24,33);
y_s(1:2:end,:) = y_pconv(1:12,:);
y_s(2:2:end,:) = y_pconv(13:24,:);

%% Temporal Conv Block -------------------------------------------------- %
[y_tconv, conv_cache] = XDWS0_TConv_block(y_s, conv_cache);

%% cTFA_ta -------------------------------------------------------------- %
[y_ta, h_cache] = XDWS0_cTFA_ta_module(y_tconv, h_cache);

%% cTFA_fa -------------------------------------------------------------- %
y_fa = XDWS0_cTFA_fa_module(y_tconv);

%%
y_t = round( y_tconv.*(y_ta.')*2^(-15) );

y = round( y_t.*y_fa*2^(-15) );

end

