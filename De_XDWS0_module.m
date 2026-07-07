function [y, h_cache] = De_XDWS0_module(x, x_enc, h_cache)

%%
x_con = x + x_enc;

%% Point-wise Conv Block ------------------------------------------------ %
y_pconv = De_XDWS0_PConv_block(x_con);

%% Shuffle -------------------------------------------------------------- %
y_s = zeros(32,33);
y_s(1:2:end,:) = y_pconv(1:16,:);
y_s(2:2:end,:) = y_pconv(17:32,:);

%% Non-Temporal DeConv Block -------------------------------------------- %
y_tconv = De_XDWS0_nonTConv_block(y_s);

%% cTFA_ta -------------------------------------------------------------- %
[y_ta, h_cache] = De_XDWS0_cTFA_ta_module(y_tconv, h_cache);

%% cTFA_fa -------------------------------------------------------------- %
y_fa = De_XDWS0_cTFA_fa_module(y_tconv);

%%
y_t = round( y_tconv.*(y_ta.')*2^(-15) );

y = round( y_t.*y_fa*2^(-15) );

end

