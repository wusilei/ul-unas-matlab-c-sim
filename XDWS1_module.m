function [y, h_cache] = XDWS1_module(x, h_cache)

%% Point-wise Conv Block ------------------------------------------------ %
y_pconv = XDWS1_PConv_block(x);

%% Shuffle -------------------------------------------------------------- %
y_s = zeros(16,33);
y_s(1:2:end,:) = y_pconv(1:8,:);
y_s(2:2:end,:) = y_pconv(9:16,:);

%% Non-Temporal Conv Block ---------------------------------------------- %
y_tconv = XDWS1_nonTConv_block(y_s);

%% cTFA_ta -------------------------------------------------------------- %
[y_ta, h_cache] = XDWS1_cTFA_ta_module(y_tconv, h_cache);

%% cTFA_fa -------------------------------------------------------------- %
y_fa = XDWS1_cTFA_fa_module(y_tconv);

%%
y_t = round( y_tconv.*(y_ta.')*2^(-15) );

y = round( y_t.*y_fa*2^(-15) );

end

