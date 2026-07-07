function [y, h_cache] = XMB1_module(x, h_cache)

%% Point-wise Conv Block 0 ---------------------------------------------- %
y_pconv0 = XMB1_PConv_block_0(x);

%% Shuffle -------------------------------------------------------------- %
y_s = zeros(32,33);
y_s(1:2:end,:) = y_pconv0(1:16,:);
y_s(2:2:end,:) = y_pconv0(17:32,:);

%% Non-Temporal Conv Block ---------------------------------------------- %
y_tconv = XMB1_nonTConv_block(y_s);

%% Point-wise Conv Block 1 ---------------------------------------------- %
y_pconv1 = XMB1_PConv_block_1(y_tconv);

%% cTFA_ta -------------------------------------------------------------- %
[y_ta, h_cache] = XMB1_cTFA_ta_module(y_pconv1, h_cache);

%% cTFA_fa -------------------------------------------------------------- %
y_fa = XMB1_cTFA_fa_module(y_pconv1);

%%
y_t = round( y_pconv1.*(y_ta.')*2^(-15) );

y_t = round( y_t.*y_fa*2^(-15) );

%% Shuffle
y = zeros(32,33);
y(1:2:end,:) = y_t(1:16,:);
y(2:2:end,:) = y_t(17:32,:);

end

