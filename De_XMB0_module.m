function [y, h_cache] = De_XMB0_module(x, x_enc, h_cache)

%%
x_con = x + x_enc;

%% Point-wise Conv Block 0 ---------------------------------------------- %
y_pconv0 = De_XMB0_PConv_block_0(x_con);

%% Shuffle -------------------------------------------------------------- %
y_s = zeros(24,33);
y_s(1:2:end,:) = y_pconv0(1:12,:);
y_s(2:2:end,:) = y_pconv0(13:24,:);

%% Non-Temporal DeConv Block -------------------------------------------- %
y_tconv = De_XMB0_nonTConv_block(y_s);

%% Point-wise Conv Block 1 ---------------------------------------------- %
y_pconv1 = De_XMB0_PConv_block_1(y_tconv);

%% cTFA_ta -------------------------------------------------------------- %
[y_ta, h_cache] = De_XMB0_cTFA_ta_module(y_pconv1, h_cache);

%% cTFA_fa -------------------------------------------------------------- %
y_fa = De_XMB0_cTFA_fa_module(y_pconv1);

%%
y_t = round( y_pconv1.*(y_ta.')*2^(-15) );

y_t = round( y_t.*y_fa*2^(-15) );

%% Shuffle
y = zeros(24,33);
y(1:2:end,:) = y_t(1:12,:);
y(2:2:end,:) = y_t(13:24,:);

end

