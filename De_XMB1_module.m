function [y, conv_cache, h_cache] = De_XMB1_module(x, x_enc, conv_cache, h_cache)

%%
x_con = x + x_enc;

%% Point-wise Conv Block 0 ---------------------------------------------- %
y_pconv0 = De_XMB1_PConv_block_0(x_con);

%% Shuffle -------------------------------------------------------------- %
y_s = zeros(12,33);
y_s(1:2:end,:) = y_pconv0(1:6,:);
y_s(2:2:end,:) = y_pconv0(7:12,:);

%% Temporal DeConv Block ------------------------------------------------ %
[y_tconv, conv_cache] = De_XMB1_TConv_block(y_s, conv_cache);

%% Point-wise Conv Block 1 ---------------------------------------------- %
y_pconv1 = De_XMB1_PConv_block_1(y_tconv);

%% cTFA_ta -------------------------------------------------------------- %
[y_ta, h_cache] = De_XMB1_cTFA_ta_module(y_pconv1, h_cache);

%% cTFA_fa -------------------------------------------------------------- %
y_fa = De_XMB1_cTFA_fa_module(y_pconv1);

%%
y_t = round( y_pconv1.*(y_ta.')*2^(-15) );

y_t = round( y_t.*y_fa*2^(-15) );

%% Shuffle
y = zeros(12,65);
y(1:2:end,:) = y_t(1:6,:);
y(2:2:end,:) = y_t(7:12,:);

end

