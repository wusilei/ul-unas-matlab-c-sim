function [y, h_cache] = GRU_module(x_t, nHidden, h_cache, ih_weight, ih_bias, hh_weight, hh_bias, Qr1, Qr2)

%%
ih_r_weight = ih_weight(:,1:nHidden);
ih_z_weight = ih_weight(:,nHidden+1:2*nHidden);
ih_n_weight = ih_weight(:,2*nHidden+1:end);
ih_r_bias = ih_bias(1:nHidden);
ih_z_bias = ih_bias(nHidden+1:2*nHidden);
ih_n_bias = ih_bias(2*nHidden+1:end);

hh_r_weight = hh_weight(:,1:nHidden);
hh_z_weight = hh_weight(:,nHidden+1:2*nHidden);
hh_n_weight = hh_weight(:,2*nHidden+1:end);
hh_r_bias = hh_bias(1:nHidden);
hh_z_bias = hh_bias(nHidden+1:2*nHidden);
hh_n_bias = hh_bias(2*nHidden+1:end);

%%
% Calculate reset gate R -------------------------------------------- %
r_t = round( x_t*ih_r_weight*2^(Qr1) ) + round( h_cache*hh_r_weight*2^(Qr2) ) + ih_r_bias + hh_r_bias;

r_t_dq = r_t*2^(-20);
r_t_dq = sigmoid_func(r_t_dq);
r_t = Fix_point(r_t_dq, 'u16f15');
    
% Calculate update gate Z ------------------------------------------- %
z_t = round( x_t*ih_z_weight*2^(Qr1) ) + round( h_cache*hh_z_weight*2^(Qr2) ) + ih_z_bias + hh_z_bias;

z_t_dq = z_t*2^(-20);
z_t_dq = sigmoid_func(z_t_dq);
z_t = Fix_point(z_t_dq, 'u16f15');
    
% Calculate candidate hidden state ---------------------------------- %
h_t = round( h_cache*hh_n_weight*2^(Qr2) ) + hh_n_bias;
n_t = round( x_t*ih_n_weight*2^(Qr1) ) + round( r_t.*h_t*2^(-15) ) + ih_n_bias;
    
n_t_dq = n_t*2^(-20);
n_t_dq = tanh(n_t_dq);
n_t = Fix_point(n_t_dq, 's16f15');
    
% Update hidden state ----------------------------------------------- %
h_cache = round( (32768-z_t).*n_t*2^(-15) ) + round( z_t.*h_cache*2^(-15) );
y = h_cache;

end

