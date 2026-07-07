function [y] = ln_func(x, weight, bias, Qr)
% *********************************************************************** %
% Layer normalize
% *********************************************************************** %

%%
x_dq = x*2^(-20);
running_mean = mean(x_dq,'all');
running_var = 1 / ( sqrt( var(x_dq,1,'all') + 1e-8 ) );

running_mean = Fix_point(running_mean, 's32f20');
running_var = Fix_point(running_var, 'u16f11');

% Normalize
x_norm = round( (x-running_mean)*running_var*2^(-11) );

% Scale and shift
y = round( x_norm.*weight*2^(Qr) ) + bias;

end

