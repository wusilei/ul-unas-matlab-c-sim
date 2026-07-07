function [y] = bn_func(x, weight, bias, running_mean, running_var, Qr1, Qr2)
% *********************************************************************** %
% Batch Normalize
% *********************************************************************** %

%%
% Normalize
x_norm = round( (x-running_mean).*running_var*2^(Qr1) );

% Scale and shift
y = round( x_norm.*weight*2^(Qr2) ) + bias;

end

