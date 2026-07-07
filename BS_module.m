function [y] = BS_module(x, weight)
% *********************************************************************** %
% Band Splitting Module
% *********************************************************************** %

% Low freq  0-2000Hz no precessed
% High freq 2031.25~8000Hz band splitting
y = zeros(1,257);
y(:,1:65) = x(:,1:65);

y(:,66:end) = round( x(1,66:129)*weight*2^(-15) );

end

