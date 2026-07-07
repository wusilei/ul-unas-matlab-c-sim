function [y] = BM_module(x, weight)
% *********************************************************************** %
% Band Merging Module
% *********************************************************************** %

% Low freq  0-2000Hz no precessed
% High freq 2031.25~8000Hz band merging
y = zeros(1,129);
y(1,1:65) = x(1,1:65);

y(1,66:end) = round( x(1,66:end)*weight*2^(-15) );

end

