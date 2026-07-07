function [y] = log_gen(x_real, x_imag)
% *********************************************************************** %
% Log-magnitude Compression
% *********************************************************************** %

x_real_dq = x_real*2^(-20);
x_imag_dq = x_imag*2^(-20);

mag = sqrt(x_real_dq.^2 + x_imag_dq.^2);

clamped = max(mag, 1e-12); 

y = log10(clamped);

y = Fix_point(y, 's32f20');

end

