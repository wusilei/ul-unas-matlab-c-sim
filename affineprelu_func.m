function [y] = affineprelu_func(x, weight, bias, slope, Qr1, Qr2)

x_copy = x;

index = x < 0;
[row, ~] = find(index);
x(index) = round( x(index).*slope(row)*2^(Qr1) );

y = round( x_copy.*weight*2^(Qr2) ) + bias + x;

end

