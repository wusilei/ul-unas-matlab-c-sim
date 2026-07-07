function [y] = MASK_module(x_mask, x_real, x_imag)

%%
y_real = round( x_real.*x_mask*2^(-15) );
y_imag = round( x_imag.*x_mask*2^(-15) );

y_q = cat(1,y_real,y_imag);

y = y_q*2^(-20);

end

