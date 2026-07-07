function [y] = pconv2d_func(x, Cin, Cout, Hout, Wout, weight, bias, Qr)

y = zeros(Cout,Wout);

for nOut = 1:Cout
    
    % Initial the conv2d result of the current output channel
    y_chan = zeros(Hout,Wout);
    
    for nIn = 1:Cin
        
        x_chan = x(nIn,:);
        
        kernel_chan = squeeze(weight(nOut,nIn,:,:));

        conv_result = round( x_chan*kernel_chan*2^(Qr) );

        % Accumulate to the output channel
        y_chan = y_chan + conv_result;

    end
    
    % Add the bias
    y_chan = y_chan + bias(nOut);
    
    y(nOut,:) = y_chan;
end

end

