function [y] = gconv2d_func(x, Cout, Hout, Wout, kernel_size, stride, weight, bias, conv_cache, Qr)

y = zeros(Cout,Wout);

for nOut = 1:Cout
    
    % Initial the conv2d result of the current output channel
    y_chan = zeros(Hout,Wout);

    x_chan = [conv_cache(nOut,:); x(nOut,:)];

    x_padd = [ zeros(2,1) x_chan zeros(2,1) ];
        
    kernel_chan = squeeze(weight(nOut,1,:,:));
        
    conv_result = zeros(Hout,Wout);
    for h_id = 1:Hout
        for w_id = 1:Wout
            x_kernel = x_padd( (h_id-1)*stride(1)+1:(h_id-1)*stride(1)+kernel_size(1), (w_id-1)*stride(2)+1:(w_id-1)*stride(2)+kernel_size(2) );
            temp = round( x_kernel.*kernel_chan*2^(Qr) );
            conv_result(h_id,w_id) = sum(temp,'all');
        end
    end

    % Accumulate to the output channel
    y_chan = y_chan + conv_result;
    
    % Add the bias
    y_chan = y_chan + bias(nOut);
    
    y(nOut,:) = y_chan;
end

end

