function [y] = non_gtconv2d_func(x, Cout, Hout, Wout, kernel_size, stride, weight, bias, Qr)

y = zeros(Cout,Wout);
x_insert = zeros(1,Wout);

for nOut = 1:Cout
    
    % Initial the conv2d result of the current output channel
    y_chan = zeros(Hout,Wout);

    x_chan = x(nOut,:);

    % Padding and insert 0
    x_insert(1:stride(1):end,1:stride(2):end) = x_chan;
    x_padd = [ zeros(1,2) x_insert zeros(1,2) ];
        
    kernel = ( squeeze(weight(nOut,1,:,:)) ).';
    kernel_chan = rot90(kernel,90);
        
    conv_result = zeros(Hout,Wout);
    for h_id = 1:Hout
        for w_id = 1:Wout
            x_kernel = x_padd( (h_id-1)+1:(h_id-1)+kernel_size(1), (w_id-1)+1:(w_id-1)+kernel_size(2) );
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

