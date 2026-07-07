function [cmp_real,cmp_imag] = STFT_func(x, N_fft, win_len, win_inc, hann_window)

%%
pad_len = N_fft/2;
x_padd = [ flip(x(2:pad_len+1)) x flip(x(end-pad_len:end-1)) ];

total_len = length(x_padd);
T = floor( (total_len-win_len)/win_inc ) + 1;
F = N_fft/2 + 1;

cmp_real = zeros(T,F);
cmp_imag = zeros(T,F);

for t = 1:T
    start_idx = (t-1)*win_inc + 1;
    end_idx = start_idx + win_len - 1;
    
    x_frame = x_padd(start_idx:end_idx);
    x_win = x_frame .* hann_window;
    
    x_spec = fft(x_win,N_fft);
    
    cmp_real(t,:) = real(x_spec(1:F));
    cmp_imag(t,:) = imag(x_spec(1:F));
end

end

