function [enhance_audio] = ISTFT_func(cmp_spec, N_fft, win_len, win_inc, hann_window)

%%
[T,F] = size(cmp_spec);
output_len = (T-1)*win_inc + win_len;

full_spec = zeros(T,N_fft);
full_spec(:,1:F) = cmp_spec;
full_spec(:,F+1:end) = flip( conj(cmp_spec(:,2:N_fft/2)) );

enhance_audio = zeros(1,output_len);
win_sum_sq = zeros(1,output_len);
for t = 1:T
    x_time = ifft(full_spec(t,:), N_fft, 2, 'symmetric');
    x_win = x_time .* hann_window;
    
    start_idx = (t-1)*win_inc + 1;
    end_idx = start_idx + win_len - 1;
    
    enhance_audio(start_idx:end_idx) = enhance_audio(start_idx:end_idx) + x_win;
    
    win_sum_sq(start_idx:end_idx) = win_sum_sq(start_idx:end_idx) + hann_window.^2;
end

eps = 1e-8;
win_sum_sq(win_sum_sq<eps) = 1;
enhance_audio = enhance_audio ./ win_sum_sq;

pad_len = N_fft/2;
enhance_audio = enhance_audio(pad_len+1:end-pad_len);

end

