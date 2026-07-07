function [x_q] = Fix_point(x, q_set)
% Fix_point — fixed-point quantization WITHOUT Fixed-Point Designer toolbox
% Replaces fi() with native MATLAB operations:
%   1. Scale by 2^frac_bits, round, saturate to [min, max]
%   2. Dequantize back to single
%   3. Return both stored-integer (x_q)
switch q_set
    case 's32f20'
        x_q = quantize_fp(x, 1, 32, 20);
    case 'u32f20'
        x_q = quantize_fp(x, 0, 32, 20);
    case 's16f15'
        x_q = quantize_fp(x, 1, 16, 15);
    case 's16f14'
        x_q = quantize_fp(x, 1, 16, 14);
    case 's16f13'
        x_q = quantize_fp(x, 1, 16, 13);
    case 's16f12'
        x_q = quantize_fp(x, 1, 16, 12);
    case 's16f11'
        x_q = quantize_fp(x, 1, 16, 11);
    case 'u16f15'
        x_q = quantize_fp(x, 0, 16, 15);
    case 'u16f14'
        x_q = quantize_fp(x, 0, 16, 14);
    case 'u16f11'
        x_q = quantize_fp(x, 0, 16, 11);
    otherwise
        error('Fix_point: unknown quantization setting: %s', q_set);
end

end

function [x_q] = quantize_fp(x, signed, word_len, frac_len)
    scale = 2^frac_len;
    if signed
        max_int = 2^(word_len - 1) - 1;
        min_int = -2^(word_len - 1);
    else
        max_int = 2^word_len - 1;
        min_int = 0;
    end
    x_q = round(double(x) * scale);
    x_q = max(min(x_q, max_int), min_int);
end

