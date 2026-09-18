function c = s_zeros(a)
% S_ZEROS  Zero struct with same field sizes as a, or zeros array of same size.
%   Uses 'like' to preserve gpuArray type when a is on GPU.
    if isstruct(a)
        c = a;
        for f = fieldnames(a)', c.(f{1}) = zeros(size(a.(f{1})), 'like', a.(f{1})); end
    else
        c = zeros(size(a), 'like', a);
    end
end
