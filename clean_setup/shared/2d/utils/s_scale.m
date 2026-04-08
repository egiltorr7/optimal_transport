function c = s_scale(s, a)
% S_SCALE  Fieldwise scalar multiplication for structs, or standard scaling for arrays.
    if isstruct(a)
        c = a;
        for f = fieldnames(a)', c.(f{1}) = s * a.(f{1}); end
    else
        c = s * a;
    end
end
