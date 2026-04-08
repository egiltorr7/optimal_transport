function c = s_sub(a, b)
% S_SUB  Fieldwise subtraction for structs, or standard subtraction for arrays.
    if isstruct(a)
        c = a;
        for f = fieldnames(a)', c.(f{1}) = a.(f{1}) - b.(f{1}); end
    else
        c = a - b;
    end
end
