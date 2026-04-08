function c = s_add(a, b)
% S_ADD  Fieldwise addition for structs, or standard addition for arrays.
    if isstruct(a)
        c = a;
        for f = fieldnames(a)', c.(f{1}) = a.(f{1}) + b.(f{1}); end
    else
        c = a + b;
    end
end
