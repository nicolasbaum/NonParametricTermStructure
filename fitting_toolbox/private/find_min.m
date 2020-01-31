function lambda = find_min(fun)
    span = linspace(-5, 10, 1000);
    for i = 1:numel(span)-1
        lam(i) = fminbnd(fun, span(i), span(i+1));
        flam(i) = fun(lam(i));
    end
    [~, i] = min(flam);
    lambda = lam(i);
end