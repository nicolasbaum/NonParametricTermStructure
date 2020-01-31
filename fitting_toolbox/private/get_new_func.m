function f = get_new_func(c, d, ksi, tspan)
    ntspan = numel(tspan);
    f = ones(ntspan, 1)*d(1);% + tspan.'*d(2);
    n = numel(c);
    for i = 1:n
        f = f + c(i).*ksi(:, i);
    end
end

