function result = getH(g0, tspan, times)
    m = numel(times);
    result = zeros(m, 1);    
    dtspan = tspan(2) - tspan(1);
    for j = 1:m
        tt = tspan((tspan <= times(j)).');
        if numel(tt) > 1
            result(j) = trapz(tt, g0(1:numel(tt)).*g0(1:numel(tt)));
        else
            tt = [tt, tt + dtspan];
            result(j) = trapz(tt, g0(1:numel(tt)).*g0(1:numel(tt)));
        end
    end

end
    
