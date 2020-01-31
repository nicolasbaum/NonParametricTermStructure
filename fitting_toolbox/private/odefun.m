function dy = odefun(x, y, tspan, G)
%    dy(1) = y(2);
%    dy(2) = y(3);
%    dy(3) = y(4);
%    dy(4) = y(3);
global glob;

    Gx = interp1q(tspan, G, x);
    dy(1) = y(2);
    dy(2) = y(3);
    dy(3) = y(4);
    dy(4) = dy(2) - Gx;
end