function result = find_z(c, x0, x1, t0, t1, zz)
    result = zz - ((t1*x0^2)/2 - (t0*x1^2)/2 - (t0*x0^2)/2 + (t1*x1^2)/2 + (x0^2*sin(2*c*t0 - 2*c*t1))/(4*c) + (x1^2*sin(2*c*t0 - 2*c*t1))/(4*c) - (x0*x1*sin(c*t0 - c*t1))/c + t0*x0*x1*cos(c*t0 - c*t1) - t1*x0*x1*cos(c*t0 - c*t1))/sin(c*(t0 - t1))^2;
end

