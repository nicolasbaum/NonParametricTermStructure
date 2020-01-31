function res = scalar_H(a,b,time)
    res = trapz(time,a.*b);