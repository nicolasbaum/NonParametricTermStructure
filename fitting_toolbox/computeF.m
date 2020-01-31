function [F t_all] = computeF (t,r,t_flag)
t_maturity = round(365.*t);
t_maturity = t;
L = length(t_maturity);

F = zeros(L, max(t_maturity)+1);

for i = 1:L
    F(i, t_maturity(i)) = 100;
    s = t_maturity(i);
    while s >= 0
        if t_flag == 1
            F(i, s+1) = F(i, s+1) + r(i)*100;
            s = s - 365;
        else
            F(i, s+1) = F(i, s+1) + r(i)*100/2;
            s = s - 183;
        end
        if t_flag == 1
            F(i, 1) = s/365*r(i)*100;
        else
            F(i, 1) = s/183*r(i)*100/2;
        end
    end
end

bool = sum(F,1) == 0;
F(:,bool) = [];
t_all = (0:max(t_maturity))/365;
t_all(bool) = [];
end