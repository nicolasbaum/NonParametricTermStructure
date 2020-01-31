function [ytm dur mat, tt, rr]=drawYieldCurve(r)
    global glob;
    F = glob.F;
    P = glob.P;
    Tk = glob.Tk;
    tspan = glob.tspan;
    [ytm, dur, mat] = findYTM(F', P, Tk');
  plot(tspan(2:end), r(2:end)*100,'k-');%, dur, ytm*100, 'm*', mat, ytm*100, 'g*');
tt = tspan(2:end);
rr = r(2:end)*100;

% figure;
% hold on;
%  pp=plot(tspan(2:end), r(2:end)*100);
%  set(pp,'Color','black','LineWidth',2);
%  hold on;
% plot(mat,ytm*100,'mo');
% hold on;
end
