function [d, r] = fitting(Tk, F, P, Eps, isGML, w)

%
% FITTING - evaluate discount curve.
%
% Input:
%       Tk: vector: [1, m] - payoffs.
%       F:  matrix: [m, n] - cashflow matrix.
%       P:  vector: [n, 1] - prices.
%       Eps: number - Newton method precision.
%       isGml: boolean - true - use GML method, false - GCV.
% Output:
%       d: vector: [1, ntspan] - discount curve points.
%       r: vector: [1, ntspan] - yield curve points.


global glob;
if numel(w) == 0
    [ytm, dur, mat] = findYTM(F', P, Tk');
    w = 1./dur;
end
[m, n] = size(F);
if (size(Tk, 1) ~= 1) || (size(Tk, 2) ~= m) || (size(P, 1) ~= n) || (size(P, 2) ~= 1)
    error('Wrong input');
end
init(Tk(end));

tspan = glob.tspan;
ntspan = glob.ntspan;
glob.F = F;
glob.P = P;
glob.Tk = Tk;
%init approx
r0 = fminsearch(@(r) FF(r, tspan, F, P, Tk), 0);
g0 = sqrt(r0).*ones(ntspan, 1);

lambda = 100; %default lambda

step = 0;

%main cycle
while true  
    step = step + 1;
    H = getH(g0, tspan, Tk);
    glob.H = H;
    y = tilde_y(g0, P, F, Tk, tspan, w);
    
    [ksi, T, Sigma] = find_eta(F, g0, Tk, tspan, w);
    
    M = Sigma + n*lambda*eye(n);
    [Q, R] = qr(T);
    [Q1, R1] = qr(T, 0);
    [sn, sm] = size(R);
    Q2 = Q(1:end, sm+1:end);
    c = Q2*(Q2'*M*Q2)^(-1)*Q2'*y;
    d = R1^(-1)*Q1'*(y - M*c);
    MM = @(lam) Sigma + n*lam*eye(n);
    aMat = @(lam) n*lam*Q2*(Q2.'*MM(lam)*Q2)^(-1)*Q2.';
    V = @(lam) 1./n.*norm(aMat(lam)*y).^2 ./ (trace(aMat(lam))./n).^2;
    eig_aMat = @(lam) eig(aMat(lam));
    GML = @(lam) y.'*aMat(lam)*y ./ prod((eig_aMat(lam) + (abs(eig_aMat(lam)) < 1e-8)).^(1/(n-1)));
    if isGML
        lam0 = fminbnd(GML, 0, 10000);
    else
        lam0 = fminbnd(V, 0, 10000);
    end
    lambda = lam0
    prev_g = g0;
    g0 = get_new_func(c, d, ksi, tspan);
    isDone = trapz(tspan, (abs(g0) - abs(prev_g)).^2) < Eps;
    isDone = isDone && (max((abs(g0) - abs(prev_g)).^2) < Eps);
    trapz(tspan, (abs(g0) - abs(prev_g)).^2)
    if isDone
        break;
    end
end

g0 = real(g0);

d(1) = 1;
r(1) = 0;
for j = 2:numel(tspan)
    d(j) = exp(-trapz(tspan(1:j), g0(1:j).^2));
    r(j) = trapz(tspan(1:j), g0(1:j).^2)./(tspan(j));
end

end

