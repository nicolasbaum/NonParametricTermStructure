function [d, r, d_span r_span, w] = fitting1(Tk, F, P, Eps, isGML, w, bid, ask, cur_curve)

%
% FITTING - evaluate discount curve.
%
% Input:
%       Tk: vector: [1, m] - payoffs.(t where kth instrument matures)
%       F:  matrix: [m, n] - cashflow matrix.
%       P:  vector: [n, 1] - prices.
%       Eps: number - Newton method precision.
%       isGml: boolean - true - use GML method, false - GCV.
%       w - 
%       bid -
%       ask -
%       cur_curve: vector: [1,m] - current zero coupon curve
% Output:
%       d: vector: [1, ntspan] - discount curve points.
%       r: vector: [1, ntspan] - yield curve points.


%{
Example run:
P=ones(17,1)*100.0
fitting1(T,F,P,0.001,1,[],nan,nan)
%}

global glob;
tic
Epsdur = 1;

%{
    ytm: YTM
    dur: Mccauley Duration
    mat: Maturity in years
%}

[ytm, dur, mat] = findYTM(F, P, Tk');

%P = P(dur > Epsdur);
%F = F(:, dur > Epsdur);
if numel(w) == 0
    if (sum(isnan(bid)) == 0) && false
        s = ask - bid;
        m = mean(s).*ones(size(s));
        w = ((1./(abs(m-s) + abs(s))));
        %w = 1./s.^2;
        % w = 1./s;
    else
        w = 1./(dur);
    end
    w = w ./ sum(w);
end

%w = ones(size(bid));


%w = w(dur > Epsdur);
F=F'
[m, n] = size(F);
wMat = repmat(w, 1, m);
F = F.*wMat';
P = P.*w;
%[ytm, dur, mat] = findYTM(F', P, Tk');

if (size(Tk, 1) ~= 1) || (size(Tk, 2) ~= m) || (size(P, 1) ~= n) || (size(P, 2) ~= 1)
    error('Wrong input');
end
init_v(max(Tk));

tspan = glob.tspan;
ntspan = glob.ntspan;
glob.F = F;
glob.P = P;
glob.Tk = Tk;

%init approx
%{
r0 como el valor que descontando los pagos da igual al precio (TIR)
Si r0<0.1% lo satura a 5%. Esto es un cambio muy grande
%}
%r0 seria la tasa para el plazo que representa la unidad de tiempo de la matriz de cashflows
%En este caso, como fabrico mi matriz de cashflows a intervalos diarios, es la tasa de descuento diaria
r0 = fminsearch(@(r) FF(r, tspan, F, P, Tk), 0);    %Ver comentarios de FF
if r0 < 1e-3
    r0 = 0.05;
end

%{
r0 como interpolacion lineal de la curva de zero-coupon
para t= valores de vector tspan
%}
%%%%%%%%%%%%%%%%%%%TEMPORARIAMENTE COMENTADO%%%%%%%%%%%%%%%%%%%
%r0 = interp1(Tk,cur_curve,tspan,'linear','extrap')

%{
g0=f0 va a ser aquella funcion tal que 
dado r0 => d(t) => d(t)=exp(-integral(F(f(t))dt)
Como indica ecuacion (3) del paper
%}
g0 = sqrt(diff(r0.*tspan)./diff(tspan))
%{
Asquerosamente repite el primer elemento de la derivada para que
tengan igual cantidad de elementos que la funcion buscada
%}
g0 = [g0(1), g0]';
lambda = 5; %default lambda
lambda = 0.6;
step = 0;

%{
Calcula d(t) y r(t) en funcion de f0
No entiendo cual es el sentido de dar la vuelta y recalcular esto.
Agrega error, aunque hace que sean consistentes r,d y f
%}
d(1) = 1;
r(1) = 0;
for j = 2:numel(tspan)
    d(j) = exp(-trapz(tspan(1:j), g0(1:j).^2));
    r(j) = trapz(tspan(1:j), g0(1:j).^2)./(tspan(j));
end


%main cycle
while true
    step = step + 1;
    for ww = 1:1
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
        prev_g = g0;
        if step > 1
            g0 = get_new_func(c, d, ksi, tspan);
        end
    end
    MM = @(lam) Sigma + n*lam*eye(n);
    aMat = @(lam) n*lam*Q2*inv(Q2.'*MM(lam)*Q2)*Q2.';
    Vhelper = @(lam) 1./n*norm(aMat(lam)*y).^2 ./ (trace(aMat(lam))./n).^2;
    eig_aMat = @(lam) eig(aMat(lam));
    help_mat = eye(sn-sm);
    help_mat = [zeros(1, sn-sm); help_mat];
    prodd = @(lam) prod(help_mat'*sort(abs(eig_aMat(lam))));
    GMLhelper = @(lam) y.'*aMat(lam)*y ./ prodd(lam).^(1/(n-1));
    GML = @(lam) real(GMLhelper(exp(lam)));
    V = @(lam)Vhelper(exp(lam));
    
    if isGML
        lam0 = find_min(GML);
    else
        lam0 = find_min(V);
    end
    lambda = exp(lam0)
    isDone = trapz(tspan, (abs(g0) - abs(prev_g)).^2) < Eps;
    isDone = isDone && (max((abs(g0) - abs(prev_g)).^2) < Eps);
    trapz(tspan, (abs(g0) - abs(prev_g)).^2)
    if step > 4
        break;
    end
end

d(1) = 1;
r(1) = 0;
for j = 2:numel(tspan)
    d(j) = exp(-trapz(tspan(1:j), g0(1:j).^2));
    r(j) = trapz(tspan(1:j), g0(1:j).^2)./(tspan(j));
end

g0 = real(g0);

d(1) = 1;
r(1) = 0;
for j = 2:numel(tspan)
    d(j) = exp(-trapz(tspan(1:j), g0(1:j).^2));
    r(j) = trapz(tspan(1:j), g0(1:j).^2)./(tspan(j));
end

for j = 1:numel(Tk)
    ss = find(tspan >= Tk(j), 1);
    if ss > 1
        d_span(j) = exp(-trapz(tspan(1:ss), g0(1:ss).^2));
        r_span(j) = trapz(tspan(1:ss),g0(1:ss).^2./tspan(ss));
    else
        d_span(j) = 1;
        %        r_span(j) = trapz(tspan(1:ss),g0(1:ss).^2./tspan(ss));
    end
end
toc
end

