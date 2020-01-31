function [eta, T, Sigma] = find_eta(F, g0, times, tspan, w)

%{
    m: # of cashflows
    n: # of bonds

    g0: f0 (previous iteration f estimation )
%}
global glob;
N = glob.N;
basis = glob.basis;
one = glob.one;
two = glob.two;

ntspan = numel(tspan);
[m, n] = size(F);
eta = zeros(ntspan, n);
c = zeros(N, ntspan);
%{
    Second dimension of T is p
    In this case p=1
%}
T = zeros(n, 1);
Sigma = zeros(n, n);

%H_j(g0)

H = glob.H;
%c = glob.c;

%eta

solinit = bvpinit(tspan, @mat4init);

    
for i = 1:n
  G = zeros(numel(tspan), 1);
  for j = 1:m
    G = G + g0.*F(j, i).*exp(-H(j)).*(tspan <= times(j)).';
  end
  glob.G = G;
  %{
  bvp4c  Solve boundary value problems for ODEs by collocation.     
    SOL = bvp4c(ODEFUN,BCFUN,SOLINIT) integrates a system of ordinary
    differential equations of the form y' = f(x,y) on the interval [a,b],
    subject to general two-point boundary conditions of the form
    bc(y(a),y(b)) = 0. ODEFUN and BCFUN are function handles. For a scalar X
    and a column vector Y, ODEFUN(X,Y) must return a column vector representing
    f(x,y). For column vectors YA and YB, BCFUN(YA,YB) must return a column 
    vector representing bc(y(a),y(b)).  SOLINIT is a structure with fields  
        x -- ordered nodes of the initial mesh with 
             SOLINIT.x(1) = a, SOLINIT.x(end) = b
        y -- initial guess for the solution with SOLINIT.y(:,i)
             a guess for y(x(i)), the solution at the node SOLINIT.x(i)
 %}
  sol = bvp4c(@(x, y)odefun(x, y, tspan, G), @bcfun, solinit);
  for k = 1:N
    projG(k) = scalar_H(G, basis(:, k), tspan);
  end
  sum = 0;
  t1 = 0;
  t2 = 0;
  for k = 1:N
      %sum = sum + projG(k).*c(k, :);
      t1 = t1 + projG(k).*one(k);
      t2 = t2 + projG(k).*two(k);
  end
  %eta(:, i) = -2*sum.';
  eta(:, i) = sol.y(1, :) - sol.y(1, 1);
  T(i, 1) = -2*t1;
  %T(i, 2) = -2*t2;
end


for j = 1:n
    for k = 1:N
        %{
            projKsi: (k bonds,p order of diff)
        %}
        projKsi(j, k) = scalar_H(eta(:, j), basis(:, k), tspan);
    end
end

for i = 1:n
    G = zeros(numel(tspan), 1);
    for j = 1:m
        G = G + g0.*F(j, i).*exp(-H(j)).*(tspan <= times(j)).';
    end
    for k = 1:N
        projG(k) = scalar_H(G, basis(:, k), tspan);
    end

    for j = 1:n
        sum = 0;
        for k = 1:N
            sum = sum + projG(k).*projKsi(j, k);
        end
        Sigma(i, j) = -2*sum.';
    end
end

end

