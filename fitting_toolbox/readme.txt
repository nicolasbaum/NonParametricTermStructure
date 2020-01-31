Hi Nicolas,

I found the code with the toolbox. Please, find it attached. I wasn't able to find the code with our applications, but hope the toolbox itself is enough. I'll construct an example how to run the code later this week. 

If you are familiar with Newton's method for finding minimum for functions, this is the right way to think about our method, try to ignore this overcomplicated math with Hilbert spaces :) 

Some comments on the algorithm (regarding the numerical method, it is better to read our paper on SSRN https://ssrn.com/abstract=2535244, rather than the published version), the main algorithm is on page 21.

The main function you need is fitting1. The main algorithm starts from line 80 in the code:
H = getH(g0, tspan, Tk)  computes \mathcal{H}_k (last formula on page 11). P.S. F(x) = x^2 for our case.
y = tilde_y(g0, P, F, Tk, tspan, w) - new_y - computes new y as in (29).
[ksi, T, Sigma] = find_eta(F, g0, Tk, tspan, w) - computes (37), (35), and (33).
M = Sigma + n*lambda*eye(n); - M as in (34)

[Q, R] = qr(T);
[Q1, R1] = qr(T, 0);
[sn, sm] = size(R);
Q2 = Q(1:end, sm+1:end);
c = Q2*(Q2'*M*Q2)^(-1)*Q2'*y;
d = R1^(-1)*Q1'*(y - M*c); - c and d as in (31) and (32).

after that we can compute a new approximation (as a new step in Newton's method):
g0 = get_new_func(c, d, ksi, tspan); using (30)

Next lines estimate GCV or GML and find its minimum to find new value of lambda.

getH and tilde_y are quite obvious. 

find_eta:

Compute eta as in (24). If you differentiate both parts, you get a differential equation, and Matlab can easily solve it. Alternatively, you can do integration for each t_i (cumtrapz in Matlab).

Then, we compute T and Sigma.

Let me know if you have any further questions and keep us updated with your progress.

Best wishes,

Vadim