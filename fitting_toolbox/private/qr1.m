function [q, r] = qr1(A,time)
[m, n] = size(A);

q = zeros(m, n);
r = zeros(n, n);

for k = n:-1:1%;1:n
    r(k,k) = sqrt(scalar_H(A(1:m, k),A(1:m,k),time));

    if r(k,k) == 0
        break;
    end

    q(1:m, k) = A(1:m, k) / r(k,k);

    for j = k-1:-1:1%k+1:n
        r(k, j) = scalar_H(q(1:m, k), A(1:m, j),time);
        A(1:m, j) = A(1:m, j) - r(k, j) * q(1:m, k);
    end
end
