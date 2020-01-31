function init(T)
global glob;
detail = 0;
%M = dlmread('func.csv',';');
%time = M(:,1);
%func = M(:,2);
% if T >= max(time)
%     func = func.*max(time)./T;
%     time = time.*T./max(time);
%     %time = [time; (linspace(time(end) + 1e-5, T, 500))'];
%     %func = [func; (func(end).*ones(500, 1))];
% else
%     k = find(time > T, 1, 'first');
%     time = time(1:k);
%     func = func(1:k);
% end

[~,func,time] = wavefun('sym3',5); 

%{
    Let suppose func is t**j/j! for some base t
    newfunc1 is func points in time=time*{2^-1, 2^-2, ..., 2^-(max(time)-1)}
    newfunc2 is func points in time=time*{2^1, 2^2, ..., 2^(max(time)-1)}
%}

for i=1:2^detail*max(time)-1
    newfunc1(:,i) = interp1(time,func,(time*2^(detail)-i),[],0);
end

for i=1:2^detail*max(time)-1
    newfunc2(:,i) = interp1(time,func,(time*2^(detail)+i),[],0);
end

%{
    Now basis is the concatenation of newfunc1 inversed, func and newfunc2
    So basis is now {time*(2^-(max(time)-1)), ... , time*(2^-1), func(1), func(2),
    func(end), time*(2^1), time*(2^2), ..., time*(2^(max(time)-1) }

    New x-axis for basis is { time/(2^(max(time)-1), ... , time*(2^(max(time)-1))
%}

basis = [fliplr(newfunc1) func newfunc2];

%{
    Now basis is normalized according to time basis.*max(time)./T
    time vector is also normalized
    
%}
basis = basis.*max(time)./T;
time = time.*T./max(time);
   

for i=1:size(basis,2)
    for j=1:size(basis,2)
        GG(i,j) = trapz(time,basis(:,i).*basis(:,j));
    end
end

%{
This QR decomposition might be related to page 15 where it states that
W1 is the subspace of functions satisfying f(0)=f'(0)=...=fp-1(0)=0
%}

basis(:,diag(GG) < 0.00000001) = [];
basis = qr1(basis,time);
N = size(basis,2); % basis size
glob.N = N;
glob.basis = basis;


for k=1:N
    dbasis(:,k) = interp1((time(1:end-1)+time(2:end))/2,diff(basis(:,k))./diff(time),time,'linear','extrap'); %basis deriv
    d2basis(:,k) = interp1((time(1:end-1)+time(2:end))/2,diff(dbasis(:,k))./diff(time),time,'linear','extrap'); %basis second deriv
end


for k=1:N
    one(k) = scalar_H(basis(:,k),ones(size(time)),time);
    two(k) = scalar_H(basis(:,k),time,time);
end

sum = 0;
for k = 1:N
    sum = sum + two(k).*basis(:,k);
end

tspan = time.';
ntspan = numel(tspan);

[ttspan, stspan] = meshgrid(tspan);

Rr = (min(stspan, ttspan)).^2.*(3*max(stspan, ttspan)-min(stspan, ttspan))/6;
glob.Rr = Rr;
%for tt = 1:ntspan
%    for k = 1:N
%        c(k, tt) = scalar_H(Rr(:, tt), basis(:, k), tspan);
%    end
%end
%sum = zeros(ntspan, ntspan);
%for tt = 1:ntspan
%for k = 1:N
%    sum(tt, :) = sum(tt, :) + c(k,tt).*basis(:, k).';
%end
%end
%glob.c = c;
glob.one = one;
glob.two = two;
glob.ntspan = ntspan;
glob.tspan = tspan;

end

