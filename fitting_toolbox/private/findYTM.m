function [ytm dur, mat] = findYTM(F,P,Tk)
N = length(P);
ytm = fsolve(@YTM_obj,ones(N,1)*0.05,optimset('TolFun',1e-10,'TolX',1e-10),P,F,Tk);
YTM_obj(ytm, P, F, Tk)
dur = sum(F.*exp(-ytm*Tk').*repmat(Tk',[size(F,1) 1]),2)./sum(F.*exp(-ytm*Tk'),2);  %durations

for i = 1:size(F, 1)
    mat(i, 1) = Tk(find(F(i,:),1,'last'));
end

function [res dres] = YTM_obj(x,P,F,Tk)
N = length(x);
for i=1:N
    d = exp(-Tk.*x(i));
    res(i) = F(i,:)*d-P(i);
    dres(i,i) = F(i,:)*(-Tk.*d);
end
fin = 0;