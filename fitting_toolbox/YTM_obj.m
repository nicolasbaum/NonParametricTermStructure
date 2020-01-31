%changed
function [res dres] = YTM_obj(r,P,F,t)
N = length(r);
for i=1:N
    d = exp(-t.*r(i));
    res(i) = F(i,:)*d'-P(i);
    dres(i,i) = F(i,:)*(-t.*d)';
end
