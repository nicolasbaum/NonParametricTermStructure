%Este codigo supone que F=x^2 y donde hay que poner f pone f- porque es justamente lo que queremos
%Que f=f- que es cuando se va a haber llegado al resultado.

function result = tilde_y(g, P, F, times, tspan, w)
    global glob;
    [m, n] = size(F);
    int = glob.H;
    result = P;
    for i = 1:n     %for each bond
        resD = 0;
        resN = 0;
        for j = 1:m     %for each payment time
            %int(j) = trapz(tspan, g.*g.*(tspan <= times(j)).');
            resD = resD + exp(-int(j)).*F(j, i).*int(j);
            resN = resN + exp(-int(j)).*F(j, i);
        end;
        result(i) = result(i) - 2*resD - resN;
        
    end;
    %result = result.*w;
end