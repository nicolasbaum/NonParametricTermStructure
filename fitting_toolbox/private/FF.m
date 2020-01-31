%{
    FF: Hace la suma bono por bono de [NPV-Precio]^2
    Es una especie de error cuadratico medio de los NPV(r)
    
    Se usa para buscar un r que haga minimo FF
%}
function result = FF(r, tspan, F, P, times)
    [m, n] = size(F);
    result = 0;
    for i = 1:n
        sum = 0;
        for j = 1:m
            sum = sum + exp(-r.*times(j)).*F(j, i);
        end;
        result = result + (sum - P(i)).^2;
    end;
end

