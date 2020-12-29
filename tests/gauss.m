clear s_Zero; %clc;

function [nMatrix] = s_Zero(piv, M, R)
    aux = zeros(1, R + 1);
    if M(piv, piv) == 0
        for i = piv + 1:R
            if M(i, piv) != 0
                aux = M(piv, :);
                M(piv, :) = M(i, :);
                M(i, :) = aux;
                break;
            end
        end
    end
    nMatrix = M;
end % permuta

% para benchmark con matrices random, comentar esta seccion.
A = [0 0 -3 1; 0 1 4 0; 0 0 -1 0; -1 0 1 0];
b = [-3 6 -1 0]';

% para benchmark con matrices random, descomentar esta seccion.
%n = rank(A);
%b2 = zeros(n, 1);
%Ab = [A b];

tic

for i = 1:n
    for j = i:n
        if Ab(j, i) != 0 && Ab(j, i) != 1
            Ab(j, :) /= Ab(j, i);
        elseif i == j && Ab(j, i) == 0
            Ab = s_Zero(j, Ab, n);
            Ab(j, :) /= Ab(j, i);
        end
    end % normaliza cada fila
    if j > n break; end
    for j = i + 1:n
        if Ab(j, i) == 1
            Ab(j, :) -= Ab(i, :);
        end
    end % pivotea
end

b2(n) = Ab(n, n + 1);
for i = n - 1:-1:1
    b2(i) = Ab(i, n + 1) - Ab(i, 1:n)*b2;
end

res = [eye(n) b2];
%clear i j n;
