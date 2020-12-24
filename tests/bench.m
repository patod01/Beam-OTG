clear all; clc;

loops = 100

% para hacer el benchmark, descomentar esta zona.
%largo = 3000
%A = rand(largo);
%b = rand(largo,1);

t = zeros(loops, 1);
for k = 1:loops
    gauss
    t(k) = toc;
end
printf('Tiempo promedio del script: %f s\n', mean(t))

t = zeros(loops, 1);
for k = 1:loops
    tic
    res1 = rref(Ab);
    t(k) = toc;
end
printf('Tiempo promedio de rref() : %f s\n', mean(t))

t = zeros(loops, 1);
for k = 1:loops
    tic
    res2 = A^-1*b;
    t(k) = toc;
end
printf('Tiempo promedio de A^-1*b : %f s\n', mean(t))

t = zeros(loops, 1);
for k = 1:loops
    tic
    res3 = A\b;
    t(k) = toc;
end
printf('Tiempo promedio de A\\b    : %f s\n', mean(t))

beep
