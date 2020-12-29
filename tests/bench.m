clear all; clc;

function vuelta(text1, text2)
    printf('vuelta: %i, round: %i\n', text1, text2)
end

loops = 10000;
% para matrices random, descomentar esta zona.
%largo = 33;
%A = rand(largo);
%b = rand(largo, 1);

t1 = zeros(loops, 1);
for k = 1:loops
    vuelta(k, 1)
    gauss
    t1(k, 1) = toc;
end

t2 = zeros(loops, 1);
for k = 1:loops
    vuelta(k, 2)
    tic
    res1 = rref(Ab);
    t2(k, 1) = toc;
end

t3 = zeros(loops, 1);
for k = 1:loops
    vuelta(k, 3)
    tic
    res2 = A^-1*b;
    t3(k, 1) = toc;
end

t4 = zeros(loops, 1);
for k = 1:loops
    vuelta(k, 4)
    tic
    res3 = A\b;
    t4(k, 1) = toc;
end

printf('\nloops: %i\nlargo: %i\n\n', loops, largo)
printf('Tiempo promedio del script: %f s\n', mean(t1))
printf('Tiempo promedio de rref() : %f s\n', mean(t2))
printf('Tiempo promedio de A^-1*b : %f s\n', mean(t3))
printf('Tiempo promedio de A\\b    : %f s\n', mean(t4))
printf('\n---- -- - -- ----\n')

beep
