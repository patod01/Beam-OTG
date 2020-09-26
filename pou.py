import numpy as np
d_p = 5
h = 60
d = h-d_p
b = 40
fc = 250
Es = 2100000
eu = 0.003
if fc < 280:
    b1 = 0.85
elif 560 > fc >= 280:
    b1 = 0.85-0.2/280*(fc-280)
elif fc > 560:
    beta1 = 0.65
fy = 4200
As_sup = 19.6
A_lat = 9.8
n_lat = 2
As_inf = 19.6
A = np.append([As_sup], np.append(np.array([A_lat for i in range(n_lat)]), [As_inf]), axis=0)
c = 3*d/8
Fsi = np.zeros(len(A))
y = np.array([5+i*round((h-2*d_p)/(len(A)-1), 3) for i in range(len(A))])
ei = np.array([round(eu*(c-y[i])/c, 6) for i in range(len(A))])
ey = fy/Es
Ps = 0
error=1
while abs(error) > 0.00001:
    ei = np.array([round(eu*(c-y[i])/c, 6) for i in range(len(y))])
    for i in range(len(ei)):
        if ei[i] > ey:
            Fsi[i] = fy
        elif -ey <= ei[i] <= ey:
            Fsi[i] = Es*ei[i]
        elif ei[i] < -ey:
            Fsi[i] = -fy
    for i in range(len(Fsi)):
        Ps = Fsi[i]*A[i]
    c2 = -Ps / (0.85 * b1 * fc * b)
    error=(c2-c)/c
    c=round((c2+c)/2,4)
print(c)
print(Fsi)
Pc = 0.85*b1*fc*b*c
Mc = Pc*(h/2-0.85*c/2)
Ms = 0
As = 0
for i in range(len(y)):
    Ms = Ms+Fsi[i]*A[i]*(h/2-y[i])
    As = As + A[i]
Mn = Ms + Mc
print(Ms/100000)
print(Mc/100000)











