""" Valor de beta """

def beta(fc):

    if fc < 280:
        beta1 = 0.85
    elif 560 > fc >= 280:
        beta1 = 0.85-0.2/280*(fc-280)
    elif fc > 560:
        beta1 = 0.65
    return beta1

""" et"""

def eT(h, dp, c):
    et = 0.003*(h-dp-c)/c
    return et

""" Valor de Ø """

def phi(et):
    Phi = 0.65
    if et < 0.002:
        Phi = 0.65
    elif 0.002 <= et <= 0.005:
        Phi = 0.65 + 0.25/0.003 * (et - 0.002)
    elif et > 0.005:
        Phi = 0.9
    return Phi

""" Excentricidad """

def exc(pn, mn):

    if abs(pn) < 0.0001:
        e = str("infinita")
    else:
        e = round(mn/pn,3)
    return e

""" Area barras de acero """

def Acirc(D):

    Ac = round(D**2*0.7854, 3)
    return Ac

""" Lista de Areas de acero por nivel """

def Alist(Dsup, nsup, Dlat, nlat, Dinf, ninf):

    Asup = Acirc(Dsup)*nsup
    Alat = Acirc(Dlat)*2
    Ainf = Acirc(Dinf)*ninf
    A = [Asup]
    for i in range(nlat):
        A.append(Alat)
    A.append(Ainf)
    return A

""" Suma de areas"""

def aSum(A):
    sum = 0
    for i in A:
        sum += i
    return sum

""" Lista de posicion de niveles de areas """

def Ylist(h, dp, nlat):

    Y = [dp]
    for i in range(1, nlat+1):
        Y.append((h-Y[i-1]-dp)/(nlat+2-i)+Y[i-1])
        if Y[i]-int(Y[i]) > 0.5:
            Y[i] = int(Y[i])+1
        else:
            Y[i] = int(Y[i])
    Y.append(h-dp)
    return Y

""" Lista de valores de ei """

def eiList(Y, c):

    ei = []
    for i in range(len(Y)):
        ei.append((c-Y[i])/c * 0.003)
    return ei

""" Lista de valores de fs en funcion de ei """

def fsList(ei, ey, fy, Es):

    fs = []
    for i in range(len(ei)):
        if ei[i] > ey:
            fs.append(fy)
        elif -ey <= ei[i] <= ey:
            fs.append(Es * ei[i])
        elif ei[i] < -ey:
            fs.append(-fy)
    return fs

""" Lista de Ps """

def psList(fs, A):

    ps = []
    for i in range(len(A)):
        ps.append(fs[i]*A[i])
    return ps

""" Sumatoria Ps"""

def sumPs(fs, A):
    sumps = 0
    for i in range(len(A)):
        sumps = sumps + (fs[i] * A[i])
    return sumps

""" Calculo Pc """

def Pc(beta, fc, b, c):

    pc = 0.85*beta*fc*b*c
    return pc

"""  Calculo de Pn """

def Pn(fs, A, pc):
    ps = sumPs(fs,A)
    pn = ps + pc
    return pn

"""  Calculo de ØPn """

def phiPn(phi, pn):

    phiPn = phi*pn
    return phiPn

""" Calculo Ms  """

def Ms(fs, A, h, Y):

    ps = psList(fs, A)
    ms=0
    for i in range(len(fs)):
        ms = ms + ps[i]*(0.5*h-Y[i])
    return ms

""" Calculo Mc """

def Mc(pc, h, c):

    mc = 0.5*pc*(h-0.85*c)
    return mc

""" Calculo de Mn """

def Mn(ms, mc):

    mn = ms+mc
    return mn

""" Calculo de ØMn """

def phiMn(phi, mn):

    phiMn = phi*mn
    return phiMn

""" Calcular valor de c en linea neutra """

def cLN(h, b, dp, fc, A, Y, ey, fy, Es):

    c1 = dp
    c2 = h
    pn = 1
    while abs(pn) > 0.00001:
        c = (c1+c2)/2
        ei = eiList(Y, c)
        fs = fsList(ei, ey, fy, Es)
        ps = sumPs(fs, A)
        pc = Pc(beta, fc, b, c)
        pn = round(pc + ps, 4)
        if pn > 0:
            c2 = c
        else:
            c1 = c
    return round(c, 3)

""" Calcular c de compresión pura """

def cComp(beta, h, dp):

    c = round(max(h/beta, 3*(h-dp)),3)
    return c

""" Calcular valor de c maximo """

def cMax(h, b, dp, fc, A, Y, ey, fy, Es):
    pc = Pc((h * b - aSum(A)) / (h * b), fc, b, h)
    ps = aSum(A) * fy
    pn = 0.8 * (pc + ps)
    Pn = 0
    c1 = 1
    c2 = 3 * (h - dp)
    while abs(pn - Pn) > 0.001:
        c = (c1 + c2) / 2
        ei = eiList(Y, c)
        fs = fsList(ei, ey, fy, Es)
        ps = sumPs(fs, A)
        pc = Pc(beta, fc, b, c)
        Pn = round(pc + ps, 4)
        if Pn - pn > 0.001:
            c2 = c
        else:
            c1 = c
    return round(c, 3)

""" Compresion 80 ó 100 """

def CompP(porc, beta, fc, h, b, dp, A, Y, ey, fy, Es):

    pc = Pc(1, fc, b, h)
    pn = round(porc*(pc + aSum(A)*fy)/100000, 2)
    if porc == 100:
        c = cComp(beta, h, dp)
    elif porc == 80:
        c = cMax(h, b, dp, fc, A, Y, ey, fy, Es)
    Phi = 0.65
    if porc == 80:
        phipn = round(pn*Phi, 2)
    else:
        phipn = round(pn*Phi*.8, 2)
    mn = 0
    phimn = 0
    e = 0
    Lres = [c, Phi, mn, pn, phimn, phipn, e]
    return Lres

""" Traccion pura """

def Trac(fy, A):

    c = 0
    pn = round(-aSum(A) * fy/1000, 2)
    Phi = 0.9
    phipn = round(pn * Phi, 2)
    mn = 0
    phimn = 0
    e = 0
    Lres = [c, Phi, mn, pn, phimn, phipn, e]
    return Lres

""" Lista resumen """

def Resumen(beta, fc, b, c, Y, h, dp, ey, fy, Es, A):

    pc = Pc(beta, fc, b, c)
    ei = eiList(Y, c)
    et = eT(h, dp, c)
    Phi = phi(et)
    fs = fsList(ei, ey, fy, Es)
    pn = round(Pn(fs, A, pc)/1000, 2)
    phipn = round(phiPn(Phi, pn), 2)
    mc = round(Mc(pc, h, c)/100000, 2)
    ms = round(Ms(fs, A, h, Y)/100000, 2)
    mn = round(Mn(ms, mc), 2)
    phimn = round(phiMn(Phi, mn), 2)
    e = exc(pn, mn)
    Lres = [c, Phi, mn, pn, phimn, phipn, e]
    return Lres

""" Calculo de flexion simple (L.N.) """

def FS(beta, fc, b, Y, h, dp, ey, fy, Es, A):

    c = round(cLN(h, b, dp, fc, A, Y, ey, fy, Es), 3)
    FS = Resumen(beta, fc, b, c, Y, h, dp, ey, fy, Es, A)
    return FS

""" Condicion balanceada """

def cBal(beta, fc, b, Y, h, dp, ey, fy, Es, A):
    c = 0.6*(h-dp)
    CB = Resumen(beta, fc, b, c, Y, h, dp, ey, fy, Es, A)
    return CB

""" Ɛ = 0.005 """

def e5(beta, fc, b, Y, h, dp, ey, fy, Es, A):
    c = 0.375*(h-dp)
    E5 = Resumen(beta, fc, b, c, Y, h, dp, ey, fy, Es, A)
    return E5

""" Datos de entrada """

fc = 250
fy = 4200
Es = 2100000
ey = fy/Es
h = 60
b = 40
dp = 5
beta = beta(fc)
Dsup = 2.5
nsup = 4
Dlat = 2.5
nlat = 2
Dinf = 2.5
ninf = 4
A = Alist(Dsup, nsup, Dlat, nlat, Dinf, ninf)
Y = Ylist(h, dp, nlat)
FS = FS(beta, fc, b, Y, h, dp, ey, fy, Es, A)
E5 = e5(beta, fc, b, Y, h, dp, ey, fy, Es, A)
CB = cBal(beta, fc, b, Y, h, dp, ey, fy, Es, A)
TR = Trac(fy, A)
CP80 = CompP(80, beta, fc, h, b, dp, A, Y, ey, fy, Es)
CP100 = CompP(100, beta, fc, h, b, dp, A, Y, ey, fy, Es)
print(CP100)
print(CP80)
print(CB)
print(E5)
print(FS)
print(TR)
et = []
Phi = []
for i in range(1, 61):
    et.append(round(eT(60, 5, i), 5))
    Phi.append(round(phi(et[i-1]), 3))

""" FU """

def FU(Pu, Mu, P1, P2, M1, M2):

    # ecuacion 1
    phiPn = (Pu/Mu)*phiMn
    # Ecuacion 2
    phiPn = P2 + (P2-P1)/(M2-M1)*(phiMn-M2)


""" Falta :"""



""" Cuantia mínima y maxima """



""" Relacion FU para encontrar combinacion optima entre Mu y Pu dado """



""" Corte """

pu = 144
mu = 30
p2 = 162.27
p1 = 387.12
m2 = 53.07
m1 = 0


m=1
error = 1
def pfijo(error,m,pu,mu,p2,p1,m2,m1):
    while abs(error) > 0.0000001:
        b = (mu / pu) * (p1 + ((p2 - p1) / (m2 - m1)) * (m - m1))
        error = (b - m) / b
        m = b
    return round(m,3)
mn = pfijo(error,m,pu,mu,p2,p1,m2,m1)
pn = round((pu/mu)*mn,3)
FU = round(pu*100/pn,3)
print(mn)
print(pn)
print(str(FU)+"%")
