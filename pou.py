# se agrega la librería time para calcular el tiempo de ejecución
from time import time
# se define la variable tinicial para medir el tiempo inicial
tinicial = time()
""" Función para calcular el valor de beta """
# Se define la función, fc es la resistencia del hormigón en [kgf/cm2]
def beta(fc):
# si el valor de fc es menor a 280[kgf/cm2], beta1 es igual a 0.85
    if fc < 280:
        beta1 = 0.85
# si el valor de fc está entre 280[kgf/cm2] y 560[kgf/cm2],
# el valor de beta1 toma un valor intermedio entre 0.85 a 0.65
    elif 560 >= fc >= 280:
        beta1 = 0.85-0.2/280*(fc-280)
# si el valor de fc supera los 560[kgf/cm2], el valor de beta1 es 0.65
    elif fc > 560:
        beta1 = 0.65
# la función devuelve el valor de beta1
    return beta1
""" Función para calcular et"""
# se define la función eT, h es la altura del perfil en [cm],
# dp es la distancia de recubrimiento de hormigón, desde un
# extremo del perfil al centro de la barra más próxima y c es la
# distancia........
def eT(h, dp, c):
# se calcula la ........
    et = 0.003*(h-dp-c)/c
# la función devuelve el valor de et
    return et
""" Función para calcular el valor de Ø """
# se define la función para el cálculo de phi
def phi(et):
# el valor inicial de phi es 0.65
    Phi = 0.65
# si el valor de et es menor a 0.002, phi se mantiene en 0.65
    if et < 0.002:
        Phi = 0.65
# si el valor de et está entre 0.002 y 0.005, éste varía linealmente entre 0.65 y 0.9
    elif 0.002 <= et <= 0.005:
        Phi = 0.65 + 0.25/0.003 * (et - 0.002)
# si el valor de et es superior a 0.005, el valor de phi es 0.9
    elif et > 0.005:
        Phi = 0.9
#la función devuelve el valor de phi
    return Phi
""" Función para calcular la excentricidad """
#se define la función para el cálculo de la excentricidad
def exc(pn, mn):
# para evitar ZeroDivisionError, se establece un valor infinito
# para valores pequeños de pn absoluto, en este caso menores a 0.0001
    if abs(pn) < 0.0001:
        e = str("infinita")
# en caso de que pn tome un valor un poco más lejano de 0, se calcula la excentricidad
# dividiendo mn por pn y redondeando la cifra a 3 decimales
    else:
        e = round(mn/pn, 3)
# la función devuelve el valor de e
    return e
""" Función para calcular las áreas en barras de acero """
# define función Acirc para el cálculo de áreas en barras, donde D es el diámetro en [cm]
def Acirc(D):
# el área del círculo es el cuadrado del diámetro por pi/4, redondeado a 3 decimales
    Ac = round(D**2*0.7854, 3)
# la función devuelve el valor de Ac
    return Ac
""" Función para crear lista de Areas de acero por nivel """
# se define la función Alist, que genera una lista de las áreas de acero por niveles
# Dsup es el diámetro de las barras del primer nivel en [cm], nsup es el número de
# barras en el nivel superior, Dlat es el diámetro de las barras laterales en [cm],
# nlat es el número niveles de dos barras laterales, Dinf es el diámetro de las barras
# inferiores en [cm], ninf es el número de barras en el nivel inferior
def Alist(Dsup, nsup, Dlat, nlat, Dinf, ninf):
# Asup llama a la función para calcular el área del círculo y
# multiplica por número de barras en el nivel superior, entrega
# el valor en [cm2]
    Asup = Acirc(Dsup)*nsup
# Alat llama a la función para calcular el área del círculo y
# multiplica por número de barras en niveles laterales, entrega
# el valor en [cm2]
    Alat = Acirc(Dlat)*2
# Ainf llama a la función para calcular el área del círculo y
# multiplica por número de barras en el nivel inferior, entrega
# el valor en [cm2]
    Ainf = Acirc(Dinf)*ninf
# A crea lista de áreas con el primer valor Asup
    A = [Asup]
# para cada nivel lateral se crea un área a continuación de Asup
    for i in range(nlat):
# agrega a la lista cada valor de área lateral
        A.append(Alat)
# finalmente agrega el nivel inferior a la lista
    A.append(Ainf)
# la función Alist devuelve valor de lista A
    return A
""" Función para calcular la suma de las areas"""
# define función que suma las áreas de la lista A
def aSum(A):
# el valor inicial de la sumatoria es cero
    sum = 0
# para cada valor de la lista a se acumula el total
    for i in A:
        sum += i
# la función devuelve el valor de de la suma de áreas en [cm2] sum
    return sum
""" Función para crear la lista de posiciones por niveles de areas """
# define función de posiciones del centro de cada nivel de barras en [cm]
def Ylist(h, dp, nlat):
# se crea la lista Y con un valor inicial que es la distantcia dp
    Y = [dp]
# se crea una posición para cada nivel de barras y se añaden a la lista Y
    for i in range(1, nlat+1):
        Y.append((h-Y[i-1]-dp)/(nlat+2-i)+Y[i-1])
# se crea condicional que aproxima la posición vertical de los niveles de barras en [cm]
        if Y[i]-int(Y[i]) > 0.5:
            Y[i] = int(Y[i])+1
        else:
            Y[i] = int(Y[i])
    Y.append(h-dp)
# la función devuelve la lista Y
    return Y
""" Función para crear la lista de valores de ei """
#......
def eiList(Y, c):
    ei = []
    for i in range(len(Y)):
        ei.append((c-Y[i])/c * 0.003)
    return ei
#.......
""" Función para crear la lista de valores de fs en función de ei """

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
""" Función para crear la lista de Ps """
def psList(fs, A):
    ps = []
    for i in range(len(A)):
        ps.append(fs[i]*A[i])
    return ps
""" Función para calcular la sumatoria Ps"""
def sumPs(fs, A):
    sumps = 0
    for i in range(len(A)):
        sumps = sumps + (fs[i] * A[i])
    return sumps
""" Función para calcular Pc """
def Pc(beta, fc, b, c):
    pc = 0.85*beta*fc*b*c
    return pc
"""  Función para calcular Pn """
def Pn(fs, A, pc):
    ps = sumPs(fs, A)
    pn = ps + pc
    return pn
"""  Función para calcular ØPn """
def phiPn(phi, pn):
    phiPn = phi*pn
    return phiPn
""" Función para calcular Ms """
def Ms(fs, A, h, Y):
    ps = psList(fs, A)
    ms=0
    for i in range(len(fs)):
        ms = ms + ps[i]*(0.5*h-Y[i])
    return ms
""" Función para calcular Mc """
def Mc(pc, h, c):
    mc = 0.5*pc*(h-0.85*c)
    return mc
""" Función para calcular Mn """
def Mn(ms, mc):
    mn = ms+mc
    return mn
""" Función para calcular phiMn """
def phiMn(phi, mn):
    phiMn = phi*mn
    return phiMn
""" Función para calcular el valor de c en línea neutra """
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
""" Función para calcular el valor de c en compresión pura """
def cComp(beta, h, dp):
    c = round(max(h/beta, 3*(h-dp)),3)
    return c
""" Función para calcular el valor máximo de c """
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
""" Función para calcular el elemento en compresion al 80% ó 100% """
def CompP(porc, beta, fc, h, b, dp, A, Y, ey, fy, Es):
    pc = Pc(1, fc, b, h)
    pn = round(porc*(pc + aSum(A)*fy)/100000, 2)
    Phi = 0.65
    if porc == 100:
        c = cComp(beta, h, dp)
        et = round(eT(h, dp, c), 5)
        phipn = round(pn*Phi*.8, 2)
        mn = 0
        phimn = 0
        e = 0
    elif porc == 80:
        c = cMax(h, b, dp, fc, A, Y, ey, fy, Es)
        et = round(eT(h, dp, c),5)
        phipn = round(pn*Phi, 2)
        mc = Mc(pc*.85, h, c)
        ei = eiList(Y, c)
        fs = fsList(ei, ey, fy, Es)
        ms = Ms(fs, A, h, Y)
        mn = round(Mn(ms, mc)/100000,2)
        phimn = phiMn(Phi, mn)
        e = round(mn/pn,3)
    Lres = [et, c, Phi, mn, pn, phimn, phipn, e]
    return Lres
""" Función para calcular el elemento a traccion pura """
def Trac(fy, A):
    c = 0
    pn = round(-aSum(A) * fy/1000, 2)
    Phi = 0.9
    phipn = round(pn * Phi, 2)
    mn = 0
    phimn = 0
    e = 0
    et = "--"
    Lres = [et, c, Phi, mn, pn, phimn, phipn, e]
    return Lres
""" Función para crear una lista resumen """
def Resumen(beta, fc, b, c, Y, h, dp, ey, fy, Es, A):
    pc = Pc(beta, fc, b, c)
    ei = eiList(Y, c)
    et = round(eT(h, dp, c), 5)
    Phi = phi(et)
    fs = fsList(ei, ey, fy, Es)
    pn = round(Pn(fs, A, pc)/1000, 2)
    phipn = round(phiPn(Phi, pn), 2)
    mc = round(Mc(pc, h, c)/100000, 2)
    ms = round(Ms(fs, A, h, Y)/100000, 2)
    mn = round(Mn(ms, mc), 2)
    phimn = round(phiMn(Phi, mn), 2)
    e = exc(pn, mn)
    Lres = [et, c, Phi, mn, pn, phimn, phipn, e]
    return Lres
""" Función para calcular elementos en flexion simple """
def FS(beta, fc, b, Y, h, dp, ey, fy, Es, A):
    c = round(cLN(h, b, dp, fc, A, Y, ey, fy, Es), 3)
    FS = Resumen(beta, fc, b, c, Y, h, dp, ey, fy, Es, A)
    return FS
""" Función para calcular elementos en condicion balanceada """
def cBal(beta, fc, b, Y, h, dp, ey, fy, Es, A):
    c = 0.6*(h-dp)
    CB = Resumen(beta, fc, b, c, Y, h, dp, ey, fy, Es, A)
    return CB
""" Función para calcular elementos con deformaciones del 0.5% """
def e5(beta, fc, b, Y, h, dp, ey, fy, Es, A):
    c = 0.375*(h-dp)
    E5 = Resumen(beta, fc, b, c, Y, h, dp, ey, fy, Es, A)
    return E5
""" Función para encontrar el rango de interacción """
def Rang(pu, mu, CP100m, CP80p, CP80m, CBp, CBm, E5p, E5m, FSp, FSm, TRp, TRm):
    e1 = CP80m/CP80p
    e2 = CBm/CBp
    e3 = E5m/E5p
    if pu == 0:
        pu = 0.001
    e = mu/pu
    if e <= e1:
        p2 = CP80p
        p1 = CP80p
        m2 = CP100m
        m1 = CP80m
        print("1")
    elif e1 < e < e2:
        p2 = CP80p
        p1 = CBp
        m2 = CP100m
        m1 = CBm
        print("2")
    elif e2 <= e < e3:
        p2 = CBp
        p1 = E5p
        m2 = CBm
        m1 = E5m
        print("3")
    elif pu > e3:
        p2 = E5p
        p1 = FSp
        m2 = E5m
        m1 = FSm
        print("4")
    elif pu<0:
        p2 = FSp
        p1 = TRp
        m2 = FSm
        m1 = TRm
        print("5")
    Rango=[p1, p2, m1, m2]
    print(Rango)
    return Rango
""" Función para calcular Vn"""
def Vn(fc, Nu, b, h, dp, AVS):
    d = h - dp
    Ag = b * h  #cm2
    if Nu == 0:
        Vc = (fc/10)**0.5*10/6*b*d
    elif Nu>0:
        factor = (1+(Nu/(14*Ag*10)))
        factorLIM = (1+0.29*Nu/(Ag*10))**0.5
        if factor > factorLIM:
            factor = factorLIM
        Vc = (fc / 10)**0.5*10/6*b*d*factor
    else:
        factor = 1+0.29*Nu/(Ag*10)
        Vc = (fc/10)**0.5*10/6*b*d*factor
        if Vc < 0:
            Vc = 0
    Vs = AVS*fy*d
    VsLIM = 4*(fc/10)**0.5*10/6*b*d
    if Vs > VsLIM:
        Vs = VsLIM
    Vn = Vc + Vs
    return Vn
""" Función para calcular la ecuacion de la recta """
m=1
error = 1
def ecRecta(mu, pu, p2, p1, m2, m1):
    e = mu/pu
    m = (p2-p1)/(m2-m1)
    phimn = round((p1-m*m1)/(1/e-m), 3)
    phipn = round(phimn/e, 3)
    FU = round(pu/phipn, 3)
    return print("ØMn = " + str(phimn) + "\n ØPn = " +
                 str(phipn) + "\n FU = " + str(FU))
""" Datos de entrada """
pu = 144
mu = 30
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
""" Cálculo de flexión compuesta"""
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
""" Cálculo de Corte """
Nu = 0
A_est = 0.79
Nramas = 4
s = 20
Av = A_est*Nramas
AVS = Av/s
phiV = 0.6
vn = Vn(fc, Nu, b, h, dp, AVS)
phiVn = vn*phiV
print(vn)
print(phiVn)
""" Cálculo de FU """
CP80p = CP80[6]
CP100m = CP100[5]
CP80m = CP80[5]
CBp = CB[6]
CBm = CB[5]
E5p = E5[6]
E5m = E5[5]
FSp = FS[6]
FSm = FS[5]
TRp = TR[6]
TRm = TR[5]
Pu_Mu = Rang(pu, mu, CP100m, CP80p, CP80m, CBp, CBm, E5p, E5m, FSp, FSm, TRp, TRm)
p1 = Pu_Mu[0]
p2 = Pu_Mu[1]
m1 = Pu_Mu[2]
m2 = Pu_Mu[3]
print(round(mu/pu, 3))
print(str(p1) + " \n " + str(p2) + " \n " + str(m1) + " \n " + str(m2))
ecRecta(mu, pu, p2, p1, m2, m1)
# Falta:
#Ordenar y comentar las funciones
#Indicar unidades en los comentarios
tfinal=time()
ttotal=tfinal-tinicial
print(ttotal)
