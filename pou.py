
def b1(fc):
    if fc < 280:
        b1 = 0.85
    elif 550 >= fc >= 280:
        b1 = 0.85 - 0.05 / 70 * (fc - 280)
    else:
        b1 = 0.65
    return round(b1, 3)


def et(c, dp, eu, h):
    return round(eu * abs(h - dp - c) / c, 4)


def phi(eu, et, ey):
    if et < ey:
        phi = 0.65
    elif ey <= et <= (eu + ey):
        phi = 0.65 + 0.25 / eu * (et - ey)
    elif et > (eu + ey):
        phi = 0.9
    return round(phi, 3)


def aCir(d):
    return round(0.007854 * d ** 2, 3)


def aLst(dI, dL, dS, nI, nL, nS):
    aS = aCir(dS) * nS
    aL = aCir(dL) * nL
    aI = aCir(dI) * nI
    aLst = [aS]
    for i in range(nL):
        aLst.append(aL)
    aLst.append(aI)
    return aLst


def yLst(dp, h, nL):
    yLst = [dp]
    for i in range(1, nL + 1):
        yi = round((h - yLst[i - 1] - dp) / (nL + 2 - i) + yLst[i - 1], 0)
        yLst.append(int(yi))
    yLst.append(h - dp)
    return yLst


def eiLst(c, eu, yLst):
    if c < 0.01:
        c = 0.01
    eiLst = ((round(eu * (c - i) / c, 5) for i in yLst))
    return eiLst


def fsLst(eiLst, es, ey, fy):
    fsLst = []
    for i in eiLst:
        if i > ey:
            fs = fy
        elif -ey <= i <= ey:
            fs = es * i
        else:
            fs = -fy
        fsLst.append(round(fs, 2))
    return fsLst


def psLst(aLst, fsLst):
    psLst = []
    for i in range(len(fsLst)):
        psLst.append(fsLst[i] * aLst[i])
    return psLst


def ps(aLst, fsLst):
    psSum = round(sum(fsLst[i] * aLst[i] for i in range(len(fsLst))) / 1000, 2)
    return psSum


def pc(b, b1, c, fc):
    return round(0.85 * b1 * fc * b * c / 1000, 2)


def pn(pc, ps):
    return round(pc + ps, 2)


def phiPn(phi, pn):
    return round(phi * pn, 2)


def pnMax(aLst, b, fc, fy, h):
    return round((0.85 * fc * (h * b - sum(aLst)) + sum(aLst) * fy) / 1000, 2)


def phiPnMax(phi, pnMax):
    return round(phi * pnMax, 2)


def pnMin(aLst, fy):
    return round((-sum(aLst) * fy) / 1000, 2)


def phiPnMin(phi, pnMin):
    return round(phi * pnMin, 2)


def mc(c, pc, h):
    return round(pc / 2 * (h - 0.85 * c) / 100, 2)


def ms(fsLst, h, psLst, yLst):
    ms = 0
    for i in range(len(fsLst)):
        ms = ms + psLst[i] * (h / 2 - yLst[i])
    return round(ms / 100000, 2)


def mn(mc, ms):
    mn = mc + ms
    if mn < 0:
        mn = 0
    return round(mn, 2)


def phiMn(mn, phi):
    return round(phi * mn, 2)


def cuanMin(fc, fy):
    return round(max((fc * 0.1) ** 0.5 / (4 * fy), 14 / fy), 5)


def cuanMax(b1, eu, ey, fc, fy):
    return round(0.75 * 0.85 * b1 * eu / (eu + ey) * fc / fy, 5)


def cPnMax(b1, dp, h):
    return max(h / b1, 3 * (h - dp))


"""
fsLpr = fsLst(eiL, es, ey, fy * 1.25)
psLpr = psLst(aLst, fsLpr)
PsPr = ps(aLst, fsLpr)
Ppr = pn(Pc, PsPr)
MsPr = ms(fsLpr, h, psLpr, yLst)
Mpr = mn(Mc, MsPr)
"""


def cPn(aLst, b, b1, dp, es, eu, ey, fc, fy, h, pnB, yLst):
    c1 = 0
    c2 = cPnMax(b1, dp, h)
    pnB = round(pnB, 1)
    PnMax = pnMax(aLst, b, fc, fy, h)
    PnMin = pnMin(aLst, fy)
    PhiPn = pnB + 1
    i = 0
    if pnB > PnMin * 0.9:
        if pnB >= PnMax * 0.8 * 0.65:
            pnB = PnMax * 0.8 * 0.65
        while abs(pnB - PhiPn) > 0.1:
            c = round((c1 + c2) / 2, 3)
            eiL = eiLst(c, eu, yLst)
            fsL = fsLst(eiL, es, ey, fy)
            eT = et(c, dp, eu, h)
            Phi = phi(eu, eT, ey)
            Pc = pc(b, b1, c, fc)
            Ps = ps(aLst, fsL)
            PhiPn = pn(Pc, Ps) * Phi
            if PhiPn > pnB:
                c2 = c
            else:
                c1 = c
            i += 1
            if i == 20:
                break
            Mc = mc(c, Pc, h)
            psL = psLst(aLst, fsL)
            Ms = ms(fsL, h, psL, yLst)
            PhiMn = (Mc + Ms) * Phi
    else:
        PhiPn = PnMin * 0.9
        c = 0
        PhiMn = 0
    if pnB >= PnMax * 0.8 * 0.65:
        PhiMn = 0
    return round(c, 2), round(PhiMn, 2), round(PhiPn, 2)


def cFind(aLst, b, b1, dp, es, eu, ey, fc, fy, h, mu, pu, yLst):
    pu = round(abs(pu + 0.01) / (pu + 0.01) * 0.01 + pu, 1)
    mu = round(mu, 1)
    e = round(abs(mu) / pu, 3)
    if abs(mu) < 0.01:
        e = 0
    ex = e + 0.1
    PnMin = pnMin(aLst, fy)
    if e != 0 and pu > PnMin:
        i = 0
        if e > 0:
            c1 = cPnMax(b1, dp, h)
            c2 = 0
        elif e < 0:
            c1 = cPn(aLst, b, b1, dp, es, eu, ey, fc, fy, h, 0, yLst)[0]
            c2 = 0
        while abs(e - ex) > 0.001:
            c = round((c1 + c2) / 2, 2)
            eiL = eiLst(c, eu, yLst)
            fsL = fsLst(eiL, es, ey, fy)
            psL = psLst(aLst, fsL)
            Pc = pc(b, b1, c, fc)
            Ps = ps(aLst, fsL)
            Mc = mc(c, Pc, h)
            Ms = ms(fsL, h, psL, yLst)
            Mn = mn(Mc, Ms)
            Pn = pn(Pc, Ps)
            ex = round((abs(Mn)) / Pn, 3)
            if ex < e:
                c1 = c
            elif ex > e:
                c2 = c
            i += 1
            if i == 20:
                break
        eT = et(c, dp, eu, h)
        Phi = phi(eu, eT, ey)
        phipn = phiPn(Phi, Pn)
        phimn = phiMn(Mn, Phi)
    elif pu < PnMin:
        c = 0
        phipn = PnMin
        phimn = 0
        eT = et(c, dp, eu, h)
        Phi = 0.9
    elif e == 0:
        Pn = pnMax(aLst, b, fc, fy, h) * 0.8 * 0.65
        C = cPn(aLst, b, b1, dp, es, eu, ey, fc, fy, h, Pn, yLst)
        c = C[0]
        phipn = C[1]
        phimn = 0
        eT = et(c, dp, eu, h)
        Phi = 0.65
    return c, phimn, phipn, eT, Phi


def FU(pu, mu, cFound):
    if abs(mu) < 0.1:
        FU = abs(pu / (cFound[2]+0.01))
    else:
        FU = max(abs(pu / (cFound[2] + 0.01)), abs(mu / (cFound[1]+0.01)))
    return round(FU * 100, 1)


def aS(nS, dS, nL, dL, nI, dI):
    aS = (nS * aCir(dS) + 2 * nL * aCir(dL) + nI * aCir(dI))
    return aS


def aH(b, h, nS, dS, nL, dL, nI, dI):
    ah = b * h - aS(nS, dS, nL, dL, nI, dI)
    return ah


def cosL(b, h, nS, dS, nL, dL, nI, dI, cH, cS):
    costo = (aS(nS, dS, nL, dL, nI, dI) * cS +
             aH(b, h, nS, dS, nL, dL, nI, dI) * cH)/10000
    return round(costo, 0)


def cuantia(b, h, nS, dS, nL, dL, nI, dI):
    cuantia = aS(nS, dS, nL, dL, nI, dI) / aH(b, h, nS, dS, nL, dL, nI, dI)
    return round(cuantia, 5)


def rangBar(b, h, dp):
    hMin = int(1 + (b - 2 * dp) / 15)
    hMax = int(round(1 + (b - 2 * dp) / 10, 0))
    vMin = int(1 + (h - 2 * dp) / 15)
    vMax = int(round(1 + (h - 2 * dp) / 10, 0))
    return [hMin, hMax, vMin, vMax]


def diamList(fc, fy, b1, eu, ey, b, h, dp, dList):
    rang = rangBar(b, h, dp)
    cumin = cuanMin(fc, fy)
    cumax = cuanMax(b1, eu, ey, fc, fy)
    nBMin = rang[0] * 2 + (rang[2] - 2) * 2
    nBMax = rang[1] * 2 + (rang[3] - 2) * 2
    # entrada b y h en cm, salida diámetro en mm
    dMax = int(
        ((cumax * b * h * 127.324) / (nBMin * (1 + cumax))) ** 0.5)
    # coeficiente 127.324 = 4 * 100 / pi()
    dMin = int(round(
        ((cumin * b * h * 127.324) / (nBMax * (1 + cumin))) ** 0.5, 0))
    i = 0
    lista = []
    while dList[i] < dMax:
        if dList[i] >= dMin:
            lista.append(dList[i])
            if i < len(dList):
                i += 1
            else:
                break
        else:
            i += 1
    return lista


def supList(b, h, dp):
    rang = rangBar(b, h, dp)
    nS = []
    for i in range(rang[0], rang[1] + 1, 1):
        nS.append(i)
    return nS


def latList(b, h, dp):
    rang = rangBar(b, h, dp)
    nL = []
    for i in range(rang[2] - 2, rang[3] - 1, 1):
        nL.append(i)
    return nL


""" Cálculo de columna óptima"""


def optimusCol(b1, dp, es, eu, ey, fc, fy, muC, puC, dList, lList, cH, cS):
    minor = 9999999
    lista = ([i, a] for i in lList for a in lList if a == i)
    for i, a in lista:
        b = i
        h = a
        nS = supList(b, h, dp)
        nL = latList(b, h, dp)
        dS = diamList(fc, fy, b1, eu, ey, b, h, dp, dList)
        dL = dS
        listaND = ([j, k] for j in nS for k in nL if 10 <= (b - 2 * dp) / (j - 1) <= 15 and
                   10 <= (h - 2 * dp) / (k + 1) <= 15)
        for j, k in listaND:
            ylist = yLst(dp, h, k)
            listaDm = ([l, m] for l in dS for m in dL if m == l)
            for l, m in listaDm:
                alist = aLst(l, m, l, j, k, j)
                cFound = cFind(alist, b, b1, dp, es, eu, ey, fc, fy, h, muC, puC, ylist)
                fu = FU(puC, muC, cFound)
                cuan = cuantia(b, h, j, l, k, m, j, k)
                if fu < 100:
                    costo = cosL(b, h, j, l, k, m, j, l, cH, cS)
                    if costo < minor:
                        minor = costo
    e = cFound[1] / (cFound[2] + 0.001)
    optimo = [minor, h, b, j, k, l, m, fu, cuan, cFound[0], e, cFound[3], muV, puV]
    return optimo


"""cálculo de viga óptima"""


def optimusVig(b1, dp, es, eu, ey, fc, fy, muV, puV, dList, lList, cH, cS, lV):
    hMin = lV / 0.16
    minor = 9999999
    lista = ([i, a] for i in lList for a in lList if a >= i and a >= hMin)
    for i, a in lista:
        b = i
        h = a
        nS = supList(b, h, dp)
        nL = latList(b, h, dp)
        dS = diamList(fc, fy, b1, eu, ey, b, h, dp, dList)
        dL = dS
        listaND = ([j, k] for j in nS for k in nL if 10 <= (b - 2 * dp) / (j - 1) <= 15 and
                   10 <= (h - 2 * dp) / (k + 1) <= 15)
        for j, k in listaND:
            ylist = yLst(dp, h, k)
            listaDm = ([l, m] for l in dS for m in dL if l >= 16 and m <= l)
            for l, m in listaDm:
                alist = aLst(l, m, l, j, k, j)
                cFound = cFind(alist, b, b1, dp, es, eu, ey, fc, fy, h, muV, puV, ylist)
                fu = FU(puV, muV, cFound)
                cuan = cuantia(b, h, j, l, k, m, j, k)
                if fu < 100:
                    costo = cosL(b, h, j, l, k, m, j, l, cH, cS)
                    if costo < minor:
                        minor = costo
                        e = cFound[1] / (cFound[2] + 0.001)
                        optimo = [minor, h, b, j, k, l, m, fu, cuan, cFound[0], e, cFound[3], muV, puV]
    return optimo


def avs(dE, nRam, s):
    avs = aCir(dE) * nRam / s
    return avs


def vn(fc, nu, b, h, dp, avs):
    d = h - dp
    ag = b * h
    if nu == 0:
        vc = (fc / 10) ** 0.5 * 10 / 6 * b * d
    elif nu > 0:
        fact = (1 + (nu / (14 * ag * 10)))
        factLim = (1 + 0.29 * nu / (ag * 10)) ** 0.5
        if fact > factLim:
            fact = factLim
        vc = (fc / 10) ** 0.5 * 10 / 6 * b * d * fact
    else:
        fact = 1 + 0.29 * nu / (ag * 10)
        vc = (fc / 10) ** 0.5 * 10 / 6 * b * d * fact
        if vc < 0:
            vc = 0
    vs = avs * fy * d
    vsLim = 4 * (fc / 10) ** 0.5 * 10 / 6 * b * d
    if vs > vsLim:
        vs = vsLim
    return round((vc + vs) / 1000, 3)


def phiVn(vn, phiV):
    return round(vn * phiV, 2)


# def cortVig():
#     dE = 10
#     nr = 2
#     s = 15
#     avs = avs(dE, nr, s)
#     vn = vn(fc, nu, b, h, dp, avs)
#     phiV = 0.6
#     phiVn = phiVn(vn, phiV)
#     return 0
#
#
# def cortCol():
#     dE = 10
#     nr = 2
#     s = 15
#     avs = avs(dE, nr, s)
#     vn = vn(fc, nu, b, h, dp, avs)
#     phiV = 0.6
#     phiVn = phiVn(vn, phiV)
#     return 0


from time import time

muV = int(round(float(input('ingrese mu de viga (en Tf-m): ')), 0))
puV = int(round(float(input('ingrese pu de viga (en Tf): ')), 0))
VuV = int(round(float(input('ingrese vu de viga (en Tf): ')), 0))
muC0 = int(round(float(input('ingrese mu de columna (en Tf-m): ')), 0))
puC = int(round(float(input('ingrese pu de columna (en Tf): ')), 0))
VuC = int(round(float(input('ingrese vu de viga (en Tf): ')), 0))
#Tabla 9.3.1.1 ACI
lV = round(float(input('ingrese largo de viga (en m): ')), 1)
lC = round(float(input('ingrese columna de viga (en m): ')), 1)
dp = 5
es = 2100000
fc = 250
fy = 4200
cH = 60000
cS = 23550000
ey = 0.002
eu = 0.003
b1 = b1(fc)
lList = range(30, 110, 10)
dList = [12, 16, 18, 22, 25, 28, 32, 36]
tinicial = time()
optV = optimusVig(b1, dp, es, eu, ey, fc, fy, muV, puV, dList, lList, cH, cS, lV)
print(optV)
print("\nLos parámetros óptimos de diseño para la viga son: ")
print("\nCosto por metro lineal: $", str(int(optV[0])))
print("Altura del perfil:", str(optV[1]), "cm")
print("Ancho del perfil:", str(optV[2]), "cm")
print("Número de barras superiores e inferiores:", str(optV[3]))
print("Diámetro de barras superiores e inferiores:", str(optV[5]), "mm")
print("Número de pares de barras laterales:", str(optV[4]))
print("Diámetro de pares de barras laterales:", str(optV[6]), "mm")
print("Factor de utilización mayor:", str(optV[7]), "%")
print("Cuantía de acero:", str(optV[8]))
print("Profundidad de la línea neutra (c):", str(optV[9]), "cm")
print("Excentricidad:", str(round(optV[10]*100, 2)), "cm")
print("Deformación unitaria del acero:", str(optV[11]))
print("Momento último solicitado:", str(optV[12]), "Tf-m")
print("Carga última solicitada:", str(optV[13]), "Tf")
muC = max(int(round(muV / optV[7] * 100 * 1.2, 0)), muC0) # considerar excentricidad original de la columna
optC = optimusCol(b1, dp, es, eu, ey, fc, fy, muC, puC, dList, lList, cH, cS)
print("\n\n\nLos parámetros óptimos de diseño para la columna son:")
print("\nCosto por metro lineal:$", str(int(optC[0])))
print("Altura del perfil:", str(optC[1]), "cm")
print("Ancho del perfil:", str(optC[2]), "cm")
print("Número de barras superiores e inferiores:", str(optC[3]))
print("Diámetro de barras superiores e inferiores:", str(optC[5]), "mm")
print("Número de pares de barras laterales:", str(optC[4]))
print("Diámetro de pares de barras laterales:", str(optC[6]), "mm")
print("Factor de utilización mayor:", str(optC[7]), "%")
print("Cuantía de acero:", str(optC[8]))
print("Profundidad de la línea neutra (c):", str(optC[9]), "cm")
print("Excentricidad:", str(round(optC[10]*100, 2)), "cm")
print("Deformación unitaria del acero:", str(optC[11]))
print("Momento último solicitado:", str(optC[12]), "Tf-m")
print("Carga última solicitada:", str(optC[13]), "Tf")
tiempo = round(time() - tinicial, 1)
print("tiempo de ejecución =", str(tiempo), "segundos")
