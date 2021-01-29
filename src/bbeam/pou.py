import matplotlib.pyplot as plt


def gen2array(a):
    b = []
    for i in a:
        b.append(i)
    return b


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


def aLstC(dEsq, dLat, nHor, nVer):
    a1 = round(aCir(dEsq) * 2 + nHor * aCir(dLat), 3)
    ai = round(aCir(dLat) * 2, 3)
    aLst = [a1]
    for i in range(nVer):
        aLst.append(ai)
    aLst.append(a1)
    return aLst


def yLstC(dp, h, nVer):
    yLst = [dp]
    for i in range(1, nVer + 1):
        yi = round((h - yLst[i - 1] - dp) / (nVer + 2 - i) + yLst[i - 1], 0)
        yLst.append(int(yi))
    yLst.append(h - dp)
    return yLst


def eiLst(c, eu, yLst):
    if c < 0.01:
        c = 0.01
    eiLst = []
    for i in yLst:
        eiLst.append(round(eu * (c - i) / c, 5))
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


def cPnMax(b1, dp, h):
    return max(h / b1, 3 * (h - dp))


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
        eiL = eiLst(c, eu, yLst)
        fsL = fsLst(eiL, es, ey, fy)
        eT = et(c, dp, eu, h)
    if pnB >= PnMax * 0.8 * 0.65:
        PhiMn = 0
        eiL = eiLst(c, eu, yLst)
        fsL = fsLst(eiL, es, ey, fy)
        eT = et(c, dp, eu, h)
    return round(c, 2), round(PhiMn, 2), round(PhiPn, 2), round(Mc + Ms, 2), eT


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
        FU = abs(pu / (cFound[2] + 0.01))
    else:
        FU = max(abs(pu / (cFound[2] + 0.01)), abs(mu / (cFound[1] + 0.01)))
    return round(FU * 100, 1)


def cosLC(b, h, dEsq, dLat, nHor, nVer, cH, cS):
    aS = aCir(dEsq) * 4 + aCir(dLat) * (2 * nHor + 2 * nVer)
    costo = (aS * cS + (b * h - aS) * cH) / 10000
    return round(costo, 0)


def cuantiaC(b, h,  dEsq, dLat, nHor, nVer):
    aS = aCir(dEsq) * 4 + aCir(dLat) * (2 * nHor + 2 * nVer)
    cuantia = aS/(b * h - aS)
    return round(cuantia, 5)


def supListC(b, h, dp):
    hMin = int(1 + (b - 2 * dp) / 15)
    hMax = int(round(1 + (b - 2 * dp) / 10, 0))
    nHor = []
    for i in range(hMin - 2, hMax - 1, 1):
        nHor.append(i)
    return nHor


def latListC(b, h, dp):
    vMin = int(1 + (h - 2 * dp) / 15)
    vMax = int(round(1 + (h - 2 * dp) / 10, 0))
    nVer = []
    for i in range(vMin - 2, vMax - 1, 1):
        nVer.append(i)
    return nVer


def latListV(h, dp):
    vMin = int(1 + (h - 2 * dp) / 15)
    vMax = int(round(1 + (h - 2 * dp) / 10, 0))
    nVer = []
    for i in range(vMin, vMax + 1, 1):
        nVer.append(i)
    return nVer


def supListV(b, dp):
    hMin = int(1 + (b - 2 * dp) / 15)
    hMax = int(round(1 + (b - 2 * dp) / 10, 0))
    nHor = []
    for i in range(hMin, hMax + 1, 1):
        nHor.append(i)
    return nHor


def cuminV(fc, fy):
    return round(max(0.8 / fy * (fc ** 0.5), 14 / fy), 4)


def cuantNiv(b, h, aS, dp):
    return round(aS / (b * (h - dp) - aS), 4)


def cuantiaV(b, h, aS):
    return round(aS / (b * h - aS), 4)


def yLstV(h, dp):
    blat = min(int((h - 3 * dp) / 25), int((h - 3 * dp) / 20) + 1)
    Y = [dp, 2 * dp]
    for i in range(blat):
        Y.append(round(Y[-1] + (h - 3 * dp) / (blat + 1), 0))
    if Y[-1] < h - dp:
        Y.append(h - dp)
    return Y


def aLstV(a1, a2, ai, a3, yv):
    A = [a1, a2]
    for i in range(len(yv) - 3):
        A.append(ai)
    A.append(a3)
    return A


def reverV(A):
    B = A
    B.reverse()
    return B


def dBarV(A, b, dp, dList):
    sup = supListV(b, dp)
    minlist = [abs(sup[-1] * i * i / 400 * 3.1416 - A) for i in dList]
    maxlist = [abs(sup[0] * i * i / 400 * 3.1416 - A) for i in dList]
    minerr = min(minlist)
    maxerr = min(maxlist)
    ind1 = minlist.index(minerr)
    ind2 = maxlist.index(maxerr)
    if ind1 > 1:
        ind1 = max(ind1 - 2, 0)
    if ind2 < 6:
        ind2 = min(ind2 + 2, 7)
    listad = []
    for i in range(ind1, ind2 + 1):
        listad.append(dList[i])
    return listad, sup


def diamBarV(A, b, fc, fy, dp, h, lista):
    min = 10 * A
    d1 = 0
    d2 = 0
    alist = ([j, k] for j in lista[0] for k in lista[0] if j>=k)
    cumin = cuminV(fc, fy)
    for i in lista[1]:
        if i - 2 * int(i / 2) > 0:
            n1 = int(i / 2)
            n2 = n1 + 1
        else:
            n1 = int(i / 2)
            n2 = int(i / 2)
        for j, k in alist:
            if n1 + n2 <= 2:
                j = k
            area = round(n1 * aCir(j) + n2 * aCir(k), 2)
            if abs(A - area):
                min = abs(A - area)
                d1 = j
                d2 = k
            diamlist = n1, d1, n2, d2, area
    return diamlist


def uLim(b1):
    return round(0.375 * b1 * (1 - 0.1875 * b1), 3)


def U(mu, fc, b, h, dp):
    d = h - dp
    # 0.00000765=0.9*0.85/100000
    muu = round(mu / (7.65e-06 * fc * b * d * d), 3)
    if muu > 0.5:
        muu = 0.5
    return muu


def delP(h, dp):
    return round(dp/(h - dp), 3)


def wP(ulim, u, delta):
    wp = round((u - ulim) / (1 - delta), 3)
    if u<ulim:
        wp=0
    return wp


def W(wp, u):
    if wp > 0:
        w = 0.375 + wp
    else:
        w = round(1 - (1 - 2 * u) ** 0.5, 3)
    return w


def areaV(mu, b, b1, h, fc, fy, dp, dList):
    muu = U(mu, fc, b, h, dp)
    ulim = uLim(b1)
    wp = wP(b1, muu, dp / (h - dp))
    w = W(wp, muu)
    a = round(w * 0.85 * fc * b * (h - dp) / fy, 2)
    return a


def areaLstV(mnn, mpp, b, b1, fc, fy, h, dp, dList, ai):
    aN = areaV(mnn, b, b1, h, fc, fy, dp, dList)/2
    aP = areaV(mpp, b, b1, h, fc, fy, dp, dList)
    dBarN = dBarV(aN, b, dp, dList)
    dBarP = dBarV(aP, b, dp, dList)
    dlistN = diamBarV(aN, b, fc, fy, dp, h, dBarN)
    dlistP = diamBarV(aP, b, fc, fy, dp, h, dBarP)
    Y = yLstV(h, dp)
    alist = aLstV(round(dlistN[4], 3), round(dlistN[4], 3), 1, round(dlistP[4], 3), Y)
    return dlistN, dlistP, Y, alist, aN, aP


def ylstRev(h, ylst):
    ylstrev = []
    for i in reversed(ylst):
        ylstrev.append(h - i)
    return ylstrev


def resumen(alst, c, b, h, eu, fy, fc, b1, es, ey, ylst):
    eiL = eiLst(c, eu, ylst)
    eiL = gen2array(eiL)
    eT = et(c, dp, eu, h)
    fsL = fsLst(eiL, es, ey, fy)
    fsLpr = fsLst(eiL, es, ey, fy * 1.25)
    psL = psLst(alst, fsL)
    psLpr = psLst(alst, fsLpr)
    Phi = phi(eu, eT, ey)
    Pc = pc(b, b1, c, fc)
    Ps = ps(alst, fsL)
    PsPr = ps(alst, fsLpr)
    Pn = pn(Pc, Ps)
    Ppr = pn(Pc, PsPr)
    PhiPn = phiPn(Phi, Pn)
    PnMax = pnMax(alst, b, fc, fy, h)
    if PhiPn > PnMax * 0.65 * 0.8:
        PhiPn = round(PnMax * 0.65 * 0.8, 2)
    CMax = cPn(alst, b, b1, dp, es, eu, ey, fc, fy, h, PnMax, ylst)
    if c > CMax[0]:
        c = CMax[0]
    Mc = mc(c, Pc, h)
    Ms = ms(fsL, h, psL, ylst)
    MsPr = ms(fsLpr, h, psLpr, ylst)
    Mn = mn(Mc, Ms)
    Mpr = mn(Mc, MsPr)
    if Mn == 0:
        PhiMn = 0.01
    else:
        PhiMn = phiMn(Mn, Phi)
    return PhiPn, Pn, Ppr, PhiMn, Mn, Mpr


def XYplotCurv(alst, b, h, dp, eu, fy, fc, b1, es, ey, ylst):
    PnMax = round((0.85 * fc * (h * b - sum(alst)) + sum(alst) * fy) / 1000, 2)
    PnMaxPr = round((0.85 * fc * (h * b - sum(alst)) + sum(alst) * fy * 1.25) / 1000, 2)
    phiPnMin = 0.9 * sum(alst) * -fy / 1000
    PnMin = sum(alst) * -fy / 1000
    PnMinPr = 1.25 * sum(alst) * -fy / 1000
    C = [0]
    X1 = [0]
    Y1 = [phiPnMin]
    X2 = [0]
    Y2 = [PnMin]
    X3 = [0]
    Y3 = [PnMinPr]
    CMax = cPn(alst, b, b1, dp, es, eu, ey, fc, fy, h, PnMax, ylst)
    for i in range(2, 41):
        C.append(i/40 * h)
    for c in C[1::]:
        res = resumen(alst, c, b, h, eu, fy, fc, b1, es, ey, ylst)
        X1.append(res[3])
        Y1.append(res[0])
        X2.append(res[4])
        Y2.append(res[1])
        X3.append(res[5])
        Y3.append(res[2])
    X1.append(0)
    X2.append(0)
    X3.append(0)
    Y1.append(Y1[-1])
    Y2.append(PnMax)
    Y3.append(PnMaxPr)
    plt.plot(X1, Y1, label='ØMn - ØPn', color='steelblue')
    plt.plot(X2, Y2, label='Mn - Pn', color='crimson')
    plt.plot(X3, Y3, label='Mpr - Ppr', color='forestgreen')
    plt.xlabel('Mn[tonf-m]')
    plt.ylabel('Pn[tonf]')
    plt.title("Curvas de interacción")
    plt.legend()
    plt.grid()
    plt.show()
    return 0


""" Cálculo de columna óptima"""


def optimusVig(mpp, mnn, es, eu, ey, b1, fc, fy, dp, dList, lList, ai, lo, cH, cS):
    min = 99999999
    cH = cH/10000
    cS = cS/10000
    lista = ([i, j] for i in lList if i >= lo / 16 for j in lList if i >= j and j >= 0.4 * i)
    cont = 100
    for i, j in lista:
        h = i
        b = j
        ylst = gen2array(yLstV(h, dp))
        ylstrev = ylstRev(h, ylst)
        aLst = areaLstV(mnn, mpp, b, b1, fc, fy, h, dp, dList, ai)
        aSLst = gen2array(aLst[3])
        alstrev = reverV(aSLst)
        aS = round(sum(aSLst), 2)
        aG = h * b - aS
        cuanT = round(aS / (aG - aS), 4)
        cumin = cuminV(fc, fy)
        cuan1 = round(aSLst[0] / ((b * (h - dp) - aSLst[0])), 4)
        cuan2 = round(2 * aSLst[-1] / ((b * (h - dp) - aSLst[-1])), 4)
        cond = False
        asdf = cPn(aSLst, b, b1, dp, es, eu, ey, fc, fy, h, 0, ylst)
        asdfrev = cPn(alstrev, b, b1, dp, es, eu, ey, fc, fy, h, 0, ylstrev)
        c = asdf[0]
        eT = et(c, dp, eu, h)
        if 0.025 >= cuan1 >= cumin and\
                0.0125 >= cuan2 >= cumin and eT >= 0.005\
                and asdf[1] >= mnn and asdfrev[1] >= mpp:
            cond = True
            cont = cont - 1
            costo = round(aS * cS + aG * cH, 0)
            if costo < min and cond != False:
                min = costo
                mpr1 = cPn(alstrev, b, b1, dp, es, eu, ey, fc, 1.25 * fy, h, 0, ylstrev)[3]
                mpr2 = cPn(aSLst, b, b1, dp, es, eu, ey, fc, 1.25 * fy, h, 0, ylst)[3]
                FU = round(max(mnn / asdf[1], mpp / asdfrev[1]) * 100, 1)
                listaT = min, h, b, mpr1, mpr2, aSLst, ylst, cuan1, cuan2*2, ylstrev, alstrev, FU, aLst[0], aLst[1]
    return listaT


def optimusCol(b1, dp, es, eu, ey, fc, fy, muC, puC, dList, lList, cH, cS):
    minor = 9999999
    lista = ([i, a] for i in lList for a in lList if a == i)
    for i, a in lista:
        b = i
        h = a
        nH = supListC(b, h, dp)
        nV = latListC(b, h, dp)
        listaND = ([j, k] for j in nH for k in nV if 10 <= (b - 2 * dp) / (j + 1) <= 15 and
                   10 <= (h - 2 * dp) / (k + 1) <= 15)
        for j, k in listaND:
            ylist = yLstC(dp, h, k)
            listaDm = ([l, m] for l in dList for m in dList if m <= l)
            for l, m in listaDm:
                alist = aLstC(l, m, j, k)
                cFound = cFind(alist, b, b1, dp, es, eu, ey, fc, fy, h, muC, puC, ylist)
                fu = FU(puC, muC, cFound)
                cuan = cuantiaC(b, h, l, m, j, k)
                if fu < 90 and 0.01 <= cuan <= 0.06:
                    costo = cosLC(b, h, l, m, j, k, cH, cS)
                    if costo < minor:
                        minor = costo
                        e = round(cFound[1] / (cFound[2] + 0.001), 3)
                        optimo = [minor, h, b,    j,    k,  l, m, fu, cuan, cFound[0], e, alist, ylist, fu]
    return optimo


def ramas(b, dp):
    dlibre = b - 2 * dp
    return int(dlibre / 35) + 2


def VueV(l, mpr1, mpr2, vug):
    vue = (mpr1 + mpr2) / l
    return vue + vug


def s(phi, av, fy, h, dp, vu):
    return phi * av * fy * (h- dp) / vu


from time import time


dp = 5
es = 2100000
fc = 250
fy = 4200
cH = 75000
cS = 7850000
ey = 0.002
eu = 0.003
b1 = 0.85
lList = range(30, 110, 10)
dList = [12, 16, 18, 22, 25, 28, 32, 36]
tinicial = time()
asdf = optimusVig(58.7, 30.29, es, eu, ey, b1, fc, fy, dp, dList, lList, 1, 700, cH, cS)
gen2array(asdf)
print(asdf)
optC = optimusCol(b1, dp, es, eu, ey, fc, fy, 82, 60, dList, lList, cH, cS)
print(optC)
tiempo = round(time() - tinicial, 4)
print("tiempo de ejecución =", str(tiempo), "segundos")
XYplotCurv(optC[11], optC[1], optC[2], dp, eu, fy, fc, b1, es, ey, optC[12])
XYplotCurv(asdf[5], asdf[2], asdf[1], dp, eu, fy, fc, b1, es, ey, asdf[6])
XYplotCurv(asdf[10], asdf[2], asdf[1], dp, eu, fy, fc, b1, es, ey, asdf[9])
