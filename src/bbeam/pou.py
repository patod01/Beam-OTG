import matplotlib.pyplot as plt

def b1(fc):
    if 550 >= fc >= 280:
        b1 = round(0.85 - 0.05 / 70 * (fc - 280), 2)
    else: b1 = 0.85 if fc < 280 else 0.65
    return b1

def phi(eu, et, ey):
    if ey <= et <= (eu + ey):
        phi = round(0.65 + 0.25 / eu * (et - ey), 2)
    else: phi = 0.65 if et < ey else 0.9
    return phi

def aCir(d): return round(0.007854*d**2, 3)

def aLstC(dEsq, dLat, nHor, nVer):
    a = round(aCir(dEsq)*2+nHor*aCir(dLat), 3)
    return [a]+[round(aCir(dLat)*2,3) for i in range(nVer)]+[a]

def yLstC(dp, h, nVer):
    yLst = [dp]
    for i in range(1, nVer + 1):
        yi = round((h-yLst[i-1]-dp)/(nVer+2-i)+yLst[i-1], 0)
        yLst.append(int(yi))
    return yLst + [h-dp]

def pmC(aLst, b, b1, c, es, eu, ey, fc, fy, h, yLst):
    eiLst = [round(eu*(c-i)/c, 5) for i in yLst]
    fsLst = [fy*abs(i)/i if abs(i)>ey else es*i for i in eiLst]
    psLst = [fsLst[i] * aLst[i] for i in range(len(aLst))]
    Pc, Ps = 0.85*b1*fc*b*c, sum(psLst)
    Mc = Pc/2*(h-0.85*c)
    Ms = sum((psLst[i]*(h/2-yLst[i]) for i in range(len(aLst))))
    return round((Pc + Ps)/1000, 2), round((Mc + Ms)/100000, 2)

def cPn(aLst, b, b1, dp, es, eu, ey, fc, fy, h, pnB, yLst):
    c1, c2 = 0, max(h/b1, 3*(h-dp))
    PnMax = round((0.85*fc*(h*b-sum(aLst))+sum(aLst)*fy)/1000, 2)
    PhiPnMax = PnMax*0.8*0.65
    PnMin, PhiPn, i = round((-sum(aLst)*fy)/1000, 2), pnB+1, 0
    if pnB > PnMin * 0.9:
        pnB = PhiPnMax if pnB >= PhiPnMax else pnB
        while abs(pnB-PhiPn) > 0.1 and i<15:
            c, i = round((c1+c2)/2, 3), i+1
            PMC = pmC(aLst, b, b1, c, es, eu, ey, fc, fy, h, yLst)
            eT = round(eu * abs(h - dp - c) / c, 4)
            Phi = phi(eu, eT, ey)
            PhiPn, PhiMn = (PMC[0]) * Phi, (PMC[1]) * Phi
            c2 = c if PhiPn > pnB else c2
            c1 = c if PhiPn < pnB else c1
    else: c, PhiPn, PhiMn = 0, PnMin * 0.9, 0
    return round(c, 2), round(PhiMn, 1), round(PhiPn, 1)

def cFind(aLst, b, b1, dp, es, eu, ey, fc, fy, h, mu, pu, yLst):
    pu = round(abs(pu + 0.01) / (pu + 0.01) * 0.01 + pu, 1)
    mu = round(mu, 1)
    e = round(abs(mu) / pu, 3)
    if abs(mu) < 0.1: e = 0
    PnMin = round((-sum(aLst) * fy) / 1000, 1)
    if e != 0 and pu > PnMin:
        i, c2, ex = 0, 0, e+0.001
        if e > 0:  c1 = max(h / b1, 3 * (h - dp))
        elif e < 0:
            c1 = cPn(aLst, b, b1, dp, es, eu, ey, fc, fy, h, 0, yLst)[0]
        while abs(e - ex) > 0.001 and i<20:
            c, i = round((c1 + c2) / 2, 2), i+1
            PMC = pmC(aLst, b, b1, c, es, eu, ey, fc, fy, h, yLst)
            ex = round((abs(PMC[1])) / (PMC[0]), 3)
            c1 = c if ex < e else c1
            c2 = c if ex > e else c2
        eT = round(eu*abs(h-dp-c)/c, 4)
        Phi = phi(eu, eT, ey)
        phipn, phimn = PMC[1]*Phi, PMC[0]*Phi
    elif pu < PnMin:
        c, phipn, phimn = 0, PnMin, 0
        eT = round(eu * abs(h - dp - c) / c, 4)
        Phi = phi(eu, eT, ey)
    elif e == 0:
        Pn = round((0.85*fc*(h*b-sum(aLst))+sum(aLst)*fy)/1000, 2)
        C = cPn(aLst, b, b1, dp, es, eu, ey, fc, fy, h, Pn, yLst)
        eT = round(eu*abs(h-dp-c)/c, 4)
        Phi = phi(eu, eT, ey)
        c, phipn, phimn = C[0], C[2]*Phi, 0
    return c, round(phimn, 1), round(phipn, 1)

def resumen(aLst, c, b, dp, h, eu, fy, fc, b1, es, ey, yLst):
    PnMax = round((0.85*fc*(h*b-sum(aLst))+sum(aLst)*fy)/1000, 2)
    CMax = cPn(aLst, b, b1, dp, es, eu, ey, fc, fy, h, PnMax, yLst)[0]
    PMC = pmC(aLst, b, b1, c, es, eu, ey, fc, fy, h, yLst)
    eT = round(eu * abs(h - dp - c) / c, 4)
    Phi = phi(eu, eT, ey)
    PMCpr = pmC(aLst, b, b1, c, es, eu, ey, fc, fy*1.25, h, yLst)
    return PMC[0]*Phi, PMC[0], PMCpr[0], PMC[1]*Phi, PMC[1], PMCpr[1]

def FU(pu, mu, pn, mn):
    if abs(mu) < 0.1: FU = abs(pu/(pn+0.01))
    else: FU = max(abs(pu/(pn+0.01)), abs(mu/(mn+0.01)))
    return round(FU * 100, 1)

def optimusCol(b1, dp, es, eu, ey, fc, fy, muC, puC, dList, lList, cH, cS):
    minor = 9999999
    lista = ([b, h] for b in lList for h in lList if b == h)
    for b, h in lista:
        nH = [i for i in range(int((b - 2 * dp) / 15) - 1, int(round((b - 2 * dp) / 10, 0)), 1)]
        nV = [i for i in range(int((h - 2 * dp) / 15) - 1, int(round((h - 2 * dp) / 10, 0)), 1)]
        listaND = ([j, k] for j in nH for k in nV if 10 <= (b - 2 * dp) / (j + 1) <= 15 and
                   10 <= (h - 2 * dp) / (k + 1) <= 15)
        for j, k in listaND:
            ylist = yLstC(dp, h, k)
            listaDm = ([l, m] for l in dList for m in dList if m <= l)
            for l, m in listaDm:
                alist = aLstC(l, m, j, k)
                cF = cFind(alist, b, b1, dp, es, eu, ey, fc, fy, h, muC, puC, ylist)
                fu = FU(puC, muC, cF[1], cF[2])
                aS = aCir(l) * 4 + aCir(m) * (2 * j + 2 * k)
                cuan = round(aS / (b * h - aS), 5)
                if fu < 90 and 0.01 <= cuan <= 0.06:
                    costo = round((aS*cS+(b*h-aS)*cH)/10000, 0)
                    if costo < minor:
                        minor, e = costo, round(cF[1] / (cF[2] + 0.001), 3)
                        optimo = [minor, h, b,    j,    k,  l, m, fu, cuan, cF[0], e, alist, ylist]
    return optimo

def yLstV(h, dp):
    blat = min(int((h-3*dp)/25), int((h-3*dp)/20)+1)
    Y = [dp, 2*dp]
    for i in range(blat):
        Y.append(round(Y[-1]+(h-3*dp)/(blat+1), 0))
    return Y + [h - dp] if Y[-1] < (h-dp) else []

def diamBarV(A, b, fc, fy, dp, h, dList):
    sup = [i for i in range(int(1 + (b - 2 * dp) / 15), int(round(1 + (b - 2 * dp) / 10, 0)) + 1, 1)]
    minlist = [abs(sup[-1] * i * i / 400 * 3.1416 - A) for i in dList]
    maxlist = [abs(sup[0] * i * i / 400 * 3.1416 - A) for i in dList]
    minerr, maxerr = min(minlist), min(maxlist)
    ind1, ind2 = minlist.index(minerr), maxlist.index(maxerr)
    ind1, ind2 = max(ind1 - 2, 0) if ind1 > 1 else ind1, min(ind2 + 2, 7) if ind2 < 6 else ind2
    listad = [dList[i] for i in range(ind1, ind2 + 1)]
    minim, d1, d2 = 10*A, 0, 0
    alist = ([j, k] for j in listad for k in listad if j>=k)
    cumin = round(max(0.8/fy*(fc**0.5), 14/fy), 4)
    for i in sup:
        n1, n2 = int(i/2), int(i/2)+1 if i-2*int(i/2) > 0 else int(i/2)
        for j, k in alist:
            j = k if n1+n2 <= 2 else j
            area = round(n1*aCir(j)+n2*aCir(k), 2)
            if abs(A-area)<minim:
                minim, d1, d2 = abs(A-area), j, k
    return n1, d1, n2, d2, area

def areaV(mu, b, b1, h, fc, fy, dp):
    muu = round(mu/(0.9*0.85/100000*fc*b*(h-dp)**2), 3)
    muu = 0.5 if muu > 0.5 else muu
    ulim = round(0.375 * b1 * (1 - 0.1875 * b1), 3)
    wp = 0 if muu < ulim else round((muu-ulim)/(1-dp/(h-dp)), 3)
    w = 0.375+wp if wp > 0 else round(1-(1-2*muu)**0.5, 3)
    return round(w*0.85*fc*b*(h-dp)/fy, 2)

def optimusVig(mpp, mnn, es, eu, ey, b1, fc, fy, dp, dList, lList, ai, lo, cH, cS):
    minim = 99999999
    lista = ([i, j] for i in lList if i >= lo/16 for j in lList if i >= j and j >= 0.4*i)
    for i, j in lista:
        h, b = i, j
        ylst = list(yLstV(h, dp))
        ylstrev = [(h-i) for i in reversed(ylst)]
        aN = areaV(mnn, b, b1, h, fc, fy, dp) / 2
        aP = areaV(mpp, b, b1, h, fc, fy, dp)
        alN = diamBarV(aN, b, fc, fy, dp, h, dList)
        alP = diamBarV(aP, b, fc, fy, dp, h, dList)
        #1 en la tercera posición corresponde a dos barras de diámetro 8mm
        aSLst = [round(alN[4], 3) for i in range(2)]+\
                [1 for i in range(len(ylst) - 3)] + [round(alP[4], 3)]
        alstrev = reversed(aSLst)
        cuanT = round(sum(aSLst)/(h*b-sum(aSLst)), 4)
        cumin = round(max(0.8/fy*(fc**0.5), 14/fy), 4)
        cuan1 = round(aSLst[0]/((b*(h-dp)-aSLst[0])), 4)
        cuan2 = round(2*aSLst[-1]/((b * (h - dp) - aSLst[-1])), 4)
        cpn = cPn(aSLst, b, b1, dp, es, eu, ey, fc, fy, h, 0, ylst)
        cpnrev = cPn(alstrev, b, b1, dp, es, eu, ey, fc, fy, h, 0, ylstrev)
        c, cond = cpn[0], False
        eT = round(eu*abs(h-dp-c)/c, 4)
        if 0.025 >= cuan1 >= cumin and 0.0125 >= cuan2 >= cumin and eT >= 0.005\
                and cpn[1] >= mnn and cpnrev[1] >= mpp:
            cond, costo = True, round((sum(aSLst) * cS + (h*b-sum(aSLst)) * cH)/10000, 0)
            if costo < minim and cond != False:
                minim = costo
                FU = round(max(mnn/cpn[1], mpp/cpnrev[1]) * 100, 1)
                listaT = minim, h, b, aSLst, ylst, cuan1, cuan2*2, ylstrev, alstrev, FU, alN, alP
    return listaT

def XYplotCurv(alst, b, h, dp, eu, fy, fc, b1, es, ey, ylst):
    PnMax = round((0.85*fc*(h*b-sum(alst))+sum(alst)*fy)/1000, 2)
    PnMaxPr = round(PnMax+sum(alst)*fy*0.25/1000, 2)
    PnMin, phiPnMin, PnMinPr = sum(alst)*-fy/1000, 0.9*sum(alst)*-fy/1000, 1.25*sum(alst)*-fy/1000
    C = [0]+[i/40*h for i in range(2, 41)]
    X1, X2, X3, Y1, Y2, Y3 = [0], [0], [0], [phiPnMin], [PnMin], [PnMinPr]
    for c in C[1::]:
        res = resumen(alst, c, b, dp, h, eu, fy, fc, b1, es, ey, ylst)
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

def espc(xlist, esp):
    i=0
    espacio=0
    while xlist[i+1]-5<=esp and len(xlist)>i:
        i+=1
        espacio = xlist[i]-5
    return espacio

def ramList(xlist, esp):
    nram = int((xlist[-1] - xlist[0] - 0.01) / esp) + 2
    if len(xlist) % 2 == 0:
        nram += 1
    return nram, len(xlist)

def ramas(b, dp):
    dlibre = b-2*dp
    return int(dlibre/35) + 2

def vueV(l, mpr1, mpr2):
    vue = (mpr1+mpr2)/l
    return vue

# 'avs' = (Av / s)_nec = Vs / (fy * d)
def avs(av, fy, h, dp, vu):
    return vu/(fy * (h - dp))

def fest(avs, nRam):
    return round(100 * avs/nRam, 3)

def limEst(h, dp, db, s):
    d = h-dp
    cond1 = min(d / 4, 0.6 * db, 15)
    cond2 = min(d / 2, 30)
    c1 = []
    c2 = []
    for i in s:
        if cond1 >= i:
            c1.append(i)
        if cond2 >= i:
            c2.append(i)
        else:
            break
    return c1, c2

def remrep(a):
    a.sort()
    a = list(dict.fromkeys(a))
    return a

xList = [5, 15, 25, 35, 45]
# xList = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125]

def ramLst(xList):
    if xList[-1]-xList[0] <=30:
        return [xList[0], xList[-1]]
    b = xList[-1] + xList[0]
    mid = b/2
    dist = mid-xList[0]
    larg = len(xList)
    ind = int(larg / 2)
    rang1 = xList[0:ind-1]
    rang1.append(xList[ind])
    estLst = []
    if larg % 2 == 0:
        c = ind-1
        while mid - xList[c] <= 15:
            c -= 1
        c += 1
        estLst.append(xList[c])
        for i in range(c, -1, -1):
            if estLst[0] - xList[i] >= 30:
                if estLst[0] - xList[i] > 30:
                    estLst.insert(0, xList[i+1])
                else:
                    estLst.insert(0, xList[i])
        for i in range(len(estLst) - 1, -1, -1):
            indx = len(xList)-1-xList.index(estLst[i])
            estLst.append(xList[indx])
        estLst.insert(0,xList[0])
        estLst.append(xList[-1])
        estLst = remrep(estLst)
    else:
        estLst1 = []
        estLst2 = []
        c = ind-1
        while mid - xList[c] <= 15:
            c -= 1
        c += 1
        estLst1.append(xList[c])
        estLst2.append(xList[ind])
        for i in range(c, -1, -1):
            if estLst1[0] - xList[i] >= 30:
                if estLst1[0] - xList[i] > 30:
                    estLst1.insert(0, xList[i + 1])
                else:
                    estLst1.insert(0, xList[i])
        for i in range(c, -1, -1):
            if estLst2[0] - xList[i] >= 30:
                if estLst2[0] - xList[i] > 30:
                    estLst2.insert(0, xList[i + 1])
                else:
                    estLst2.insert(0, xList[i])
        for i in range(len(estLst) - 1, -1, -1):
            indx = len(xList)-1-xList.index(estLst[i])
            estLst1.append(xList[indx])
            estLst2.append(xList[indx])
        if len(estLst1) < len(estLst2):
            for i in range(len(estLst1)-1, -1, -1):
                indx = len(xList) - 1 - xList.index(estLst1[i])
                estLst1.append(xList[indx])
            estLst1.insert(0, xList[0])
            estLst1.append(xList[-1])
            estLst1 = remrep(estLst1)
            estLst = estLst1
            estLst1 = remrep(estLst1)
        else:
            for i in range(len(estLst2)-2, -1, -1):
                indx = len(xList) - 1 - xList.index(estLst2[i])
                estLst2.append(xList[indx])
            estLst2.insert(0, xList[0])
            estLst2.append(xList[-1])
            estLst2 = remrep(estLst2)
            estLst = estLst2
    return estLst

def vS(fy, nRam, aEst, h, dp, s):
    return round(fy * nRam * aEst * (h-dp) / s, 2)

# se desprecia vc
def vReqV(vdl, vue):
    return 0.75 * (vdl + vue)

def corteV():
    pass

from time import time

dp, es, fc, fy, ey, eu, b1, cH, cS = 5, 2100000, 250, 4200, 0.002, 0.003, 0.85, 75000, 7850000
lList, dList, estList = range(30, 110, 10), [12, 16, 18, 22, 25, 28, 32, 36], [10, 12, 16, 18, 22, 25]
tinicial = time()
asdf = list(optimusVig(58.7, 30.29, es, eu, ey, b1, fc, fy, dp, dList, lList, 1, 700, cH, cS))
print(asdf)
optC = optimusCol(b1, dp, es, eu, ey, fc, fy, 30, 144, dList, lList, cH, cS)
print(optC)
tiempo = round(time() - tinicial, 4)
print("tiempo de ejecución =", str(tiempo), "segundos")
XYplotCurv(optC[11], optC[1], optC[2], dp, eu, fy, fc, b1, es, ey, optC[12])
XYplotCurv(asdf[3], asdf[2], asdf[1], dp, eu, fy, fc, b1, es, ey, asdf[4])
XYplotCurv(asdf[8], asdf[2], asdf[1], dp, eu, fy, fc, b1, es, ey, asdf[7])
