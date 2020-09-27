""" Valor de beta """

def beta(fc):
    if fc < 280:
        beta1 = 0.85
    elif 560 > fc >= 280:
        beta1 = 0.85-0.2/280*(fc-280)
    elif fc > 560:
        beta1 = 0.65
    return beta1

""" Area barras de acero """

def Acirc(D):

    Ac = round(D**2*0.7854, 3)
    return Ac

""" Lista de Areas de acero por nivel """

def Alist(Dsup, Dlat, nlat, Dinf):

    Asup = Acirc(Dsup)
    Alat = Acirc(Dlat)
    Ainf = Acirc(Dinf)
    A = [Asup]
    for i in range(nlat):
        A.append(Alat)
    A.append(Ainf)
    return A

""" Lista de posicion de niveles de areas """

def Ylist(h, dp, nlat):

    Y = [dp]
    for i in range(1,nlat+1):
        Y.append((h-Y[i-1]-dp)/(nlat+2-i)+Y[i-1])
        if Y[i]-int(Y[i])>0.5:
            Y[i]=int(Y[i])+1
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

    for i in range(len(ei)):
        if ei[i] > ey:
            fs = fy
        elif -ey <= ei[i] <= ey:
            fs = Es * ei[i]
        elif ei[i] < -ey:
            fs = -fy
        return fs

""" Lista de Ps """

def psList(fs, A):

    ps = []
    for i in range(len(A)):
        ps = fs[i]*A[i]
    return ps

""" Calculo Pc """

def Pc(beta, fc, b, c):

    pc = 0.85*beta*fc*b*c
    return pc

"""  Calculo de ØPn """

def phiPn(phi, ps, pc):
    pn = phi*(ps + pc)
    return pn

""" Calculo Ms  """

def Ms(ps, h, Y):

    for i in range(len(ps)):
        ms = ps[i]*(0.5*h-Y[i])
    return ms

""" Calculo Mc """

def Mc(pc, h, c):

    mc = 0.5*pc*(h-0.85*c)
    return mc

""" Calculo de ØMn """

def phiMn(phi, ms, mc):

    mn = phi*(ms+mc)
    return mn

""" Calcular valor de c en linea neutra """

def cLN(h, b, dp, pc, fc, A, Y, ey, fy, Es):
    
    c1 = dp
    c2 = h-dp
    while abs(pn)>0.00001:
        c = (c1 + c2) / 2
        pc = Pc(beta, fc, b, c)
        ps = psList(fs, A)
        ei = eiList(Y, c)
        fs = fsList(ei, ey, fy, Es)
        pn = pc + ps
        if pn<0:
            c1 = c
        else:
            c2 = c
    return c

