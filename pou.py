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
    # el valor 0.003 corresponde a eu, que es la tensión de rotura del hormigón
    # el valor 0.002 es el valor de fy
    # el valor 0.005 es la suma de fy y eu
    elif 560 >= fc >= 280:
        beta1 = 0.85 - 0.2 / 280 * (fc - 280)
        # si el valor de fc supera los 560[kgf/cm2], el valor de beta1 es 0.65
    elif fc > 560:
        beta1 = 0.65
        # la función devuelve el valor de beta1
    return beta1


""" Función para calcular et"""


# se define la función eT, que es la deformación unitaria
# del acero a una distancia c, h es la altura del perfil, se mide en [cm],
# dp es la distancia de recubrimiento de hormigón, se mide en [cm], desde
# un extremo del perfil al centro de la barra más próxima y c es la distancia
# donde se anaizan las tensiones del acero en el perfil, se mide en [cm]
def eT(h, dp, c):
    #el valor 0.003 corresponde a la deformación de rotura del hormigón
    et = 0.003 * (h - dp - c) / c
    # la función devuelve el valor de et
    return et


""" Función para calcular el valor de Ø """


# se define la función para el cálculo del factor de minoración phi para
# flexocompresión
def phi(et):
# el valor inicial de phi es 0.65
    Phi = 0.65
    # si el valor de et es menor a 0.002, phi se mantiene en 0.65
    if et < 0.002:
        Phi = 0.65
    # si el valor de et está entre 0.002 y 0.005,
    # éste varía linealmente entre 0.65 y 0.9
    elif 0.002 <= et <= 0.005:
        Phi = 0.65 + 0.25 / 0.003 * (et - 0.002)
    # si el valor de et es superior a 0.005, el valor de phi es 0.9
    elif et > 0.005:
        Phi = 0.9
    # la función devuelve el valor de phi
    return Phi


""" Función para calcular la excentricidad """


# se define la función para el cálculo de la excentricidad
def exc(pn, mn):
    # para evitar ZeroDivisionError, se establece un valor infinito
    # para valores pequeños de pn absoluto, en este caso menores a 0.0001
    if abs(pn) < 0.0001:
        e = str("infinita")
    # en caso de que pn tome un valor un poco más lejano de 0, se calcula la excentricidad
    # dividiendo mn por pn y redondeando la cifra a 3 decimales
    else:
        e = round(mn / pn, 3)
    # la función devuelve el valor de e
    return e


""" Función para calcular las áreas en barras de acero """


# define función Acirc para el cálculo de áreas en barras, donde D es el diámetro en [cm]
def Acirc(D):
    # el área del círculo es el cuadrado del diámetro por pi/4, redondeado a 3 decimales
    Ac = round(D ** 2 * 0.007854, 3)
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
    Asup = Acirc(Dsup) * nsup
    # Alat llama a la función para calcular el área del círculo y
    # multiplica por número de barras en niveles laterales, entrega
    # el valor en [cm2]
    Alat = Acirc(Dlat) * 2
    # Ainf llama a la función para calcular el área del círculo y
    # multiplica por número de barras en el nivel inferior, entrega
    # el valor en [cm2]
    Ainf = Acirc(Dinf) * ninf
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
    for i in range(1, nlat + 1):
        Y.append((h - Y[i - 1] - dp) / (nlat + 2 - i) + Y[i - 1])
        # se crea condicional que aproxima la posición vertical de los niveles
        # de barras en [cm], donde si es mayor a 0.5 su parte decimal se le añade
        # una unidad y en caso contrario nada
        if Y[i] - int(Y[i]) > 0.5:
            Y[i] = int(Y[i]) + 1
        else:
            Y[i] = int(Y[i])
    # se añade a la lista el último valor de Y, que corresponde al nivel inferior
    Y.append(h - dp)
    # la función devuelve la lista Y
    return Y


""" Función para crear la lista de valores de ei """


# se define la función eiList que crea una lista de la
# deformación  unitaria en cada línea de armadura
def eiList(Y, c):
# se crea una lista vacía que uardará los valores de ei
    ei = []
    # se calcula el valor de ei para cada nivel de enfierradura
    for i in range(len(Y)):
        # se añaden los valores de ei a la lista vacia para cada valor de Y
        # el valor 0.003 corresponde a la deformación de rotura del hormigón
        ei.append((c - Y[i]) / c * 0.003)
    # la función devuelve la lista de valores de ei generada
    return ei


""" Función para crear la lista de valores de fs en función de ei """


# se define función fsList que crea una lista de tensiones de compresión
# entrega la tensión del acero ubicado a una distancia y de la fibra más
# comprimida en [kgf/cm2]
def fsList(ei, ey, fy, Es):
    # se crea una lista vacía fs
    fs = []
    # se añade a la lista cada valor de tensión en compresión a la lista
    for i in range(len(ei)):
        # se crea la condición donde si el valor de ei supera a ey, que es la relación
        # de módul{os de elasticidad del hormión y acero, fc/Es y cuyo valor para este caso
        # es 4200 [kf/cm2], conserva este valor.
        if ei[i] > ey:
            # se añade el valor fy en la posición del ei correspondiente en la lista fs
            fs.append(fy)
        # en caso de que el valor de ei esté entre -ey y ey, se multiplica su valor
        # por el módulo de elasticidad del acero
        elif -ey <= ei[i] <= ey:
            # al igual que en el caso anterior, se añade ei a fs
            fs.append(Es * ei[i])
        # si el valor de ei es menor a -ey, su valor serrá -fy
        elif ei[i] < -ey:
            # se añade este valor a la lista fs
            fs.append(-fy)
    # la función devuelve el la lista fs enumerada
    return fs


""" Función para crear la lista de Ps """


# se define una función psList, que calcula la carga en compresión del acero en [kgf]
def psList(fs, A):
    # se crea una lista vacía para los valores de ps
    ps = []
    # para cada valor en la lista de A y fs, se obtiene su producto ps
    for i in range(len(A)):
    # se agregan a la lista los valores de ps
        ps.append(fs[i] * A[i])
    # la fución devuelve la lista ps generada
    return ps


""" Función para calcular la sumatoria Ps"""


# se define la función sumPs, que suma los valores de la lista ps, medida en [kgf]
def sumPs(fs, A):
    # el valor inicial de la sumatoria es cero
    sumps = 0
    # para cada valor de A y fs se calcula su producto y se acumula en sumps
    for i in range(len(A)):
        sumps = sumps + (fs[i] * A[i])
    # la función devuelve el valor de la sumatoria sumps
    return sumps


""" Función para calcular Pc """


# se define la función Pc, correspondiente a la carga en compresión aportada
# por el hormigón en [kgf]
def Pc(beta, fc, b, c):
    pc = 0.85 * beta * fc * b * c
    # la función devuelve el valor de pc
    return pc


"""  Función para calcular Pn """


# se calcula el valor de la carga nominal, correspondiente a la suma
# de ps y pn en [kgf]
def Pn(fs, A, pc):
    # se llama a la función sumPs para geerar lista ps
    ps = sumPs(fs, A)
    pn = ps + pc
    # la función devuelve el valor de pn
    return pn


"""  Función para calcular ØPn """


# define la función phiPn, que añade el factor de minoración
# phi calculado anteriormente
def phiPn(phi, pn):
    phiPn = phi * pn
    # la función devuelve el valor de phiPn calculado
    return phiPn


""" Función para calcular Ms """


# se define la funcion Ms, que calcula el momento nominal que aporta el acero,
# se mide enn [kgf*cm]
def Ms(fs, A, h, Y):
    # se calcula la lista ps llamando a la función psList
    ps = psList(fs, A)
    # se considera un valor de ms igual a cero inicialmente
    ms = 0
    # se hace la sumatoria invlucrando cada valor de ms
    for i in range(len(fs)):
        ms = ms + ps[i] * (0.5 * h - Y[i])
    # la función devuelve el valor total de ms
    return ms


""" Función para calcular Mc """


# se define la función Mc, que calcula el moento noinal que aporta el hormigón,
# se mide en [kgf*cm]
def Mc(pc, h, c):
    mc = 0.5 * pc * (h - 0.85 * c)
    # la función devuelve el valor calulado mc
    return mc


""" Función para calcular Mn """


# se define la función Mn, que calcula el momento nominal del elemento analizado
# es la suma de ms y mc, y, se mide en [kgf*cm]
def Mn(ms, mc):
    mn = ms + mc
    # la funcion devuelve el valor mn calculado
    return mn


""" Función para calcular phiMn """


# se define la función phiMn, que calcula el momento nominal del elemento
# con el factor de minoracion correspondiente incorporado, se mide en [kgf*cm]
def phiMn(phi, mn):
    phiMn = phi * mn
    # la función devuelve el valor de phiMn calculado
    return phiMn


""" Función para calcular el valor de c en línea neutra """


# se define la función cLN, que calcula el valor de c que pasa por la línea neutra
# quiere decir, que su valor de Pn es cero, se mide en [cm]
def cLN(h, b, dp, fc, A, Y, ey, fy, Es):
    #c1 toma el valor inicial dp
    c1 = dp
    #c2 toma el valor inicial h
    c2 = h
    # se establece un valor pn igual a 1 para entrar al ciclo while
    pn = 1
    # se calcula pn mientras el valor absoluto de pn sea mayor a 0.00001
    while abs(pn) > 0.00001:
        # c toma el valor de la semisuma de los valores iniciales c1 y c2
        c = (c1 + c2) / 2
        # se calcula la lista ei, fs y ps con el nuevo valor de c
        ei = eiList(Y, c)
        fs = fsList(ei, ey, fy, Es)
        ps = sumPs(fs, A)
        # se calcula el valor de Pc con el nuevo valor de c
        pc = Pc(beta, fc, b, c)
        # se calcula y redondea el valor de Pn a 4 decimales
        pn = round(pc + ps, 4)
        # si el nuevo valor de c es mayor a cero, c2 toma el valor de c
        if pn > 0:
            c2 = c
        # de lo contrario, c1 toma el valor de c
        else:
            c1 = c
        # cuando el valor absoluto de Pn es menor a 0.00001, se detiene el
        # bucle while y la función devuelve el valor de c ubicado en la
        # línea neutra aproximado a 3 decimales
    return round(c, 3)


""" Función para calcular el valor de c en compresión pura """


# se define función cComp, que calcula el valor máximo de c en compresión en [cm]
def cComp(beta, h, dp):
    # c entrega el valor máximo entre "h/beta" y "3(h-dp)"
    # y lo redondea a 3 decimales
    c = round(max(h / beta, 3 * (h - dp)), 3)
    # la función devuelve el valor c
    return c


""" Función para calcular el valor máximo de c """


# se define cMax, función que calcula el máximo valor de c para el 80% de la
# carga nominal del elemento en [cm]
def cMax(h, b, dp, fc, A, Y, ey, fy, Es):
    # pc es la carga nominal de compresión que aporta el hormigón del elemento en [kgf/cm2]
    pc = Pc((h * b - aSum(A)) / (h * b), fc, b, h)
    # ps es la carga nominal de compresión que aporta el acero del elemento en [kgf/cm2]
    ps = aSum(A) * fy
    # se calcula el 80% de la carga nominal, que corresponde a la suma de pc y ps en [kgf/cm2]
    pn = 0.8 * (pc + ps)
    # se le da un valor inicial cero a Pn, que corresponde al valor de carga nominal en [kgf/cm]
    # calculado con un valor de c determinado, la idea es encontrar un valor aproximado
    # a pn en [kgf/cm2]
    Pn = 0
    # c1 toma el valor inicial de 1
    c1 = 1
    # c2 toma el valor inicial 3(h-dp), que podría coincidir con el valor máximo de
    # c para la cargas nominal en compresión al 100%
    c2 = 3 * (h - dp)
    # se calcula el valor de Pn hasta que el valor absoluto de su diferencia sea
    # menor a 0.001
    while abs(pn - Pn) > 0.001:
        # c toma el valor de la semisuma de c1 y c2 para aproximarse
        # al valor de Pn buscado en [cm]
        c = (c1 + c2) / 2
        # se llama a la función eiList para generar lista de ei
        ei = eiList(Y, c)
        # se llama a la función fsList para generar una lista de fs
        fs = fsList(ei, ey, fy, Es)
        # se calcula la carga nominal en compresión que aporta el acero del elemento
        # realizando la sumatoria de los valores de ps con los valores de fs y
        # el área correspondiente a cada nivel de acero en [kgf/cm2]
        ps = sumPs(fs, A)
        # se calcula la carga nominal en compresión que aporta el hormigón del
        # elemento en [kgf/cm2]
        pc = Pc(beta, fc, b, c)
        # se calcula Pn sumando pc y ps, y luego, redondeando la cifra a 4 decimales en [kgf/cm2]
        Pn = round(pc + ps, 4)
        # si la diferencia de Pn y pn es positiva, c2 toma el valor de c
        if Pn - pn > 0.001:
            c2 = c
        # de lo contrario, c1 toma el valor de c
        else:
            c1 = c
    # cuando se encuentra un valor de c que entregue un valor de Pn muy aproximado a pn, se
    # detiene el bucle while y se guarda el último valor de c, y la función devuelve el valor
    # de c en [cm]
    return round(c, 3)


""" Función para calcular el elemento en compresion al 80% ó 100% """


# se define la función compP, que entrega una lista con los valores et, c en [cm], el factor
# de minoración phi, el momento nominal mn en [tonf*m], la carga nominal pn en [tonf], el
# momento nominal phiMn, que incluye el factor de minoración phi, en [tonf*m], la carga nominal
# phiPn, que incluye el factor de minoración phi, en [tonf], y la excentricidad e, que
# corresponde a mn/pn.
def CompP(porc, beta, fc, h, b, dp, A, Y, ey, fy, Es):
    # se calcula la carga nominal en compresión que aporta el hormigón del elemento,
    # en [kgf], cuyo valor de beta1 se considera 1, para no dividirlo por el mismo después
    pc = Pc(1, fc, b, h)
    # se calcula la carga nominal en compresión que aporta el elemento, para ello se calcula
    # en el mismo ps, y se multiplica por el porcentaje al cual corresponde, 100 u 80%,
    # luego se divide por 10e5 para pasarlo a [tonf]
    pn = round(porc * (pc + aSum(A) * fy) / 100000, 2)
    # ya que está en compresión, su factor de minoración es 0.65
    Phi = 0.65
    # en caso de que sea 100%
    if porc == 100:
        # se calcula el valor de c correspondiente, en [cm]
        c = cComp(beta, h, dp)
        # se calcula el valor de et, dado un c y se redondea a 5 decimales
        et = round(eT(h, dp, c), 5)
        # se calcula el valor de phiPn, que toma el valor pn y lo muliplica
        # por el factor de minoración y por 0.8, luego, lo redondea a 2 decimales
        phipn = round(pn * Phi * .8, 2)
        # ya que no hay excentricidad, su momento nominal es cero, en [tonf]
        mn = 0
        # por lo que su momento nominal también es cero
        phimn = 0
        # al estar 100% en compresión, no existe excentricidad
        e = 0
    # en caso de que sea 80%
    elif porc == 80:
        # se calcula el valor de c para el elemento en compresión al 80%, en [cm]
        c = cMax(h, b, dp, fc, A, Y, ey, fy, Es)
        # se calcula el valor de et y se redondea a 5 decimales
        et = round(eT(h, dp, c), 5)
        # se calcula el valor de phiPn para este caso, que toma el valor pn
        # y lo muliplica por el factor de minoración, luego, lo redondea a 2 decimales
        phipn = round(pn * Phi, 2)
        # se calcula el momento nominal que aporta el hormigón del elemento, en [kgf*cm]
        mc = Mc(pc * .85, h, c)
        # se calcula la lista de ei con el valor de c calculado
        ei = eiList(Y, c)
        # se calcula la lista de fs con la lista ei generada, en [kgf/cm2]
        fs = fsList(ei, ey, fy, Es)
        # se calcula el momento nominal que aporta el acero del elemento con la lista
        # ei generada, en [kgf*cm]
        ms = Ms(fs, A, h, Y)
        # se calcula el momento nominal que aporta el elemento con este valor de c, el
        # resultado se divide por 1000 y aproxima a 2 decimales para tener las unidades
        # en [tonf*m]
        mn = round(Mn(ms, mc) / 100000, 2)
        # se calcula phiMn, multiplicando por el factor de minoración phi, en [tonf*m]
        phimn = round(phiMn(Phi, mn), 2)
        # se calcula la excentricidad dividiendo mn por pn y redondeando la cifra a 3 decimales
        e = round(mn/pn, 3)
    # se genera una lista con los valores calculados et, c, Phi,
    # mn, pn, phimn, phipn y e
    Lres = [et, c, Phi, mn, pn, phimn, phipn, e]
    # la función dedvuelve la lista Lres generada
    return Lres


""" Función para calcular el elemento a traccion pura """


# se define una función que generará una lista que calcule et, c, Phi, mn, pn, phimn, phipn, e
# para un elemento en tracción pura
def Trac(fy, A):
    # se define el valor de c como cero
    c = 0
    # se calcula el momento nominal considerando solo ps, ya que c=0, en [tonf]
    pn = round(-aSum(A) * fy / 1000, 2)
    # el valor de phi es 0.9 por estar en tracción al 100%
    Phi = 0.9
    # se calcula phiPn multiplicando pn por el factore de minoración phi, en [tonf]
    phipn = round(pn * Phi, 2)
    # no hay momento porque c=0
    mn = 0
    # no hay phiMn porque mn=0
    phimn = 0
    # si el momento es cero, la excentricidad también
    e = 0
    # et toma un valor indefinido
    et = "--"
    # se genera una lista con los valores calculados et, c, Phi,
    # mn, pn, phimn, phipn y e
    Lres = [et, c, Phi, mn, pn, phimn, phipn, e]
    # la función devuelve la lista Lres generada
    return Lres


""" Función para crear una lista resumen """


# se define la función Resumen, que crea una lista para cualquier valor de c entre
# flexión simple y compresión pura al 80%, iicorpora a la lista et, c, Phi, mn, pn,
# phimn, phipn y en [kgf]
def Resumen(beta, fc, b, c, Y, h, dp, ey, fy, Es, A):
    # se calcula la carga nominal que aporta el hormigón del elemento en compresión en kgf
    pc = Pc(beta, fc, b, c)
    # enera la lista ei para el valor de c inresado
    ei = eiList(Y, c)
    # se calcula et y se redondea a 5 decimales
    et = round(eT(h, dp, c), 5)
    # se calcula el factor de minoración phi
    Phi = phi(et)
    # se genera la lista fs, medida en [kgf/cm2]
    fs = fsList(ei, ey, fy, Es)
    # se calcula el valor de pn, inresando el valor de pc y calculando el de ps con la lista
    # anterior, el resultado se redondea a 2 decimales y queda en [tonf]
    pn = round(Pn(fs, A, pc) / 1000, 2)
    # se calcula la carga nominal con el factor de minoración phi inncluído y
    # redondeado a 2 decimales, las unidades son en [tonf]
    phipn = round(phiPn(Phi, pn), 2)
    # se calcula el momento nominal que aporta el hormigón del elemento,
    # se redondea el valor a 2 decimales y se transforma a [tonf*m]
    mc = round(Mc(pc, h, c) / 100000, 2)
    # se calcula el momento nominal que aporta el acero del elemento
    # y se redondea a 2 decimales, luego se transforma a [tonf*m]
    ms = round(Ms(fs, A, h, Y) / 100000, 2)
    # se calcula el momento nominal total sumando mc y ms, el resultado queda en [tonf*m]
    mn = round(Mn(ms, mc), 2)
    # se calcula el valor de phiMn, donde se multiplica mn por el factor de minoración phi,
    # el resultado se redondea a 2 decimales y queda en [tonf*m]
    phimn = round(phiMn(Phi, mn), 2)
    # se calcula la excentricidad dividiendo mn por pn, el resulado queda en [m]
    e = exc(pn, mn)
    # se genera la lista Lres, que alacena los valores de et, c, Phi, mn, pn, phimn, phipn y e
    Lres = [et, c, Phi, mn, pn, phimn, phipn, e]
    # la función devuelve la lista Lres
    return Lres


""" Función para calcular elementos en flexion simple """


# se define la función FS, donde se calculan os parámetros et, c, Phi,
# mn, pn, phimn, phipn y e en flexión simple
def FS(beta, fc, b, Y, h, dp, ey, fy, Es, A):
    # se calcula el valor de c que pasa por lla línea neutra en [cm]
    c = round(cLN(h, b, dp, fc, A, Y, ey, fy, Es), 3)
    # Se genera la lista de valores et, c, Phi, mn, pn, phimn, phipn y e para
    # flexion simple
    FS = Resumen(beta, fc, b, c, Y, h, dp, ey, fy, Es, A)
    # La función devuelve la lista generada por FS
    return FS


""" Función para calcular elementos en condicion balanceada """


# se define la función FS, donde se calculan os parámetros et, c, Phi,
# mn, pn, phimn, phipn y e en condición balanceada
def cBal(beta, fc, b, Y, h, dp, ey, fy, Es, A):
    # se calcula el valor de c para condición balanceada en [cm]
    c = 0.6 * (h - dp)
    # Se genera la lista de valores et, c, Phi, mn, pn, phimn, phipn y e para
    # la condición balanceada
    CB = Resumen(beta, fc, b, c, Y, h, dp, ey, fy, Es, A)
    # La función devuelve la lista generada por CB
    return CB


""" Función para calcular elementos con deformaciones del 0.5% """


# se define la función FS, donde se calculan os parámetros et, c, Phi,
# mn, pn, phimn, phipn y e en epsilon=0.005
def e5(beta, fc, b, Y, h, dp, ey, fy, Es, A):
    # se calcula el valor de c para condición ep=0.005 en [cm]
    c = 0.375 * (h - dp)
    # Se genera la lista de valores et, c, Phi, mn, pn, phimn, phipn y e para
    # ep=0.005
    E5 = Resumen(beta, fc, b, c, Y, h, dp, ey, fy, Es, A)
    # La función devuelve la lista generada por E5
    return E5


""" Función para encontrar el rango de interacción """


# se define la función Rang, donde se establecenn los valores iniciaes y finales de
# mn y pn para interpolar los valores de pu y mu, para posteriormente callcular FU,
# CP100m corresponde al valor de mn del elemento en compresión al 100%,
# CP80p corresponde al valor de pn del elemento en compresión al 80%
# CP80m corresponde al valor de mn del elemento en compresión al 80%
# CBp corresponde al valor de pn del elemento en condición balanceada
# CBm corresponde al valor de mn del elemento en condición balanceada
# E5p corresponde al valor de pn del elemento en epsilon = 0.005
# E5m corresponde al valor de mn del elemento en epsilon = 0.005
# FSp corresponde al valor de pn del elemento en flexión simple
# FSm corresponde al valor de mn del elemento en flexión simple
# TRp corresponde al valor de pn del elemento en Tracción pura
# TRm corresponde al valor de mn del elemento en Tracción pura
def Rang(pu, mu, CP100m, CP80p, CP80m, CBp, CBm, E5p, E5m, FSp, FSm, TRp, TRm):
    # e1 es la excentricidad del elemento en compresón pura all 80%
    e1 = CP80m / (CP80p+0.0001)
    # e2 es la excentrcidad del elemento en condición balanceada
    e2 = CBm / (CBp+0.0001)
    # e3 es la excentricidad del elemento en epsilon=0.005
    e3 = E5m / (E5p+0.0001)
    # si la carga solicitada es cero, toma el valor de 0.001 para evitar zeroErrorDivision
    if pu == 0:
        pu = 0.001
    # luego de lo anterior, se define la excentricidad entre mu y pu
    e = mu / pu
    # si la excentricidad solicitada es menor a e1, la recta a interpolar toma los valores
    # p2 = CP80p, p1 = CP80p, m2 = CP100m y m1 = CP80m
    if 0 <= e <= e1:
        p2 = CP80p
        p1 = CP80p
        m2 = CP100m
        m1 = CP80m
        zona = "compresión_pura"
    # si la excentricidad solicitada está entre e1 y e2, la recta a interpolar toma los valores
    # p2 = CP80p, p1 = CBp, m2 = CP100m y m1 = CBm
    elif e1 < e < e2:
        p2 = CP80p
        p1 = CBp
        m2 = CP100m
        m1 = CBm
        zona = "condición_balanceada"
    # si la excentricidad solicitada está entre e2 y e3, la recta a interpolar
    # toma los valores p2 = CBp, p1 = E5p, m2 = CBm y m1 = E5m
    elif e2 <= e < e3:
        p2 = CBp
        p1 = E5p
        m2 = CBm
        m1 = E5m
        zona = "epsilon=0.005"
    # si la excentricidad solicitada es mayor a e3, la recta a interpolar
    # toma los valores p2 = E5p, p1 = FSp, m2 = E5m y m1 = FSm
    elif e > e3:
        p2 = E5p
        p1 = FSp
        m2 = E5m
        m1 = FSm
        zona = "flexión_simple"
    # si la excentricidad solicitada es a cero, la recta a interpolar
    # toma los valores p2 = FSp, p1 = TRp, m2 = FSm y m1 = TRm
    elif e < -0.001:
        p2 = FSp
        p1 = TRp
        m2 = FSm
        m1 = TRm
        zona = "tracción_pura"
    # se crea la lista Rango, que contiene los valores de p1, p2, m1, m2
    Rango = [p1, p2, m1, m2, zona]
    # print(Rango)
    # la función devuelve la lista Rango
    return Rango


""" Cálculo de AVS (revisar)"""


# se defne funcion AVS, que calcula el area total de acero del elemento que aportará al
# corte nominal Vn
def AVS(Dest, Nramas, s):
    AVS = Acirc(Dest) * Nramas/s
    # la función devuelve el valor AVS calculado
    return AVS


""" Función para calcular Vn (revisar)"""


# se define la función Vn, que calcula el corte nominal que tolera el elemento
def Vn(fc, Nu, b, h, dp, AVS):
    d = h - dp
    # se calcula el área de la sección de hormigón en corte en [cm2]
    Ag = b * h
    # si Nu es igual a cero, el valor de corte nominal aportado por el hormigón
    # Vc se calcula como Vc=(fc/10)**0.5*10/6*b*d
    if Nu == 0:
        Vc = (fc / 10) ** 0.5 * 10 / 6 * b * d
    # si el valor de Nu es mayor a cero, se calcula un factor inicial "factor" y
    # un factor límite factorLIM
    elif Nu > 0:
        factor = (1 + (Nu / (14 * Ag * 10)))
        factorLIM = (1 + 0.29 * Nu / (Ag * 10)) ** 0.5
        # si el factor es superior al factor límite factorLIM, "factor"
        # toma el valor de factor LIM
        if factor > factorLIM:
            factor = factorLIM
        # para este caso, el corte nominal que aporta el hormigón Vc es:
        Vc = (fc / 10) ** 0.5 * 10 / 6 * b * d * factor
    # en caso contrario, quiere decir, que Nu sea menor a cero, se considera:
    else:
        factor = 1 + 0.29 * Nu / (Ag * 10)
        Vc = (fc / 10) ** 0.5 * 10 / 6 * b * d * factor
        # si Vc toma un valor negativo, se considera cero
        if Vc < 0:
            Vc = 0
    # se calcula el valor de corte nominal que aporta el acero del elemento Vs y
    # el límite de éste VsLIM
    Vs = AVS * fy * d
    VsLIM = 4 * (fc / 10) ** 0.5 * 10 / 6 * b * d
    # en caso de que Vs sea mayor a su valor límite, toma el valor VsLIM
    if Vs > VsLIM:
        Vs = VsLIM
    # finalmente, el corte nominal tolerado por elemento es la suma de Vs y Vc
    Vn = Vc + Vs
    # la función devuelve el valor de Vn, redondeado a 3 decimales en [tonf]
    return round(Vn / 1000, 3)


""" Cálculo de phiVn """


# se define la función phiVn, que incorpora el factor de reducción phiV. éste será 0.6
# para elementos sismorresistentes y 0.75 para los que no
def phiVn(vn, phiV):
    phiVn = vn * phiV
    # la función devuelve el valor de phiVn en [tonf]
    return phiVn


""" Función para calcular la ecuacion de la recta """


# m toma el valor inicial 1
m = 1
# el error toma el valor inicial 1
error = 1


# se define la función ecRecta, que calcula el valor interpolado de la proyección de
# la recta que se genera desde el origen hasta el punto (mu,pu), e intersecta con la
# curva de interacción, entre los puntos p1, p2, m1, m2 del rango en que se encuentre
def ecRecta(mu, pu, p2, p1, m2, m1):
    # se calcula valor absoluto del momento y añade 0.0001 para evitar ZeroDivisionError
    mu = abs(mu)+0.0001
    #
    pu = pu + 0.0001
    # se calcula la excentricidad dividiendo mu por pu
    e = mu/pu
    # se calcula la pendiente de la recta m
    m = (p2-p1)/(m2-m1+0.0001)
    # se calcula el valor de phimn intersectando ambas curvas, el valor se redondea
    # a 3 decimales en [tonf*m]
    phimn = round((p1-m*m1)/(1/e-m), 3)
    # se calcula el valor de phipn intersectando ambas curvas, el valor se redondea
    # a 3 decimales en [tonf]
    phipn = round(phimn/e, 3)
    # el factor de utilización FU, es la división entre la carga solicitada pu y la carga
    # nominal pn, el resultado se redondea a 3 decimales
    if phipn < 0.0001:
        FU = abs(round(pu/0.0001, 3))
    else:
        FU = abs(round(pu/phipn, 3))
    # la función imprime los valores de phimn, phipn y FU
    ecList = [FU, phimn, phipn, round(phimn/phipn, 3)]
    return ecList


""" Datos de entrada """


# la carga solicitada pu en [tonf]
pu = [0.1, 200, 400, 700]
# el momento solicitado mu en [tonf*m]
mu = [0.1, 50, 100, 300]
# la resistencia del hormigón a la compresión en [kgf/cm2]
fc = 250
# la resistencia del acero en su rango elástico a compresión y tracción en [kgf/cm2]
fy = 4200
# el módulo de elasticidad del acero A63 42H en [kgf/cm2]
Es = 2100000
# la relación entre fy y Es
ey = fy/Es
# la atura de la sección h en [cm]
h = [40, 60, 80]
# el ancho de la sección b en [cm]
b = [20, 40, 60, 80]
# la distancia de recubrimiento desde el centro de la primera
# barra a un extremo de la sección en [cm]
dp = 5
# el valor de beta1 según la resistencia del hormigón
beta = beta(fc)
# el diámetro de las barras de acero ubicadas en la posición superior del perfil en [cm]
Dsup = [22, 25, 28, 32, 36]
# la cantidad de barras de acero en la posición superior del perfil en [cm]
nsup = [2, 3, 4, 5, 6, 8]
# el diámetro de las barras de acero laterales en [cm]
Dlat = [16, 22, 25, 28, 32, 36]
# la cantidad de niveles de barras laterales de acero, cada nivel lleva 2 barras
nlat = [1, 2, 3, 4, 5, 6, 8]
# el diámetro de las barras de acero del nivel inferior del perfil en [cm]
Dinf = [22, 25, 28, 32, 36]
# el número de barras de acero en el nivel inferior
ninf = [2, 3, 4, 5, 6]


""" Cálculo de flexión compuesta"""


def listaexc(pu, mu, fc, fy, Es, ey, h, b, dp, beta, Dsup, Dlat, Dinf, nsup, nlat, ninf):
    # entrada = [pu, mu, fc, fy, Es, ey, h, b, dp, beta, Dsup, Dlat, Dinf, nsup, nlat, ninf]
    # print(entrada)
    # A es la lista de áreas de acero por nivel
    A = Alist(Dsup, nsup, Dlat, nlat, Dinf, ninf)
    # Y es la lista de posiciones Yi de los niveles de barras de acero
    Y = Ylist(h, dp, nlat)
    # se crea lista con valores de et, # c, Phi, mn, pn, phimn,
    # phipn y e en flexión simple
    Fs = FS(beta, fc, b, Y, h, dp, ey, fy, Es, A)
    # se crea lista con valores de et, # c, Phi, mn, pn, phimn,
    # phipn y e en epsilon=0.005
    E5 = e5(beta, fc, b, Y, h, dp, ey, fy, Es, A)
    # se crea lista con valores de et, c, Phi, mn, pn, phimn,
    # phipn y e en condición balanceada
    CB = cBal(beta, fc, b, Y, h, dp, ey, fy, Es, A)
    # se crea lista con valores de et, c, Phi, mn, pn, phimn,
    # phipn y e en tracción pura
    TR = Trac(fy, A)
    # se crea lista con valores de et, c, Phi, mn, pn, phimn,
    # phipn y e en compresión pura al 80%
    CP80 = CompP(80, beta, fc, h, b, dp, A, Y, ey, fy, Es)
    # se crea lista con valores de et, c, Phi, mn, pn, phimn,
    # phipn y e en compresión pura al 100%
    CP100 = CompP(100, beta, fc, h, b, dp, A, Y, ey, fy, Es)
    # print(CP100)
    # print(CP80)
    # print(CB)
    # print(E5)
    # print(FS)
    # print(TR)


    """ Cálculo de Corte """


    # # se inresa el valor de Nu solicitado
    # Nu = 0
    # # se ingresa el de estribos en [cm]
    # Dest = 1
    # # se igresa el número de ramas
    # Nramas = 4
    # # s es la distancia de los estribos en [cm]
    # s = 20
    # # se calcula AVS
    # AVS = AVS(Dest, Nramas, s)
    # # se calcula el corte nominnal Vn en [tonf]
    # vn = Vn(fc, Nu, b, h, dp, AVS)
    # # se define el factor de minoración phiV, donde 0.6 corresponde a elementos
    # # sismorresistentes y 0.75 al resto de elementos
    # phiV = 0.6
    # # se calcula el corte nominal minorado phiVn multiplicando V por phiV
    # phiVn = phiVn(vn, phiV)
    # print(vn)
    # print(phiVn)


    """ Cálculo de FU """


    # CP80p corresponde al valor de pn del elemento en compresión al 80%
    CP80p = CP80[6]
    # CP100m corresponde al valor de mn del elemento en compresión al 100%,
    CP100m = CP100[5]
    # CP80m corresponde al valor de mn del elemento en compresión al 80%
    CP80m = CP80[5]
    # CBp corresponde al valor de pn del elemento en condición balanceada
    CBp = CB[6]
    # CBm corresponde al valor de mn del elemento en condición balanceada
    CBm = CB[5]
    # E5p corresponde al valor de pn del elemento en epsilon = 0.005
    E5p = E5[6]
    # E5m corresponde al valor de mn del elemento en epsilon = 0.005
    E5m = E5[5]
    # FSp corresponde al valor de pn del elemento en flexión simple
    FSp = Fs[6]
    # FSm corresponde al valor de mn del elemento en flexión simple
    FSm = Fs[5]
    # TRp corresponde al valor de pn del elemento en Tracción pura
    TRp = TR[6]
    # TRm corresponde al valor de mn del elemento en Tracción pura
    TRm = TR[5]
    # Pu_Mu llama a la función Rang para determinar en que rango de la curva de interacción
    # se encuentra la recta proyectada entre el orenn y el punto (mu, pu), devolviendo así
    # los puntos p1, p2, m1 y m2 del rango en cuestión
    Pu_Mu = Rang(pu, mu, CP100m, CP80p, CP80m, CBp, CBm, E5p, E5m, FSp, FSm, TRp, TRm)
    p1 = Pu_Mu[0]
    p2 = Pu_Mu[1]
    m1 = Pu_Mu[2]
    m2 = Pu_Mu[3]
    zona = Pu_Mu[4]
    # print(round(mu / pu, 3))
    # print(str(p1) + " \n " + str(p2) + " \n " + str(m1) + " \n " + str(m2))
    # se llaa a la función ecRecta para determinar los valores de phiMn, phiPn y FU de
    # acuerdo a la carga pu y momento mu solicitados
    ec = ecRecta(mu, pu, p2, p1, m2, m1)
    # ec = [e, phimn, phipn, FU]
    # pu, mu, fc, fy, Es, ey, h, b, dp, beta, Dsup, Dlat, Dinf, nsup, nlat, ninf
    cuantia = 0.7854*(Dsup**2*nsup+Dlat**2*nlat+Dinf**2*ninf)/(b*h*100)
    a = [round(mu/pu, 3), mu, pu, ec[0], ec[1], ec[2], ec[3], h, b, Dsup, nsup, nlat, Dlat, Dinf, ninf, zona, round(cuantia,5)]
    # print(str(a[0])+" "+str(a[1])+" "+str(a[2])+" "+str(a[3])+" "+str(a[4])+" "+str(a[5])+" "+str(a[6])+" "+str(a[7])+" "
    #       +str(a[8])+" "+str(a[9])+" "+str(a[10])+" "+str(a[11])+" "+str(a[12])+" "+str(a[13])+" "+str(a[14])+" "+str(a[15])+" ")
    return a


""" Iteraciones """


file = open("C:/p/asdf.txt", 'a')
# datos de prueba
# asd = listaexc(144, 30, fc, fy, Es, ey, 60, 40, dp, beta, 25, 4, 2, 25, 25, 4)
# print(asd)
file.write("e mu pu FU phimn phipn emn h b Dsup nsup nlat Dlat Dinf ninf zona \n")
cuenta = 0
for i in pu:
    for j in mu:
        for k in h:
            for l in b:
                for n in Dsup:
                    for o in Dlat:
                        for p in Dinf:
                            for q in nsup:
                                for r in nlat:
                                    for s in ninf:
                                        a=listaexc(i, j, fc, fy, Es, ey, k, l, dp, beta, n, o, p, q, r, s)
                                        print(a)
                                        if a[3] < 1 and 0.00333 < a[16] < 0.01935:
                                            file.write(str(a)+"\n")
                                            cuenta += 1
                                        else:
                                            pass
file.close()
# print(cuenta)


""" Calcula tiempo de ejecución """


# se registra el tiempo al final de la ejecución del programa
tfinal = time()
# se calcula el tiempo total de la ejecución, desde su inicio hasta el fin
ttotal = tfinal - tinicial
# se imprime el tiempo de ejecución ttotal
print(ttotal)
