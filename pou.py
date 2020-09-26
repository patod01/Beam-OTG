import numpy as np

# [cm] Distancia de recubrimiento
dp = 5
# [cm] Altura sección
h = 60
# [cm] Altura efectiva de la seccion
d = h-dp
# [cm] Ancho
b = 40
# [kgf/cm2] Resistencia a compresión del hormigon
fc = 250
# [kgf/cm2] Modulo de elasticidad del acero
Es = 2100000
# Deformación del 3 por mil
eu = 0.003
# Valor de beta basado en resistencia del hormigon
if fc < 280:
    beta1 = 0.85
elif 560 > fc >= 280:
    beta1 = 0.85-0.2/280*(fc-280)
elif fc > 560:
    beta1 = 0.65
# [kgf/cm2] Resistencia del acero A63-42H
fy = 4200
# [cm2] Area de acero para el refuerzo superior
As_sup = 19.6
# [cm2] Area de cada nivel de refuerzo lateral
A_lat = 9.8
# Numero de refuerzos laterales
n_lat = 2
# [cm2] Area de acero para el refuerzo inferior
As_inf = 19.6
# [cm2] Lista de areas de acero por nivel
A = np.append([As_sup], np.append(np.array([A_lat for i in range(n_lat)]), [As_inf]), axis=0)
# [cm] valor de c inicial correspondiente a 3/8 del valor de d
c = 3*d/8
# [kgf/cm2] Lista de resistencias del acero por nivel rellena por ceros
Fsi = np.zeros(len(A))
# Lista de posiciones de cada nivel de acero
y = np.array([5+i*round((h-2*dp)/(len(A)-1), 3) for i in range(len(A))])
# Lista de deformaciones por nivel de acero
ei = np.array([round(eu*(c-y[i])/c, 6) for i in range(len(A))])
# Deformacion del acero necesaria para que entre en fluencia
ey = fy/Es
# Valor inicial de PS
Ps = 0
# Valor inicial de error
error = 1
# Actualizacion de ei para las iteraciones de Fsi y c mientras no exista un error menor a 0.00001
while abs(error) > 0.00001:
    ei = np.array([round(eu*(c-y[i])/c, 6) for i in range(len(y))])
    # [kgf/cm2] actualizacion de Fsi por cada cambio de c
    for i in range(len(ei)):
        if ei[i] > ey:
            Fsi[i] = fy
        elif -ey <= ei[i] <= ey:
            Fsi[i] = Es*ei[i]
        elif ei[i] < -ey:
            Fsi[i] = -fy
    # [kgf/cm2] Lista de valores para la resistencia axial del acero
    for i in range(len(Fsi)):
        Ps = Fsi[i]*A[i]
    # [cm] Valor de c anterior
    c2 = -Ps/(0.85*beta1*fc*b)
    # [cm] Nuevo valor de c, correspondiente a la semi-suma de su valor anterior y su valor actual
    c = round((c2+c)/2, 4)
    # valor del error basado en diferencias de valor anterior y nuevo
    error = (c2-c)/c
# [kgf] Resistencia a compresion generada por el hormigon
Pc = 0.85*beta1*fc*b*c
# [kgf * cm] Momento de compresion generado por la zona de compresion
Mc = Pc*(h/2-0.85*c/2)
# Valor inicial de Ms
Ms = 0
# Valor inicial de As
As = 0
for i in range(len(y)):
    # [kgf * cm] Sumatoria de momentos por nivel de acero
    Ms = Ms+Fsi[i]*A[i]*(h/2-y[i])
    # [cm2] Sumatoria de areas por nivel de acero
    As = As + A[i]
# [tf * m] Valor de momento nominal con cambio de unidades
Mn = (Ms + Mc)/100000.
print(Ms/100000)
print(Mc/100000)
print(Mn)






