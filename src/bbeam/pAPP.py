if MERYTEST := 0:
    import tracemalloc
    tracemalloc.start()
from math import pi as π, sin
from numpy import array, zeros, round as mround
from numpy.linalg import inv
import matplotlib.pyplot as pete

def line(): print('---- -- - -- ----')
p = print
# def varsC(dirl=dir()):
#     i = 0
#     lista = []
#     ipyvars = ['In', 'Out', 'exit', 'get_ipython', 'quit']
#     for j in dirl:
#         if j[0] == '_': continue
#         elif j in ipyvars: continue
#         i += 1
#         lista.append((i, j))
#     return lista
# def period():
#     """formulas del paper en base al ejemplo de la tesis"""
#     Sc = 67500*8*0.01**4
#     Sb = 160000*6*0.01**4
#     B = 1.5*3
#     H = 1.2*2
#     W = 0 # FALTA
#     T = 0.374*H**1.1*W/(B**0.3*Sc**0.31*Sb**0.35)
#     print(0.09*H/3**0.5)
#     print(0.075*H**0.75)
#     print(0.0466*H**0.9)
#     print(T)
#     print(0.879*H**0.43*3**0.65*W**0.18/B)
#     print(0.475*H**0.547*3**0.103/B**0.271)
#     print(0.428*H**0.545/B**0.185)
#     print(0.28*H**0.54)
#     print(0.012*H)
#     return

def arbol(task='', project='test_proj'):
    """Genera el arbol de un proyecto."""
    from os import getcwd, listdir, chdir, walk
    from os import mkdir, rename, unlink, rmdir
    from os.path import exists
    print('\ncurrent directory:', getcwd(), '(.)', end='\n\n')
    if task == 'del':
        carpetas = [i[0] for i in list(walk(project))]
        archivos = []
        for folder in list(walk(project)):
            for file in folder[2]:
                archivos.append(f'{folder[0]}/{file}')
        print('carpetas por borrar:')
        for folder in carpetas: print(folder)
        print('archivos por borrar:')
        for file in archivos: print(file)
        ask = input('\nwanna continue? (si/no) ')
        if ask == 'si':
            for file in archivos: unlink(file)
            for folder in carpetas[::-1]:
                rmdir(folder)
            print('everything has been deleted.')
        else:
            print('no se ha eliminado nada.')
    elif task == 'new':
        if exists(project):
            print('the project already exists!!!\n')
            return
        else:
            mkdir(project)
            mkdir(project + '/src')
            mkdir(project + '/tmp')
            [print(f'creado: {i[0]}') for i in list(walk(project))]
            with open(project + '/input.txt', 'w', encoding='utf-8') as fcuk:
                fcuk.write("numero de pisos          : 4\nnumero de vanos          : 3\nnumero de marcos         : 4\naltura de piso           : 300 [cm]\ndistancia entre columnas : 700 [cm]\ndistancia entre marcos   : 700 [cm]\n\nancho de columnas        : 90 [cm]\nancho de vigas           : 50 [cm]\naltura de vigas          : 90 [cm]\nespesor de losas         : 15 [cm]\n\nmodulo de poisson 𝜈      : 0.25\nresistencia f'c hormigon : 25 [MPa]\ndensidad del hormigon    : 2500 [kg/m3]\nprecio hormigon          : 75000 [$/m3]\nprecio acero             : 1000 [$/kg]\n\nzona sismica             : 3\ntipo de suelo            : E\ncategoria de edificacion : II\nfactor de modificacion R : 7\nsobrecarga de uso        : 500 [kg/m2]\nporcentaje de sobrecarga : 0.25\n")
            print(f'creado: {project}/input.txt\n')
    return

# lectura de datos

papel = {
    'proyecto': '',
    'input': {
        # geometria del edificio
        'n_pisos': '',
        'n_vanos': '',
        'n_marcos': '',
        'h_pisos': '',
        'b_losas': '',
        'd_losas': '',
        # geometria de los elementos
        'b_COL': '',
        'b_VIG': '',
        'h_VIG': '',
        't_losas': '',
        # propiedades de los materiales
        '𝜈': '',
        'fc': '',
        '𝛾': '',
        'valH': '',
        'valA': '',
        # caracteristicas sismicas
        'zona': '',
        'suelo': '',
        'categoria': '',
        'R': '',
        'Q_L': '',
        'prcL': '',
    },
    'config': {'format': 'png', 'graf': False},
    'VERB': False, # true = verbose on
}

def lectura():
    """Lee los parametros definidos por el usuario desde un txt. Si\
    se omite el valor, se autocompleta con los valores por defecto."""

    if 1:
        what = input('comenzar un proyecto nuevo? (si/no) ')
        proyecto = input('ingrese el nombre del proyecto: ')
    else:
        what = 'no'
        proyecto = 'test_proj'
    proyecto = './' + proyecto
    papel['proyecto'] = proyecto
    if what == 'si':
        arbol('del', proyecto)
        input('presiona enter...')
        arbol('new', proyecto)
    
    with open(proyecto + '/input.txt', encoding='utf-8') as in_text:
        in_temp = in_text.readlines()
    
    if input('imprimir reporte grafico al finalizar el analisis? (si/no) ') == 'si':
        papel['config']['graf'] = True
    
    if papel['VERB']: print(in_temp)
    default = {
        'numero de pisos': '4',
        'numero de vanos': '3',
        'numero de marcos': '4',
        'altura de piso': '300',
        'distancia entre columnas': '700',
        'distancia entre marcos': '700',
        
        'ancho de columnas': '90',
        'ancho de vigas': '50',
        'altura de vigas': '90',
        'espesor de losas': '15',
        
        'modulo de poisson 𝜈': '0.25',
        'resistencia f\'c hormigon': '25',
        'densidad del hormigon': '2500',
        'precio hormigon': '75000',
        'precio acero': '1000',
        
        'zona sismica': '3',
        'tipo de suelo': 'E',
        'categoria de edificacion': 'II',
        'factor de modificacion R': '7',
        'sobrecarga de uso': '500',
        'porcentaje de sobrecarga': '0.25',
    }

    user_op = [] # option names from txt file
    user_in = [] # values given by the user

    for i in in_temp:
        if i.startswith('\n') == False:
            user_op.append(i.partition(':')[0].strip())
            user_in.append(i.partition(':')[2].split()[0])
    if papel['VERB']: print(user_in)

    if user_op != list(default.keys()): raise Exception('wena')

    for i in range(len(user_in)):
        if user_in[i] == '#':
            user_in[i] = list(default.values())[i]
        if user_in[i].isalpha() == False:
            user_in[i] = float(user_in[i])
    for i in range(len(user_in)):
        if i in [10, 16, 17, 20]: continue
        user_in[i] = int(user_in[i])
    if papel['VERB']: print(user_in)
    #print(user_op)

    for i, j in enumerate(papel['input']):
        papel['input'][j] = user_in[i]
    
    if papel['input']['n_pisos'] < 2:
        raise Exception('numero de pisos no puede ser inferior a 2!!')
    elif papel['input']['n_vanos'] < 2:
        raise Exception('numero de vanos no puede ser inferior a 2!!')
    elif papel['input']['n_marcos'] < 3:
        raise Exception('numero de marcos no puede ser inferior a 3!!')
    return

#lectura()

class Grilla():
    """Base para definir las dimensiones una estructura."""
    def __init__(grid):
        grid.grilla = grid.build()[0]
        grid.grilla3 = grid.build()
        grid.dimXY = grid.dimensions()
        grid.dim = {'x': grid.dimXY[1], 'y': grid.dimXY[0]}
        grid.nodos = grid.nodes()
        grid.max_barras = (grid.dim['x'] - 1)*grid.dim['y'] \
                        + grid.dim['y']*grid.dim['x']
        grid.miembro = []
        grid.nivel = {i: [] for i in range(1, grid.dim['y'] + 1)}
        grid.K = zeros((3*grid.nodos, 3*grid.nodos)) # rigidez del sistema
        grid.drift = {}
        grid.datos = {
            'sismo': {
                'I': None,
                'A0': None,
                'paramS': [],
                'C': [],
                'pesoP': None,
                'Q_L': None,
                'pesoL': None,
                'prcL': None,
                'Psism': None,
                'Tn': None,
                'Q0': None,
                'E': [],
            },
            'perfiles': [],
        }
        grid.annainfo = {}
        grid.BLC_gen = ['D', 'L', 'E']
        grid.combo_gen = [
            '1.4 D',
            '1.2 D + 1.6 L',
            '1.2 D + L',
            'E',
            '- E',
            '1.2 D + 1.4 E + L',
            '1.2 D - 1.4 E + L',
            '0.9 D + 1.4 E',
            '0.9 D - 1.4 E',
        ]
        grid.caso = {}
        grid.combo = {}
        grid.r = {} # desplazamientos
        return
    def clean(grid):
        grid.annainfo = {
            'esfuerzos': {
                'head': [
                    ['perfil', 'combo', 'nodo', 'axial', 'corte', 'momento'],
                    ['[kg]', '[kg]', '[kg-m]']
                ],
                'body': []
            },
            'delta-nodos': {
                'head': [
                    ['combo', 'nodo', 'X', 'Y', 'Giro'],
                    ['[mm]', '[mm]', '[rad]']
                ],
                'body': []
            },
            'drifts': {
                'head': [
                    ['combo', 'nivel', 'δmax', 'δmin', 'drift'],
                    ['[mm]', '[mm]', '[‰]']
                ],
                'body': []
            }
        }
        return
    def dimensions(grid):
        """Retorna la tupla (nodos en Y, nodos en X)."""
        return len(grid.grilla), len(grid.grilla[0])
    def nodes(grid):
        """Retorna la cantidad de nodos."""
        return grid.dimXY[0]*grid.dimXY[1]
    def build(grid):
        """Construye la grilla del plano sin dimensiones."""
        nodos = papel['input']['n_vanos'] + 1
        niveles = papel['input']['n_pisos']
        planos = papel['input']['n_marcos']
        tiger, j = [], 0
        for marco in range(planos):
            tiger.append([])
            for nivel in range(niveles):
                tiger[marco].append([])
                for nodo in range(nodos):
                    j += 1
                    tiger[marco][nivel].append(j)
        return tiger
    def maxElemets(grid):
        return
    def dibujar(grid):
        n_pisos = papel['input']['n_pisos']
        n_vanos = papel['input']['n_vanos']
        n_marcos = papel['input']['n_marcos']
        h_pisos = papel['input']['h_pisos']
        b_losas = papel['input']['b_losas']
        d_losas = papel['input']['d_losas']
        formato = papel['config']['format']
        
        marco2, ejes2 = pete.subplots(1, 1, figsize=(8.0,4.0))
        ejes2.set_xticks([i*b_losas for i in range(n_vanos + 1)])
        ejes2.set_yticks([i*h_pisos for i in range(n_pisos + 1)])
        for m in grid.miembro:
            if not m: continue
            ejes2.plot([m.x0, m.x1], [m.y0, m.y1], 'k', linewidth=2)
        marco2.savefig(papel['proyecto'] + f'/src/marco_2D.{formato}')
        pete.close()
        
        marco3 = pete.figure()
        ejes3 = pete.axes(projection='3d')
        ejes3.grid(False)
        ejes3.set_xticks([i*b_losas/100 for i in range(n_vanos + 1)])
        ejes3.set_yticks([i*d_losas/100 for i in range(n_marcos)])
        ejes3.set_zticks([])
        z = 0
        for marco in range(1, n_marcos + 1):
            for m in grid.miembro:
                if not m: continue
                if m.name == 'COL' and marco < n_marcos:
                    ejes3.plot3D([m.x1/100, m.x1/100], [z/100, (z + d_losas)/100], [m.y1/100, m.y1/100], 'r', linewidth=0.5)
                if marco == n_marcos - 1:
                    ejes3.plot3D([m.x0/100, m.x1/100], [z/100, z/100], [m.y0/100, m.y1/100], 'k', linewidth=2)
                else:
                    ejes3.plot3D([m.x0/100, m.x1/100], [z/100, z/100], [m.y0/100, m.y1/100], 'b', linewidth=0.5)
            z += d_losas
        ejes3.view_init(3, -45)
        marco3.savefig(papel['proyecto'] + f'/src/marco_vista_a.{formato}')
        ejes3.view_init(6, -45)
        marco3.savefig(papel['proyecto'] + f'/src/marco_vista_b.{formato}')
        ejes3.view_init(12, -12)
        marco3.savefig(papel['proyecto'] + f'/src/marco_vista_c.{formato}')
        ejes3.view_init(60, -72)
        marco3.savefig(papel['proyecto'] + f'/src/marco_vista_d.{formato}')
        pete.close()
        return

#struct = Grilla()

# definicion de elementos

class Barra():
    # agregar rigidos
    def __init__(bar, nodos, i, j, x0, y0, x1, y1, b, h, ϰ, 𝜈, 𝛾, fc):
        """Define los parametros del perfil y material de un elemento 'barra'."""
        
        #### ## input ## ####
        bar.i = i # nodo A
        bar.j = j # nodo B
        bar.x0 = x0 # [cm]
        bar.y0 = y0 # [cm]
        bar.x1 = x1 # [cm]
        bar.y1 = y1 # [cm]
        bar.b = b # [cm]
        bar.h = h # [cm]
        bar.ϰ = ϰ # factor de forma de la seccion
        bar.𝜈 = 𝜈 # modulo de poisson
        bar.𝛾 = 𝛾/100**3 # [kg/m3] -> [kg/cm3] - densidad
        bar.fc = fc # [MPa] - resistencia en compresion
        
        if x0 != x1:
            bar.name = 'VIGA'
        else:
            bar.name = 'COL'

        #### ## internals ## ####
        bar.L = ((x1 - x0)**2 + (y1 - y0)**2)**0.5 # [cm]
        bar.s = (y1 - y0)/bar.L # senos
        bar.c = (x1 - x0)/bar.L # coseno
        bar.A = b*h # [cm2]
        bar.I = b*h**3/12 # [cm4]
        bar.E = 𝛾**1.5*0.043*fc**0.5/9.80665/0.01 # [kg/mm2]/0.01 = [kg/cm2]
        bar.G = bar.E/(2*(1+𝜈)) # modulo de cizalle
        bar.β = 6*bar.E*bar.I*ϰ/(bar.G*bar.A*bar.L**2)

        bar.GdL( # grados de libertad
            i,
            j,
            (papel['input']['n_vanos'] + 1)*papel['input']['n_pisos']
        )
        bar.rigidez_local(bar.L, bar.A, bar.I, bar.E, bar.β)
        bar.compatibilidad_geometrica(bar.s, bar.c, bar.L)

        bar.K = bar.a.T@bar.k@bar.a # rigidez global
        
        bar.ID = None
        bar.nivel = None
        bar.hayPP = False # indexador caso PesoPropio - actualizar en el futuro
        bar.losa = {'D': [], 'L': []} # indexador casos losa
        bar.dst_loads = [] # (a, b, [kg/cm]) - cargas distribuidas
        bar.jnt_loads = [] # [kg] [kg-cm] - cargas nodales
        bar.caso = {} # suma de todas las cargas basicas por cada caso
        bar.combo = {} # vectores de cada combinacion de casos
        bar.q_caso = {}
        bar.q_combo = {}

        bar.r = {} # desplazamientos
        bar.ε = {} # deformaciones
        bar.σ = {} # esfuerzos en coordenadas locales [Ma, Mb, F]
        bar.V = {} # esfuerzo de corte
        bar.sigma = {} # esfuerzos en coordenadas globales [Na, Va, Ma, Nb, Vb, Mb]
        bar.resumen = {} # acumula en un vector por combinacion todas las fuerzas en ij
        return
            
    def GdL(bar, a, b, nodosEstruc, tipo=''):
        """Prepara la matriz de transformacion de cada barra."""
        bar.T = zeros([6, 3*nodosEstruc])
        if a > nodosEstruc or b > nodosEstruc:
            print('index error')
            return
        if a < 0 or b < 0:
            print('index error')
            return
        if a != 0:
            bar.T[0, 3*a - 3] = 1
            bar.T[1, 3*a - 2] = 1
            bar.T[2, 3*a - 1] = 1
        if b != 0:
            bar.T[3, 3*b - 3] = 1
            bar.T[4, 3*b - 2] = 1
            bar.T[5, 3*b - 1] = 1
        return

    def compatibilidad_geometrica(bar, s, c, L):
        bar.a = array([
            [-s/L, c/L, 1,  s/L, -c/L, 0],
            [-s/L, c/L, 0,  s/L, -c/L, 1],
            [ c,   s,   0, -c,   -s,   0]
        ])
        return

    def rigidez_local(bar, L, A, I, E, β):
        k11 = 2*E*I*(2 + β)/(L*(1 + 2*β))
        k12 = 2*E*I*(1 - β)/(L*(1 + 2*β))
        k21 = k12
        k22 = k11
        k33 = A*E/L
        bar.k = array([
            [k11, k12, 0  ],
            [k21, k22, 0  ],
            [0,   0,   k33]
        ])
        return

    def addDistF(bar, a, b, fa, fb, coord='Y'):
        """Ingresa una fuerza distribuida desde 'a' hasta 'b'."""
        pass

    def sigma(bar, r_global): # sin uso - borrar
        # σ = k@ε
        # ε = a@r_i
        # r_i = T@r
        # σ = k@a@T@r
        bar.σ = bar.k@bar.a@bar.T@r_global # esfuerzos
        
    def dist2joint(bar): pass

    def calcular_peso_propio(bar):
        """Devuelve el peso propio de la barra en kg/cm. La tupla\
        contiene tramo inicial y final donde la carga no es constante."""
        𝛾 = bar.𝛾
        b = bar.b
        h = bar.h
        bar.dst_loads.append(('_pp-bar', 'D', (0, 0, 𝛾*b*h)))
        return
    
    def case_gen(bar, accion='generar'):
        """Genera las categorias de cargas basicas de la barra."""
        if accion == 'generar':
            case_keys = ['D', 'L']
            bar.q_caso = {} # rect dist loads
            for i in case_keys: bar.q_caso.update({i: 0})
            bar.caso = {} # joint loads
            for i in case_keys: bar.caso.update({i: zeros(6)})
        return
    
    def update(bar):
        bar.I = bar.b*bar.h**3/12
        bar.A = bar.b*bar.h
        bar.rigidez_local(bar.L, bar.A, bar.I, bar.E, bar.β)
        bar.K = bar.a.T@bar.k@bar.a # rigidez global
        return
    
    #ned
    # despues agregar tramos rigidos

# definicion geometrica de cada elemento

def armar_estructura():
    """Genera todos los elementos 'barra' de la estructura."""

    NODOS = struct.nodos
    b_COL = papel['input']['b_COL']
    h_COL = papel['input']['b_COL']
    b_VIG = papel['input']['b_VIG']
    h_VIG = papel['input']['h_VIG']
    ϰ = 0 # 1.2
    𝜈 = papel['input']['𝜈']
    𝛾 = papel['input']['𝛾']
    fc = papel['input']['fc']

    barras = []
    barras.append( Barra(NODOS, 0, 1, 0, 0, 0, 300, b_COL, h_COL, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 0, 2, 700, 0, 700, 300, b_COL, h_COL, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 0, 3, 1400, 0, 1400, 300, b_COL, h_COL, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 0, 4, 2100, 0, 2100, 300, b_COL, h_COL, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 1, 5, 0, 300, 0, 600, b_COL, h_COL, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 2, 6, 700, 300, 700, 600, b_COL, h_COL, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 3, 7, 1400, 300, 1400, 600, b_COL, h_COL, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 4, 8, 2100, 300, 2100, 600, b_COL, h_COL, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 5, 9, 0, 600, 0, 900, b_COL, h_COL, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 6, 10, 700, 600, 700, 900, b_COL, h_COL, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 7, 11, 1400, 600, 1400, 900, b_COL, h_COL, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 8, 12, 2100, 600, 2100, 900, b_COL, h_COL, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 9, 13, 0, 900, 0, 1200, b_COL, h_COL, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 10, 14, 700, 900, 700, 1200, b_COL, h_COL, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 11, 15, 1400, 900, 1400, 1200, b_COL, h_COL, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 12, 16, 2100, 900, 2100, 1200, b_COL, h_COL, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 1, 2, 0, 300, 700, 300, b_VIG, h_VIG, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 2, 3, 700, 300, 1400, 300, b_VIG, h_VIG, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 3, 4, 1400, 300, 2100, 300, b_VIG, h_VIG, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 5, 6, 0, 600, 700, 600, b_VIG, h_VIG, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 6, 7, 700, 600, 1400, 600, b_VIG, h_VIG, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 7, 8, 1400, 600, 2100, 600, b_VIG, h_VIG, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 9, 10, 0, 900, 700, 900, b_VIG, h_VIG, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 10, 11, 700, 900, 1400, 900, b_VIG, h_VIG, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 11, 12, 1400, 900, 2100, 900, b_VIG, h_VIG, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 13, 14, 0, 1200, 700, 1200, b_VIG, h_VIG, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 14, 15, 700, 1200, 1400, 1200, b_VIG, h_VIG, ϰ, 𝜈, 𝛾, fc) )
    barras.append( Barra(NODOS, 15, 16, 1400, 1200, 2100, 1200, b_VIG, h_VIG, ϰ, 𝜈, 𝛾, fc) )

    cols, vigs = 0, 0
    nivel = 1
    for m in barras:
        if m.name == 'COL':
            cols += 1
            m.nivel = nivel
            struct.nivel[nivel].append(m)
        if cols == struct.dim['x']:
            cols = 0
            nivel += 1
    nivel = 1
    for m in barras:
        if m.name == 'VIGA':
            vigs += 1
            m.nivel = nivel
            struct.nivel[nivel].append(m)
        if vigs == struct.dim['x'] - 1:
            vigs = 0
            nivel += 1
    for i, m in enumerate(barras, 1):
        m.ID = i

    return barras

# struct.miembro = [None] + armar_estructura()

#print(struct.miembro[1].a)

# K del sistema
def update():
    struct.K = zeros((3*struct.nodos, 3*struct.nodos))
    for m in struct.miembro:
        if m == None: continue
        m.update()
        struct.K += m.T.T@m.K@m.T
    return

def h_trib_losa(L, d, c_borde, neg=False): #actualizar descripcion
    """Retorna la proyeccion de los lados de un trapecio sobre su\
    base y el valor del tramo constante de la carga distribuida 'q'.
    L       = Largo de la base de la losa planta (viga del marco en\
    analisis)
    d       = Distancia entre marcos
    t       = Espesor de la losa
    𝛾       = Densidad del hormigon
    c_borde = Condicion de borde
    neg     = Indica si las proyecciones 'a' y 'b' se deben voltear
    Por defecto se calcula siempre 'a' adyacente a un angulo de 45\
    grados cuando es posible."""
    if c_borde == 1: pass
    elif c_borde == '2':
        if d <= L:
            h = d/2**0.5*sin(60/180*π)/sin(75/180*π)
        else:
            h = L/2**0.5*sin(60/180*π)/sin(75/180*π)
        a = h
        b = a*2**0.5*sin(75/180*π)/sin(60/180*π) - a
    elif c_borde == '3c':
        k = 2/2**0.5*sin(60/180*π)/sin(75/180*π)
        if d <= k*L:
            h = d/2
        else:
            h = k*L/2
        a = h
        b = a*2**0.5*sin(75/180*π)/sin(60/180*π) - a
    elif c_borde == '3u':
        k = 2**0.5/2*sin(75/180*π)/sin(60/180*π)
        if d <= k*L:
            h = d/2**0.5*sin(60/180*π)/sin(75/180*π)
        else:
            h = L/2
        a = b = h
    elif c_borde == '4' or c_borde == '0':
        print('estoy aca')
        if d <= L:
            h = d/2
        else:
            h = L/2
        a = b = h
    if neg: a, b = b, a
    return [a, b, h]

#print(h_trib_losa(451.1, 355.77, '3c', 1))

def ep_rec(L, q, a, b, d):
    """Calculo de las reacciones en empotramiento perfecto para una carga constante.
    L = Largo del elemento
    q = Magnitud de la carga distribuida
    a = Distancia desde el inicio del elemento hasta la mitad de la aplicacion de la carga
    b = Distancia desde el final del elemento hasta la mitad de la aplicacion de la carga
    d = Largo de la aplicacion de la carga"""
    Ra = q*d/L**3*((2*a + L)*b**2 + (a - b)/4*d**2)
    Rb = q*d/L**3*((2*b + L)*a**2 - (a - b)/4*d**2)
    Ma = q*d/L**2*(a*b**2 + d**2/12*(L - 3*b))
    Mb = -q*d/L**2*(a**2*b + d**2/12*(L - 3*a))
    if papel['VERB']:
        print(f'(q constante) Ra: {Ra:.1f}, Ma: {Ma:.1f}, Rb: {Rb:.1f}, Mb: {Mb:.1f}')
    return Ra, Ma, Rb, Mb

# ep_rec(2, 1000, 0.5 + 0.6, 0.9, 1.2);

def ep_tri(L, q, a):
    """Calculo de las reacciones en empotramiento perfecto para una carga lineal.
    L = Largo del elemento
    q = Magnitud final de la carga distribuida lineal
    a = Distancia desde el inicio del elemento hasta el final de la aplicacion de la carga"""
    Ra = q*a/20*(10 - a**2/L**2*(15 - 8*a/L))
    Rb = q*a**3/(20*L**2)*(15-8*a/L)
    Ma = q*a**2/30*(10 - a/L*(15 - 6*a/L))
    Mb = -q*a**3/(20*L)*(5 - 4*a/L)
    if papel['VERB']:
        print(f'(q lineal) Ra: {Ra:.1f}, Ma: {Ma:.1f}, Rb: {Rb:.1f}, Mb: {Mb:.1f}')
    return Ra, Ma, Rb, Mb

# ep_tri(2, 1000, 0.5);

# superposicion de casos de carga de una viga cualquiera respecto del eje Y

def emp_perf_Y(L, a, b, q):
    """Calcula las reacciones de empotramiento perfecto de una viga dada sometida a una carga trapezoidal.
    L = Largo del elemento
    a = Distancia desde el inicio del elemento hasta el inicio de la carga constante
    b = Distancia desde el final del elemento hasta el final de la carga costante
    q = Magnitud de la carga distribuida trapezoidal"""
    r = [[],[],[]]
    joints = []
    d = L - a - b

    *r[0], = ep_tri(L, q, a)
    *r[1], = ep_rec(L, q, a + d/2, b + d/2, d)
    *r[2], = ep_tri(L, q, b)

    r[2][0], r[2][2] = r[2][2], r[2][0]
    r[2][1], r[2][3] = -r[2][3], -r[2][1]

    for i in range(4):
        joints.append(0)
        for reaction in r:
            joints[i] += reaction[i]

    if papel['VERB']:
        print(
            'Va:', round(joints[0], 2),
            '\nMa:', round(joints[1], 2),
            '\nVb:', round(joints[2], 2),
            '\nMb:', round(joints[3], 2),
        )
        line()
    return (0, *joints[0:2], 0, *joints[2:4])

#emp_perf_Y(3, 0.75, 1.4, 300)

# cargas muertas - peso propio

def pp_barras(accion='cargar'):
    """Aplica o limpia las cargas distribuidas y reacciones en los nodos producto del peso propio de todas las barras."""
    if accion == 'limpiar':
        for m in struct.miembro:
            if not m: continue
            if not m.hayPP:
                if papel['VERB']: print('no habia nada que eliminar!')
            else:
                for i, basicLoad in enumerate(m.dst_loads):
                    if basicLoad[0] == '_pp-bar':
                        m.dst_loads.pop(i)
                        m.jnt_loads.pop(i)
                        break
                if papel['VERB']: print('peso propio eliminado.')
                m.hayPP = False
    elif accion == 'cargar':
        for m in struct.miembro:
            if not m: continue
            if m.hayPP: # actualizar PP para borrar el for
                if papel['VERB']: print('ya estaba definido!')
            else:
                m.calcular_peso_propio()
                if m.name == 'VIGA':
                    for i, basicLoad in enumerate(m.dst_loads):
                        if basicLoad[0] == '_pp-bar':
                            m.jnt_loads.append((
                                '_pp-bar',
                                'D',
                                array(emp_perf_Y(
                                    m.L, *m.dst_loads[i][2]
                                ))
                            ))
                            break
                elif m.name == 'COL':
                    Na = m.dst_loads[0][2][2]*m.L/2
                    m.jnt_loads.append(
                        ('_pp-bar', 'D', array([0, Na, 0, 0, Na, 0]))
                    )
                m.hayPP = True
    return

# pp_barras('cargar')
#print(struct.miembro[1].dst_loads)
#print(struct.miembro[2].dst_loads)
#print(struct.miembro[3].dst_loads)
#struct.miembro[2].jnt_loads[0][2]@struct.miembro[2].T

# cargas muertas y vivas - descarga losa

def q_losa(m, forma='', caso='D', accion='cargar', voltear=''): # m y returns temporales
    """Aplica o elimina las cargas distribuidas y las reacciones en los nodos a todas las vigas."""
    
    d_losas = papel['input']['d_losas'] # [cm]
    t_losas = papel['input']['t_losas'] # [cm]
    Q_L = papel['input']['Q_L']/100**2 # [kg/cm2]
    
    if accion == 'cargar':
        #for m in struct.miembro:
            if not m: return #continue
            if m.name != 'VIGA': return #continue
            if caso != 'D' and caso != 'L':
                print('no trollees! que me crasheo.')
                return #break
            else:
                new_index = len(m.dst_loads)
                m.losa[caso].append(new_index)
                if voltear == 'mirror':
                    q_losa = h_trib_losa(m.L, d_losas, forma, True) # [a, b, h]
                else:
                    q_losa = h_trib_losa(m.L, d_losas, forma) # [a, b, h]
                q_losa[2] *= Q_L if caso == 'L' else t_losas*m.𝛾 # [a, b, q]
                # rectificacion
                k = (2*m.L - q_losa[0] - q_losa[1])/(2*m.L)
                if papel['VERB']:
                    print('L', m.L, 'a', q_losa[0], 'b', q_losa[1])
                    print('k', k)
                    print('q_losa antes', q_losa)
                q_losa = [0, 0, k*q_losa[2]]
                if papel['VERB']: print('q_losa despues', q_losa)
                m.dst_loads.append((
                    '_q-losa',
                    caso,
                    tuple(q_losa)
                ))
                m.jnt_loads.append((
                    '_q-losa',
                    caso,
                    array( emp_perf_Y(m.L, *m.dst_loads[m.losa[caso][-1]][2]) )
                ))
                if papel['VERB']:
                    print(f'se ha cargado la losa (caso: {caso}) en el elemento {m.name} (i):{m.i}.')
                    line()
                    line()
    elif accion == 'l': # limpiar... pero algun dia
        print('nada que ver aqui :v *se crashea*')
        raise
    return

def cargar_losas(): # !! no sobrescribe
    for caso in ['D', 'L']:
        q_losa(struct.miembro[17], '2', caso, 'cargar', 'mirror') # piso 1
        q_losa(struct.miembro[17], '3c', caso, 'cargar', 'mirror')
        q_losa(struct.miembro[18], '3u', caso)
        q_losa(struct.miembro[18], '3u', caso)
        q_losa(struct.miembro[19], '2', caso)
        q_losa(struct.miembro[19], '3c', caso)
        q_losa(struct.miembro[20], '2', caso, 'cargar', 'mirror') # piso 2
        q_losa(struct.miembro[20], '3c', caso, 'cargar', 'mirror')
        q_losa(struct.miembro[21], '3u', caso)
        q_losa(struct.miembro[21], '3u', caso)
        q_losa(struct.miembro[22], '2', caso)
        q_losa(struct.miembro[22], '3c', caso)
        q_losa(struct.miembro[23], '2', caso, 'cargar', 'mirror') # piso 3
        q_losa(struct.miembro[23], '3c', caso, 'cargar', 'mirror')
        q_losa(struct.miembro[24], '3u', caso)
        q_losa(struct.miembro[24], '3u', caso)
        q_losa(struct.miembro[25], '2', caso)
        q_losa(struct.miembro[25], '3c', caso)
        q_losa(struct.miembro[26], '2', caso, 'cargar', 'mirror') # piso 4
        q_losa(struct.miembro[26], '3c', caso, 'cargar', 'mirror')
        q_losa(struct.miembro[27], '3u', caso)
        q_losa(struct.miembro[27], '3u', caso)
        q_losa(struct.miembro[28], '2', caso)
        q_losa(struct.miembro[28], '3c', caso)
    if papel['VERB']:
        [print(m.dst_loads) for m in struct.miembro if m];
        [print(m.jnt_loads) for m in struct.miembro if m];
    return

# cargar_losas()

# cargas sismicas - metodo estatico
def metodo_estatico():
    """Calcula un vector con las fuerzas horizontales producto del sismo."""
    
    n_pisos = papel['input']['n_pisos']
    n_vanos = papel['input']['n_vanos']
    n_marcos = papel['input']['n_marcos']
    h_pisos = papel['input']['h_pisos']/100 # [m]
    b_losas = papel['input']['b_losas'] # [cm]
    d_losas = papel['input']['d_losas'] # [cm]
    t_losas = papel['input']['t_losas'] # [cm]
    zona = papel['input']['zona']
    suelo = papel['input']['suelo']
    categoria = papel['input']['categoria']
    R = papel['input']['R']
    Q_L = papel['input']['Q_L']/100**2 # [kg/cm2]
    prcL = papel['input']['prcL']
    
    TABLA_I = { # Tabla 6.1
        # "CategoriaDeEdificio": I [factor de importancia],
        "I"  : 0.6,
        "II" : 1.0,
        "III": 1.2,
        "IV" : 1.2,
    }

    TABLA_A0 = { # Tabla 6.2
        # "ZonaSismica": A0/g [aceleracion efectiva],
        1: 0.20,
        2: 0.30,
        3: 0.40,
    }

    TABLA_PARAMETROS_SUELO = { # Tabla 6.3
        # "TipoDeSuelo": (S, T0, Tp [T'], n, p),
        "A": (0.90, 0.15, 0.20, 1.00, 2.0),
        "B": (1.00, 0.30, 0.35, 1.33, 1.5),
        "C": (1.05, 0.40, 0.45, 1.40, 1.6),
        "D": (1.20, 0.75, 0.85, 1.80, 1.0),
        "E": (1.30, 1.20, 1.35, 1.80, 1.0),
        "F": (None, None, None, None, None),
    }

    TABLA_C_MAX = { # Tabla 6.4
        # "R": C_max,
        2  : 0.90,
        3  : 0.60,
        4  : 0.55,
        5.5: 0.40,
        6  : 0.35,
        7  : 0.35,
    }

    Tn = round(0.09*h_pisos*n_pisos/(d_losas/100*n_vanos)**0.5, 4)

    I = TABLA_I[categoria]
    A0 = TABLA_A0[zona]
    S, T0, Tp, n, p = TABLA_PARAMETROS_SUELO[suelo]

    # coeficiente sismico
    Cmax = round(TABLA_C_MAX[R]*S*A0, 8)
    Cmin = A0*S/6
    C = 2.75*S*A0/R*(Tp/Tn)**n

    if C < Cmin: C = Cmin
    elif C > Cmax: C = Cmax

    # numero de elementos por piso
    n_viga = n_marcos*n_vanos
    n_colum = n_marcos*(n_vanos + 1)
    n_vig_u = (n_marcos - 1)*(n_vanos + 1) # viga de union de marcos

    # masa por elemento
    for m in struct.miembro:
        if not m: continue
        if m.name == 'VIGA':
            m_viga = m.𝛾*m.A*m.L
            if papel['VERB']: print('vig: gamma, area, largo', m.𝛾, m.A, m.L)
            m_vig_u = m.𝛾*m.A*d_losas
            break
    for m in struct.miembro:
        if not m: continue
        if m.name == 'COL':
            m_colum = m.𝛾*m.A*m.L
            if papel['VERB']: print('col: gamma, area, largo', m.𝛾, m.A, m.L)
            break

    if papel['VERB']:
        print('numero de v, v_u, c:', n_viga, n_vig_u, n_colum)
        print('masa de v, v_u, c:', m_viga, m_vig_u, m_colum)

    # masa estructura por nivel
    M_st = (n_viga*m_viga) + (n_colum*m_colum) + (n_vig_u*m_vig_u)
    if papel['VERB']:
        print(f'n y m: viga {n_viga} de {m_viga}, col {n_colum} de {m_colum}, viga_un {n_vig_u} de {m_vig_u}')
    # masa losas por nivel
    M_l = m.𝛾 * b_losas*d_losas*t_losas * n_vanos*(n_marcos - 1)

    # masa sismica por nivel
    MUERTO = M_st + M_l # [kg]
    VIVO = 0.25*Q_L * b_losas*n_vanos * d_losas*(n_marcos - 1) # [kg]
    Pk = [0]
    for nivel in range(n_pisos):
        if nivel == n_pisos - 1:
            Pk.append(MUERTO - n_colum*m_colum/2 + VIVO)
            break
        Pk.append(MUERTO + VIVO)
    if papel['VERB']: print('masas por nivel:', Pk, 'largo pk:', len(Pk))

    P = sum(Pk) # [kg] - peso sismico
    struct.datos['sismo']['Psism'] = round(P/1000, 2)
    Q0 = C*I*P # [kg] - corte basal

    if papel['VERB']:
        print(
            'testing vars:\n',
            f'A0: {A0}, I: {I}, S: {S}, T0: {T0}\n',
            f'Tp: {Tp}, n: {n}, p: {p}\n',
            f'Cmax: {Cmax}, Cmin: {round(Cmin, 4)}, C: {round(C, 4)}\n',
            f'Q0: {Q0}, Pk: {Pk}, P: {P}',
            f'\nperiodo fundamental: {Tn}'
        ) # borrame
        print('masas: M_st {}, M losa {}'.format(M_st, M_l))
        print('muerto: {}, vivo: {}'.format(MUERTO, VIVO))

    H = round(h_pisos*n_pisos, 4)
    Z = [0] + [round(i*h_pisos, 4) for i in range(1, n_pisos + 1)]
    A = [0]
    for k, _ in enumerate(Z):
        if k == 0: continue
        A.append(
            (1 - Z[k - 1]/H)**0.5 - (1 - Z[k]/H)**0.5
        )
        if papel['VERB']:
            print('nivel: {}, altura acumulada: {}, Ak: {}'.format(k, Z[k], round(A[k], 3)))

    P = Pk # mejor coincidencia con formulas de la nch433
    del Pk
    if papel['VERB']: print('largos:', len(A), len(P))
    F = []
    AjPj = sum([i*k for i, k in zip(A, P)]) # Σ Aj*Pj
    for k, _ in enumerate(P):
        F.append(A[k]*P[k]/AjPj*Q0)
        if papel['VERB']: print('AP/AP:', A[k]*P[k]/AjPj)

    if papel['VERB']:
        print('A*P', AjPj) # borrame
        print('F por nivel:', F) # borrame

    fx = []
    for k, _ in enumerate(F):
        if k == 0: continue
        for i in range(n_vanos + 1):
            fx.append(F[k]/((n_vanos + 1)*(n_marcos)))
            fx.append(0)
            fx.append(0)
    
    struct.datos['sismo']['I'] = I
    struct.datos['sismo']['A0'] = A0
    struct.datos['sismo']['paramS'] = [S, T0, Tp, n, p]
    struct.datos['sismo']['C'] = [round(Cmin, 3), round(C, 3), round(Cmax, 3)]
    #struct.datos['sismo']['pesoP'] = trucoteca
    struct.datos['sismo']['Q_L'] = papel['input']['Q_L']
    struct.datos['sismo']['pesoL'] = round(VIVO*n_pisos/1000, 2)
    struct.datos['sismo']['pesoP'] = round(struct.datos['sismo']['Psism'] - struct.datos['sismo']['pesoL'], 2)
    struct.datos['sismo']['prcL'] = int(prcL*100)
    #struct.datos['sismo']['Psism'] = stolen above
    struct.datos['sismo']['Tn'] = int(Tn*100)/100
    struct.datos['sismo']['Q0'] = round(Q0/1000, 2)
    _, *struct.datos['sismo']['E'] = [round(i/1000, 2) for i in F]

    return array(fx).T

#metodo_estatico()

def load_case_gen(): # convertir en metodo (?)
    """Genera las categorias y combinaciones de carga de la estructura."""
    
    #if accion == 'generar':
    struct.caso.clear()
    struct.combo.clear()
    for caso in struct.BLC_gen:
        struct.caso.update({ caso: zeros(len(struct.K)) })
    for m in struct.miembro:
        if not m: continue
        if 1: m.q_combo.clear() # temp carga rectangular
        m.combo.clear()
    for combo in struct.combo_gen:
        struct.combo.update({ combo: zeros(len(struct.K)) })
        for m in struct.miembro:
            if not m or combo == 'E' or combo == '- E': continue
            if 1:
                m.q_combo.update({ combo: 0 }) # temp carga rectangular
            m.combo.update({ combo: zeros(len(struct.K)) })
    
    #elif accion == 'cargar':
    for i, m in enumerate(struct.miembro):
        if not m: continue
        m.case_gen()
        for F in m.jnt_loads:
            if papel['VERB'] and 0: # detalle de casos de carga
                print(
                    f'Fuente: miembro {i}',
                    f'perfil: {m.name}',
                    f'nombre: \'{F[0]}\'',
                    f'caso: \'{F[1]}\'',
                    f'vector: {F[2]}\n',
                    sep='\n'
                )
            if F[1] == 'D':
                m.caso['D'] += F[2]
                struct.caso['D'] -= F[2]@m.T
            elif F[1] == 'L':
                m.caso['L'] += F[2]
                struct.caso['L'] -= F[2]@m.T
        if 1: # temp rect
            for F in m.dst_loads:
                if F[1] == 'D':
                    m.q_caso['D'] += F[2][2]
                if F[1] == 'L':
                    m.q_caso['L'] += F[2][2]
    if papel['VERB'] and 0:
        for i, m in enumerate(struct.miembro): # resumen de casos de carga
            if m:
                print(f'\nperfil \'{m.name}{i:2}\'', m.caso, sep='\n')

    if 1: # temp
        for m in struct.miembro:
            if not m: continue
            # cargas rectangulares por combinacion
            m.q_combo['1.4 D'] = \
                1.4*m.q_caso['D']

            m.q_combo['1.2 D + 1.6 L'] = \
                1.2*m.q_caso['D'] + 1.6*m.q_caso['L']

            m.q_combo['1.2 D + L'] = \
                1.2*m.q_caso['D'] + m.q_caso['L']

            m.q_combo['1.2 D + 1.4 E + L'] = \
                1.2*m.q_caso['D'] + m.q_caso['L']

            m.q_combo['1.2 D - 1.4 E + L'] = \
                1.2*m.q_caso['D'] + m.q_caso['L']

            m.q_combo['0.9 D + 1.4 E'] = \
                0.9*m.q_caso['D']

            m.q_combo['0.9 D - 1.4 E'] = \
                0.9*m.q_caso['D']

    for m in struct.miembro:
        if not m: continue
        # vectores de empotramiento perfecto por combinacion
        m.combo['1.4 D'] = \
            1.4*m.caso['D']

        m.combo['1.2 D + 1.6 L'] = \
            1.2*m.caso['D'] + 1.6*m.caso['L']

        m.combo['1.2 D + L'] = \
            1.2*m.caso['D'] + m.caso['L']

        m.combo['1.2 D + 1.4 E + L'] = \
            1.2*m.caso['D'] + m.caso['L']

        m.combo['1.2 D - 1.4 E + L'] = \
            1.2*m.caso['D'] + m.caso['L']

        m.combo['0.9 D + 1.4 E'] = \
            0.9*m.caso['D']

        m.combo['0.9 D - 1.4 E'] = \
            0.9*m.caso['D']

    struct.caso['E'] = metodo_estatico()
    # vectores de fuerza por combinacion
    struct.combo['1.4 D'] = \
        1.4*struct.caso['D']

    struct.combo['1.2 D + 1.6 L'] = \
        1.2*struct.caso['D'] + 1.6*struct.caso['L']

    struct.combo['1.2 D + L'] = \
        1.2*struct.caso['D'] + struct.caso['L']

    struct.combo['E'] = \
        struct.caso['E']

    struct.combo['- E'] = \
        -1*struct.caso['E']

    struct.combo['1.2 D + 1.4 E + L'] = \
        1.2*struct.caso['D'] + 1.4*struct.caso['E'] + struct.caso['L']

    struct.combo['1.2 D - 1.4 E + L'] = \
        1.2*struct.caso['D'] - 1.4*struct.caso['E'] + struct.caso['L']

    struct.combo['0.9 D + 1.4 E'] = \
        0.9*struct.caso['D'] + 1.4*struct.caso['E']

    struct.combo['0.9 D - 1.4 E'] = \
        0.9*struct.caso['D'] - 1.4*struct.caso['E']
    return

# load_case_gen()

def anna():
    struct.r.clear()
    for COMBO in struct.combo.keys():
        struct.r.update({ COMBO: inv(struct.K)@struct.combo[COMBO] }) # desplazamientos

    for m in struct.miembro:
        if not m: continue
        # inicializacion
        m.r.clear()
        m.ε.clear()
        m.σ.clear()
        m.V.clear()
        m.sigma.clear()
        for COMBO in m.combo.keys():
            # definicion
            m.r.update({ COMBO: m.T@struct.r[COMBO] })
            m.ε.update({ COMBO: m.a@m.r[COMBO] })
            m.σ.update({ COMBO: m.k@m.ε[COMBO] })
            m.V.update({ COMBO: (m.σ[COMBO][0] + m.σ[COMBO][1])/m.L })
            m.sigma.update({ COMBO: m.a.T@m.σ[COMBO] })

    if papel['VERB'] and 0:
        for i, m in enumerate(struct.miembro):
            if m:
                print(f'miembro {i}, vector de esfuerzos:\n', m.sigma)
                line()

    for m in struct.miembro:
        if not m: continue
        m.resumen.clear()
        for COMBO in m.sigma.keys():
            m.resumen.update({ COMBO: mround(m.sigma[COMBO] + m.combo[COMBO], 2) })
    return

# anna()

def N(x, L, a, b, q, Na, Va, Ma):
    """Calcula el diagrama de esfuerzo axial de una viga."""
    if 0 <= x <= L:
        return Na
    else:
        return 'out of bar length!'

def V(x, L, a, b, q, Na, Va, Ma): # sera metodo
    """Calcula el diagrama de esfuerzo cortante de una viga."""
    def V1(x): return Va - q*x**2/(2*a)
    def V2(x): return V1(a) - q*(x - a)
    def V3(x): return V2(L - b) - q*(x - L + b)/2*(1 + (L - x)/b)
    
    if 0 <= x < a: return V1(x)
    elif a <= x < L - b: return V2(x)
    elif L - b <= x <= L: return V3(x)
    
    return 'out of bar length!'

def M(x, L, a, b, q, Na, Va, Ma):
    """Calcula el diagrama de momento de una viga."""
    def M1(x): return -Ma + Va*x - q*x**3/(6*a)
    def M2(x): return -Ma + Va*x - q*(a*x/2 - a**2/3) - q*(x - a)**2/2
    def M3(x): return -Ma + Va*x - q*(a*x/2 - a**2/3) - q*(L - b - a)*(x - a - (L - b - a)/2) - q*(x - L + b)**2/3*(1 + (L - x)/(2*b))
    
    if 0 <= x < a: return M1(x)
    elif a <= x < L - b: return M2(x)
    elif L - b <= x <= L: return M3(x)

    return 'out of bar length!'

def grafVV(barra, combo, ij=False, full=False):
    """Grafica los diagramas de esfuerzo interno de cada elemento."""
    formato = papel['config']['format']
    ID = barra.ID
    # q      ( a, b, q )
    q_rect = (
        0.00000001,
        0.00000001,
        barra.q_combo[combo],
    )
    Na = barra.resumen[combo][0]
    Va = barra.resumen[combo][1]
    Ma = barra.resumen[combo][2]
    L = int(barra.L)
    x, y = [], {'N':[],'V':[],'M':[]}
    ijab = {'N':[],'V':[],'M':[]}

    if ij:
        if barra.name == 'VIGA':
            L1 = struct.nivel[barra.nivel][0].b/2
            L2 =  L - struct.nivel[barra.nivel][0].b/2
            ijab['N'].append( N(L1, L, *q_rect, Na, Va, Ma) )
            ijab['V'].append( V(L1, L, *q_rect, Na, Va, Ma) )
            ijab['M'].append( M(L1, L, *q_rect, Na, Va, Ma)/100 )
            ijab['N'].append( N(L2, L, *q_rect, Na, Va, Ma) )
            ijab['V'].append( V(L2, L, *q_rect, Na, Va, Ma) )
            ijab['M'].append( M(L2, L, *q_rect, Na, Va, Ma)/100 )
        elif barra.name == 'COL':
            L1 = struct.nivel[barra.nivel][-1].h/2
            L2 =  L - struct.nivel[barra.nivel][-1].h/2
            ijab['N'].append( V(L1, L, *q_rect, Na, Va, Ma) )
            ijab['V'].append( N(L1, L, *q_rect, Na, Va, Ma)*-1 )
            ijab['M'].append( M(L1, L, *q_rect[0:2], 0, Va, Na*-1, Ma)/100 )
            ijab['N'].append( V(L2, L, *q_rect, Na, Va, Ma) )
            ijab['V'].append( N(L2, L, *q_rect, Na, Va, Ma)*-1 )
            ijab['M'].append( M(L2, L, *q_rect[0:2], 0, Va, Na*-1, Ma)/100 )
        if papel['VERB']:
            if barra.name == 'VIGA' or barra.nivel != 1:
                print(
                    'Extremo i:'
                    f'\nN({L1}):', round( ijab['N'][0], 2 ), '[kg]'
                    f'\nV({L1}):', round( ijab['V'][0], 2 ), '[kg]'
                    f'\nM({L1}):', round( ijab['M'][0], 2 ), '[kg-m]'
                )
            print(
                'Extremo j:'
                f'\nN({L2}):', round( ijab['N'][1], 2 ), '[kg]'
                f'\nV({L2}):', round( ijab['V'][1], 2 ), '[kg]'
                f'\nM({L2}):', round( ijab['M'][1], 2 ), '[kg-m]'
            )
        if barra.name == 'VIGA' or barra.nivel != 1:
            hoja = open(papel['proyecto'] + f'/tmp/miembro{ID} (i) [{combo}].txt', 'w')
            hoja.write(
                'Extremo i:\n' + \
                f'N({L1}): {ijab["N"][0]:.2f} [kg]\n' + \
                f'V({L1}): {ijab["V"][0]:.2f} [kg]\n' + \
                f'M({L1}): {ijab["M"][0]:.2f} [kg-m]\n'
            )
            hoja.close()
        hoja = open(papel['proyecto'] + f'/tmp/miembro{ID} (j) [{combo}].txt', 'w')
        hoja.write(
            'Extremo j:\n' + \
            f'N({L2}): {ijab["N"][1]:.2f} [kg]\n' + \
            f'V({L2}): {ijab["V"][1]:.2f} [kg]\n' + \
            f'M({L2}): {ijab["M"][1]:.2f} [kg-m]\n'
        )
        hoja.close()
    
    if full:
        for _x in range(L + 1):
            x.append(_x)
            y['N'].append(N(_x, L, *q_rect, Na, Va, Ma))
            y['V'].append(V(_x, L, *q_rect, Na, Va, Ma))
        if barra.name == 'VIGA':
            for _x in range(L + 1):
                y['M'].append(M(_x, L, *q_rect, Na, Va, Ma))
        elif barra.name == 'COL':
            for _x in range(L + 1):
                y['M'].append(M(_x, L, *q_rect[0:2], 0, Va, Na*-1, Ma))
        palo = [0 for i in x]
        x, palo = array(x), array(palo)
        if barra.name == 'VIGA':
            y['N'] = array(y['N'])
            y['V'] = array(y['V'])
            y['M'] = array(y['M'])/100
        elif barra.name == 'COL':
            y['N'], y['V'] = array(y['V']), array(y['N'])*-1
            y['M'] = array(y['M'])/100

        figura, ejes = pete.subplots(3, 1, figsize=(8.0,11.0))
        pete.rcParams.update({'font.size': 8.7})
        pete.subplots_adjust(hspace=0.4)
        ejes[0].set_title(f'comb: [{combo}] Miembro {ID}: {barra.name}', y=1.1, fontsize=19)
        for k, nvm in enumerate(['N', 'V', 'M']):
            ejes[k].grid()
            ejes[k].plot(x, y[nvm], 'k--', label=f'${nvm}(x)$')
            if ij:
                if barra.name == 'VIGA' or barra.nivel != 1:
                    ejes[k].plot(L1, ijab[nvm][0], 'ro')
                ejes[k].plot(L2, ijab[nvm][1], 'ro')
                if 0 < min(y[nvm]) <= max(y[nvm]):
                    if barra.name == 'VIGA' or barra.nivel != 1:
                        ejes[k].plot([L1, L1], [0, max(y[nvm])], 'r', linewidth=1)
                    ejes[k].plot([L2, L2], [0, max(y[nvm])], 'r', linewidth=1)
                elif min(y[nvm]) <= max(y[nvm]) < 0:
                    if barra.name == 'VIGA' or barra.nivel != 1:
                        ejes[k].plot([L1, L1], [min(y[nvm]), 0], 'r', linewidth=1)
                    ejes[k].plot([L2, L2], [min(y[nvm]), 0], 'r', linewidth=1)
                else:
                    if barra.name == 'VIGA' or barra.nivel != 1:
                        ejes[k].plot([L1, L1], [min(y[nvm]), max(y[nvm])], 'r', linewidth=1)
                    ejes[k].plot([L2, L2], [min(y[nvm]), max(y[nvm])], 'r', linewidth=1)
            ejes[k].plot(x, palo, 'k-')
            ejes[k].fill_between(x, y[nvm], palo, where=(y[nvm] > palo), facecolor='b', alpha=0.33)
            ejes[k].fill_between(x, y[nvm], palo, where=(y[nvm] < palo), facecolor='g', alpha=0.33)
            ejes[k].legend(loc='best', fontsize=15);
            ejes[k].set_xlabel(f'$L(x)$ [cm]', fontsize=14)
            if nvm == 'M':
                ejes[k].set_ylabel(f'${nvm}(x)$ [kg-m]', fontsize=14)
                ejes[k].invert_yaxis()
            else:
                ejes[k].set_ylabel(f'${nvm}(x)$ [kg]', fontsize=14)
        #figura.show()
        pete.close() # no graph in notebook
        figura.savefig(papel['proyecto'] + f'/src/miembro{ID} [{combo}].{formato}')
    return

#grafVV(struct.miembro[17], '1.2 D + 1.4 E + L', 1)

def informacion(resumen):
    """Genera los reportes para esfuerzos en los extremos de cada\
    elemnto, desplazamientos nodales, desplazamientos por nivel y\
    drift por nivel para cada combinacion y les da formato en un\
    archivo por cada tabla."""
    project = papel['proyecto']
    if resumen not in ['generar', 'graficos']:
        tabla = struct.annainfo[resumen]
    if resumen == 'generar':
        h_pisos = papel['input']['h_pisos']
        struct.clean()
        tabla = struct.annainfo['esfuerzos']
        for k, m in enumerate(struct.miembro):
            if not m: continue
            for COMBO in m.resumen:
                if m.name == 'VIGA':
                    tabla['body'].append([
                        m.name,
                        k,
                        COMBO,
                        'i',
                        m.resumen[COMBO][0],
                        m.resumen[COMBO][1],
                        m.resumen[COMBO][2]/100,
                    ])
                    tabla['body'].append([
                        'j',
                        m.resumen[COMBO][3]*-1,
                        m.resumen[COMBO][4]*-1,
                        m.resumen[COMBO][5]/-100,
                    ])
                elif m.name == 'COL':
                    tabla['body'].append([
                        m.name,
                        k,
                        COMBO,
                        'i',
                        m.resumen[COMBO][1],
                        m.resumen[COMBO][0]*-1,
                        m.resumen[COMBO][2]/100,
                    ])
                    tabla['body'].append([
                        'j',
                        m.resumen[COMBO][4]*-1,
                        m.resumen[COMBO][3],
                        m.resumen[COMBO][5]/-100,
                    ])
        tabla = struct.annainfo['delta-nodos']
        for combo in struct.r:
            for nodo in range(struct.nodos):
                tabla['body'].append([
                    combo,
                    nodo + 1,
                    struct.r[combo][3*nodo]*10,
                    struct.r[combo][3*nodo + 1]*10,
                    struct.r[combo][3*nodo + 2],
                ])
        tabla = struct.annainfo['drifts']
        for combo in ['E', '- E']:
            struct.drift.update({ combo: [] })
            δmax1, δmin1, δmax2, δmin2 = 0, 0, 0, 0
            nivel = 0
            deltaX = []
            for i in range(len(struct.r[combo])):
                if 3*i >= len(struct.r[combo]): break
                deltaX.append(struct.r[combo][3*i])
            for i in range(len(deltaX)):
                if struct.dim['x']*i >= len(deltaX): break
                nivel += 1
                for j in deltaX[struct.dim['x']*i:struct.dim['x']*(i + 1)]:
                    if deltaX[struct.dim['x']*i] == j:
                        δmin2 = j
                    if abs(j) > abs(δmax2):
                        δmax2 = j
                    elif abs(j) < abs(δmin2):
                        δmin2 = j
                drift = round((δmax2 - δmin1)/h_pisos*1000, 3) # [‰]
                δmax1, δmin1 = δmax2, δmin2
                tabla['body'].append([
                    combo,
                    nivel,
                    10*δmax2,
                    10*δmin2,
                    drift,
                ])
                struct.drift[combo].append(drift)
        hoja = open(project + '/tmp/' + 'K' + '.txt', 'w')
        for i in struct.K:
            hoja.write(str(i) + '\n')
        hoja.close()
        hoja = open(project + '/tmp/' + 'elementos' + '.txt', 'w')
        for k in struct.nivel:
            for i, miembro in enumerate(struct.nivel[k], 1):
                hoja.write(
                    f'elemento {miembro.ID:02}: {miembro.name:<4} {i} (i: {miembro.i:2}, j: {miembro.j:2}) en nivel {k}\n'
                )
        hoja.close()
        hoja = open(project + '/tmp/' + 'datos' + '.txt', 'w')
        for dato in struct.datos['sismo']:
            hoja.write(repr(dato) + ': ' + str(struct.datos['sismo'][dato]) + '\n')
        hoja.close()
    elif resumen == 'esfuerzos':
        if papel['VERB']:
            print(' '*30 + '{:>19}{:>13}{:>13}'.format(*tabla['head'][1]))
            print('fila  {:^6} {:^19} {:^4}{:>12}{:>13}{:>13}'.format(*tabla['head'][0]))
            print('='*75)
        hoja = open(project + '/tmp/' + resumen + '.txt', 'w', encoding='utf-8')
        hoja.write(' '*30 + '{:>19}{:>13}{:>13}'.format(*tabla['head'][1]) + '\n')
        hoja.write('fila  {:^6} {:^19} {:^4}{:>12}{:>13}{:>13}'.format(*tabla['head'][0]) + '\n')
        hoja.write('='*75 + '\n')
        i = 1
        for fila in tabla['body']:
            if papel['VERB']:
                if i%2 != 0:
                    print(f'{i:<6}' + '{:<4}{:>2} {:^19} ({}) {:>12} {:>12} {:>12.2f}'.format(*fila))
                else:
                    print(f'{i:<6}' + ' '*27 + '({}) {:>12} {:>12} {:>12.2f}'.format(*fila))
                    print('-'*75)
            if i%2 != 0:
                hoja.write(f'{i:<6}' + '{:<4}{:>2} {:^19} ({}) {:>12} {:>12} {:>12.2f}'.format(*fila) + '\n')
            else:
                hoja.write(f'{i:<6}' + ' '*27 + '({}) {:>12} {:>12} {:>12.2f}'.format(*fila) + '\n')
                hoja.write('-'*75 + '\n')
            i += 1
        hoja.close()
    elif resumen == 'delta-nodos':
        if papel['VERB']:
            print(' '*22 + '{:>21}{:>12}{:>14}'.format(*tabla['head'][1]))
            print('fila  {:^19} {:4} {:>12} {:>11} {:>13}'.format(*tabla['head'][0]))
            print('='*69)
        hoja = open(project + '/tmp/' + resumen + '.txt', 'w', encoding='utf-8')
        hoja.write(' '*22 + '{:>21}{:>12}{:>14}'.format(*tabla['head'][1]) + '\n')
        hoja.write('fila  {:^19} {:4} {:>12} {:>11} {:>13}'.format(*tabla['head'][0]) + '\n')
        hoja.write('='*69 + '\n')
        i = 1
        for fila in tabla['body']:
            if papel['VERB']:
                print(f'{i:<6}' + '{:^19} ({:02}) {:>12.4f} {:>11.4f} {:>13.3e}'.format(*fila))
                print('-'*69)
            hoja.write(f'{i:<6}' + '{:^19} ({:02}) {:>12.4f} {:>11.4f} {:>13.3e}'.format(*fila) + '\n')
            hoja.write('-'*69 + '\n')
            i += 1
        hoja.close()
    elif resumen == 'drifts':
        if papel['VERB']:
            print(' '*22 + '{:>21}{:>12}{:>11}'.format(*tabla['head'][1]))
            print('fila  {:^19} {:4} {:>11} {:>11} {:>10}'.format(*tabla['head'][0]))
            print('='*66)
        hoja = open(project + '/tmp/' + resumen + '.txt', 'w', encoding='utf-8')
        hoja.write(' '*22 + '{:>21}{:>12}{:>11}'.format(*tabla['head'][1]) + '\n')
        hoja.write('fila  {:^19} {:4} {:>11} {:>11} {:>10}'.format(*tabla['head'][0]) + '\n')
        hoja.write('='*66 + '\n')
        i = 1
        for fila in tabla['body']:
            if papel['VERB']:
                print(f'{i:<6}' + '{:^19} ({:02}) {:>12.4f} {:>11.4f} {:>10.3f}'.format(*fila))
                print('-'*66)
            hoja.write(f'{i:<6}' + '{:^19} ({:02}) {:>12.4f} {:>11.4f} {:>10.3f}'.format(*fila) + '\n')
            hoja.write('-'*66 + '\n')
            i += 1
        hoja.close()
    elif resumen == 'graficos':
        if papel['config']['graf']:
            for i, m in enumerate(struct.miembro):
                if not m: continue
                for COMBO in m.resumen:
                    grafVV(m, COMBO, ij=True, full=True)
                if i > 5: break
        else:
            for i, m in enumerate(struct.miembro):
                if not m: continue
                for COMBO in m.resumen:
                    grafVV(m, COMBO, ij=True)
    return

def reporte(accion='generar'):
    """Atajo para la funcion 'informacion()'."""
    if accion == 'generar':
        informacion('generar')
    elif accion == 'moldear':
        informacion('esfuerzos')
        informacion('delta-nodos')
        informacion('drifts')
    if accion == 'graficos':
        informacion('graficos')        
    return

def ver_drift():
    ψ = 0
    d = 0
    for i in struct.drift:
        for j in struct.drift[i]:
            if abs(j) > d:
                d = abs(j)
                r = abs(j)/(2/1.2)
    ψ = r**0.25
    
    VERB = 1
    if papel['VERB'] or VERB:
        print(struct.drift)
        print(f'drift max: {d}, r: {r}, ψ: {ψ}')
        print(f'ψ: {ψ}')
        line()

    perfiles_i = [(m.b, m.h) for m in struct.miembro if m]
    perfiles_f = [(round(b*ψ), round(h*ψ)) for b, h in perfiles_i]
    if papel['VERB'] or VERB:
        if perfiles_i == perfiles_f:
            print('verifica para el drift!')
        else:
            print('se necesita corregir los perfiles...')
        line()
    
    if papel['VERB'] or VERB:
        for m in struct.miembro:
            if not m: continue
            print('b:', m.b, 'h:', m.h, ';', m.name, m.ID, 'nivel:', m.nivel)
        line()
        for b, h in perfiles_f:
            #if not m: continue
            print( (round(b), round(h)), ',', sep='', end='' )
        p()
        line()
    
    _perfiles = []
    for m in struct.miembro:
        if not m: continue
        _perfiles.append((m.b, m.h))
    dick = struct.annainfo['esfuerzos'].copy()
    dick.update({ 'perfiles': _perfiles })
    #dick.update({ 'drift': d })
    with open('_pera.py', 'w') as _pera:
        _pera.write('tabla = ' + str(dick))
    
    return d, perfiles_f

#algo_drift()

import opti

lectura()
struct = Grilla()
struct.miembro = [None] + armar_estructura()
cargar_losas() # !! no sobrescribe

drift_max = 2/1.2
drift = None
DESIGN = False

while not DESIGN:
    update()
    pp_barras('limpiar')
    pp_barras('cargar')
    load_case_gen()
    anna()
    reporte('generar')
    reporte('moldear')
    perfil_a = [(m.b, m.h) for m in struct.miembro if m]
    drift, perfil_b = ver_drift()
    if perfil_a != perfil_b:
        for i, m in enumerate(struct.miembro):
            if not m: continue
            (m.b, m.h) = perfil_b[i - 1]
    else:
        DESIGN = True
    input('FFFF')
while DESIGN:
    perfil_a = [(m.b, m.h) for m in struct.miembro if m]
    perfil_b = opti.jandro(perfil_a, struct.annainfo['esfuerzos'], papel)
    if perfil_a != perfil_b:
        for i, m in enumerate(struct.miembro):
            if not m: continue
            (m.b, m.h) = perfil_b[i - 1]
        input('actualizando en pre-design')
        update()
        pp_barras('limpiar')
        pp_barras('cargar')
        load_case_gen()
        anna()
        reporte('generar')
        reporte('moldear')
    else:
        break
    i = 0
    while True:
        i += 1
        perfil_a = [(m.b, m.h) for m in struct.miembro if m]
        drift, perfil_b = ver_drift()
        input('en drift en design')
        if round(drift, 3) <= round(drift_max, 3) and i == 1:
            DESIGN = False
            break
        elif perfil_a == perfil_b:
            break
        else:
            for i, m in enumerate(struct.miembro):
                if not m: continue
                (m.b, m.h) = perfil_b[i - 1]
            update()
            pp_barras('limpiar')
            pp_barras('cargar')
            load_case_gen()
            anna()
            reporte('generar')
            reporte('moldear')

reporte('graficos')

def pAPP(): pass
