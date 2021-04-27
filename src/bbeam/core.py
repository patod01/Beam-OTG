costos = []
costo
costo_min
drift
drift_max = 2/1.2
perfiles = []
perfil_a
perfil_b
DESIGN = False
KAPUT = 0

def verDrift(): pass
def optijandro(perfil=[], esfuerzos={}, design=True): pass
def update(): pass

def optimize():
    while not DESIGN:
        drift, perfil_b = verDrift()
        if perfil_a != perfil_b:
            perfil_a = perfil_b
            update()
        else:
            DESIGN = True
    while DESIGN:
        costo, perfil_b = optijandro(perfil_a, esfuerzos)
        if perfil_a != perfil_b:
            perfil_a = perfil_b
            update()
        i = 0
        while True:
            i += 1
            drift, perfil_b = verDrift()
            if drift <= drift_max and i = 1:
                perfiles.append(perfil_a)
                costos.append(costo)
            if perfil_a != perfil_b:
                perfil_a = perfil_b
                update()
            else:
                 break
        if len(costos) == verificaciones or KAPUT == 100:
            c = costos.index(min(costos))
            perfil_a = perfiles[c]
            DESIGN = False
            detalle = optijandro(perfil_a, esfuerzos, DESIGN)
    return
