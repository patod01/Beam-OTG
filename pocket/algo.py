class mx():
    mnError = 'merengue'
    rangeError = 'Fuera del rango'
    t = 'for transpose some day'
    m = 'some day with m'
    n = 'some day with n'
    def __init__(matrix, *mn):
        # agregar verificacion de largos de filas
        if type(mn) == tuple:
            print(len(mn))
            matrix.body = mn
        else:
            raise Exception(matrix.mnError)
    def __lt__(matrix, mn):
        # agregar verificacion de listas (aca y atras)
        if type(mn) == tuple:
            #print(len(mn))
            matrix.body = mn
        else:
            raise Exception(matrix.mnError)
    def __repr__(matrix):
        return str(matrix.body)
    def __str__(matrix):
        for i in matrix.body: print(i)
        # update to octave like print...
        return ''
    def __getitem__(matrix, mn):
        #print(mn, type(mn))
        if type(mn) == int and mn >= 1:
            return matrix.body[mn - 1]
        if type(mn) == tuple and len(mn) == 2 and mn[0] >= 1 and mn[1] >= 1:
            return matrix.body[mn[0] - 1][mn[1] - 1]
        else:
            raise Exception(matrix.rangeError)
    def __setitem__(matrix, mn, data):
        if type(mn) == tuple and len(mn) == 2:
            matrix.body[mn[0] - 1][mn[1] - 1] = data
        else:
            raise Exception(matrix.rangeError)
    def __iter__(matrix):
        return iter(matrix.body)
    def __len__(matrix): pass
    def __add__(matrix): pass
    def __radd__(matrix): pass
    def __sub__(matrix): pass
    def __rsub__(matrix): pass
    def __mul__(matrix): pass
    def __rmul__(matrix): pass
    def __pow__(matrix): pass
    def __matmul__(matrix): pass

# rip
