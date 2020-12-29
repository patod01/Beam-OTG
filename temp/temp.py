def GdL(bar='', a, b, nodos, tipo=''):
    T = zeros([6, 3*nodos])
    
    # elseif x == 1 || y == 1
    #     return "reacts not considered"
    # elseif x == 19 || y == 19
    #     return "reacts not considered"
    # elif x > 20 or y > 20
    #     return "index error"
    
    # x == 20 ? x-=1 : x
    # y == 20 ? y-=1 : y
    
    if a > nodos or b > nodos: return 'index error'
    
    if a != 0:
        T[0, 3*a - 3] = 1
        T[1, 3*a - 2] = 1
        T[2, 3*a - 1] = 1
    if b != 0:
        T[3, 3*b - 3] = 1
        T[4, 3*b - 2] = 1
        T[5, 3*b - 1] = 1
    
    print(T, '\n')
    return T
