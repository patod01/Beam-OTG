def K(k, a, b, gdl):
    T = zeros([6, 3*gdl])
    
    if x < 0 or y < 0:
        return "index error"
    # elseif x == 1 || y == 1
    #     return "reacts not considered"
    # elseif x == 19 || y == 19
    #     return "reacts not considered"
    # elif x > 20 or y > 20
    #     return "index error"
    
    # x == 20 ? x-=1 : x
    # y == 20 ? y-=1 : y

        T[0, 2*a - 3] = 1
        T[1, 2*a - 2] = 1
        T[2, 2*a - 1] = 1

        T[3, 2*b - 3] = 1
        T[4, 2*b - 2] = 1
        T[5, 2*b - 1] = 1
    
    return T
