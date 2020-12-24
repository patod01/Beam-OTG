class mx():
    def __init__(matrix, mn=0):
        matrix.body = mn
    def __repr__(matrix):
        return matrix.body
    def __matmul__(matrix): pass
    def __add__(matrix): pass
    def __radd__(matrix): pass
    def __sub__(matrix): pass
    def __rsub__(matrix): pass
    def __mul__(matrix): pass
    def __rmul__(matrix): pass
    def __lt__(matrix): pass
    #def __getitem__(matrix): pass


def t():
    __call__(self):
        print('\n---- - ----\n')


b = array([
    [0.8746284,   0.8807474,   0.9054132,   0.0135037,   0.0342610,   0.8487201],
    [0.5771128,   0.1537398,   0.2624749,   0.1739316,   0.0048405,   0.3780156],
    [0.6321945,   0.3685543,   0.2962360,   0.7676852,   0.4170991,   0.3705051],
    [0.2406543,   0.2694103,   0.7098604,   0.7463751,   0.5079600,   0.0588400],
    [0.9199274,   0.6061314,   0.6577218,   0.4736748,   0.1278603,   0.5730456],
])

r1 = array([0, 0, 0, r[0:3]]).T
r2 = array([0, 0, 0, r[3:6]]).T
r3 = r

ϵ1 = a1*r1
ϵ2 = a1*r2
ϵ3 = a3*r3

σ1 = k1*ϵ1
σ2 = k1*ϵ2
σ3 = k3*ϵ3

σ = array([
    [round(σ1[0]/100), round(σ2[0]/100), round(σ3[0]/100)],
    [round(σ1[1]/100), round(σ2[1]/100), round(σ3[1]/100)],
    [round(σ1[2]), round(σ2[2]), round(σ3[2])]
])

V = [(σ[0,0] + σ[1,0])/3 (σ[0,1] + σ[1,1])/3 (σ[0,2] + σ[1,2])/2]

println("Ma: ", σ[0,:])
println("Mb: ", σ[1,:])
println("F:  ", σ[2,:])
println("V:  ", V)
