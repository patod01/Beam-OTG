# recuperacion de datos

with open('../IO/input.txt', encoding='utf-8') as i_text:
    i_temp = i_text.readlines()

dflt_op = [
    'numero de pisos',
    'numero habitaciones',
    'distancia entre columnas',
    'altura de piso',
    'modulo de poisson 𝜈',
    'resistencia del hormigon',
    'carga viva',
    'zona sismica',
    'tipo de suelo',
    'categoria de edificacion',
    'resistencia del acero',
    'tama;o max de columnas',
    'altura max de viga'
]

dflt_in = [
    '1', '1', '200', '300', '0.25', '19.5',
    '0', 'def', 'def', 'def', 'def', 'def', 'def'
]

user_op = []
user_in = []

for i in i_temp:
    if not i.startswith('\n'):
        user_op.append(i.partition(':')[0].strip())
        user_in.append(i.partition(':')[2].strip())
print(user_in)
if user_op != dflt_op: raise Exception('wena')

for i in range(len(user_in)):
    if user_in[i] == '':
        user_in[i] = dflt_in[i]
    elif user_in[i].isalpha() == False:
        user_in[i] = float(user_in[i])
print(user_in)
