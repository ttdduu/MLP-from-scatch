# imports {{{
import argparse, sys
import numpy as np
import csv
import itertools as it
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# }}}

# argumentos {{{

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--filename_datos",
    default="tp1_ej1_training.csv",
    help="el\
                    procesamiento de los datos en el script fue realizado \
                    utilizando un .csv con los targets en la primera columna",
)
parser.add_argument(
    "--filename_modelo",
    default="weights_ej_1",
    help="nombre \
                    de archivo que es exportado y contiene el modelo entrenado",
)
parser.add_argument(
    "--S",
    help="Nodos por capa sin contar entrada ni salida,\
                    separados por coma, sin espacios ni []",
    default="5",
)
parser.add_argument("--lr", help="learning rate", type=float, default=0.01)
parser.add_argument("--activation", help="tanh o sigmoid", default="sigmoid")
parser.add_argument("--alfa_momento", help="entre 0 y 1", default=0.9, type=float)
parser.add_argument("--epocas", default=8000, type=int)
parser.add_argument(
    "--exportar",
    default=True,
    help="si el usuario desea \
                    exportar el modelo entrenado al archivo filename_modelo.npz",
)
# parser.add_argument('--B',help='batch size',default='P')
args = parser.parse_args()
# }}}

# datos {{{
data = pd.read_csv(args.filename_datos, header=None)


def train_valid_split(datos):  # {{{
    datos_train = datos[: int(3 * len(datos) / 4), :]
    datos_valid = datos[int(3 * len(datos) / 4) :, :]
    return datos_train, datos_valid  # }}}


def min_max_norm(datos, minimo, maximo):  # {{{
    data_normalizada = (datos - minimo) / (maximo - minimo)
    return data_normalizada  # }}}


"para obtener datos aleatorizados llamando a la funcion"


def datos(data):  # {{{
    data = np.random.permutation(np.array(data))
    x = (data)[:, 1:]

    x = x.astype(float)
    estandarizacion = lambda x: (x - x.mean(0)) / np.square(x.std(0))
    # x = min_max_norm(x,x.min(axis=0),x.max(axis=0))

    if args.activation == "sigmoid":
        z = np.array([1 if dato == "M" else 0 for dato in data[:, 0:1]])

    elif args.activation == "tanh":
        z = np.array([1 if dato == "M" else -1 for dato in data[:, 0:1]])

    z = z.reshape(410, 1)

    x_v = estandarizacion(x[int(3 * len(x) / 4) :])
    x_train = estandarizacion(x[: int(3 * len(x) / 4)])

    z_v = z[int(3 * len(z) / 4) :]
    z_train = z[: int(3 * len(z) / 4)]
    return x, x_train, x_v, z_train, z_v


x, x_train, x_v, z_train, z_v = datos(data)
# }}}

# }}}

# arq {{{
P = x_train.shape[0]
S = [int(i) for i in args.S.split(",")]
S.insert(0, x.shape[1])
S.append(1)
L = len(S)

# }}}

# forward {{{

# init forward {{{

W = [
    np.random.normal(0, 0.1, (s + 1, S[index + 1]))
    for index, s in enumerate(S)
    if index < len(S) - 1
]
W.insert(0, 0)

# }}}


# bias and activation{{{
def bias_add(V):  # {{{
    bias = -np.ones((len(V), 1))
    return np.concatenate((V, bias), axis=1)  # }}}


def activation(x):  # {{{
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    if args.activation == "sigmoid":
        return sigmoid(x)
    elif args.activation == "tanh":
        return np.tanh(x)


# }}}
# }}}


def forward(Xh, W, predict=False):  # {{{
    Y = [
        np.zeros((Xh.shape[0], value + 1))
        if index != len(S) - 1
        else np.zeros((Xh.shape[0], value))
        for index, value in enumerate(S)
    ]
    Y_temp = Xh

    for k in range(1, L):
        Y[k - 1][:] = bias_add(Y_temp)
        Y_temp = activation(Y[k - 1] @ W[k])

    Y[L - 1] = Y_temp
    if predict:
        return Y[L - 1]

    return Y  # }}}


# }}}


# backpropagation {{{
# bias and activation {{{
def bias_sub(V):  # {{{
    return V[:, :-1]  # }}}


def d_activation(x):  # {{{
    d_sigmoid = lambda x: x * (1 - x)
    d_tanh = lambda x: 1 - x**2

    if args.activation == "sigmoid":
        return d_sigmoid(x)
    elif args.activation == "tanh":
        return d_tanh(x)


# }}}
# }}}

# init {{{

dw = [0 * w for w in W]
# dw_previo = dw
alfa = args.alfa_momento
# }}}


def backprop_momento(Y, z, W, dw_previo):  # {{{
    E_output = z - Y[L - 1]
    dY_output = d_activation(Y[L - 1])
    D_output = E_output * dY_output
    D = D_output

    for k in range(L - 1, 0, -1):
        dw[k][:] = lr * (Y[k - 1].T @ D) + alfa * dw_previo[k]
        E = D @ W[k].T
        dY = d_activation(Y[k - 1])
        D = bias_sub(E * dY)

    return dw


# }}}
# }}}


# training {{{
# funcs # {{{
def adaptation(W, dw):  # {{{
    for k in range(1, L):
        W[k][:] += dw[k]
    return W  # }}}


def estimation(z, outputs):  # {{{
    return np.mean(np.square(z - outputs))  # }}}


# }}}

# init training# {{{
B = P
lr = args.lr
costo_epoca = []
error_val = []
t = 0
cost_batch = []
cost_epoca = []
# }}}

# batch train{{{

while t < args.epocas:
    c = 0
    H = np.random.permutation(P)
    for batch in range(0, P, B):
        x_batch = x_train[H[batch : batch + B]]
        z_batch = z_train[H[batch : batch + B]]

        Y = forward(x_batch, W)

        cost_batch.append(estimation(z_batch, Y[L - 1]))
        # se appendea el cost del batch: la media de los errores cuadrados
        dw = backprop_momento(Y, z_batch, W, dw)
        W = adaptation(W, dw)

    # costo_epoca.append((c)/(P/B))
    cost_epoca.append(np.mean(cost_batch))

    cost_batch = []
    error_val.append(estimation(z_v, forward(x_v, W, True)))
    # hacer un paso forward y appendear la media de los errores cuadrados
    t += 1
# }}}

# }}}

# evaluacion {{{
"""
esta función puede ser utilizada para evaluar el modelo entregado con los datos
de testeo (separando objetivos de variables independientes en "datos_eval" y
"objetivo")

"""


def evaluacion(x_eval, pesos, objetivo_eval):
    output_modelo_entrenado = forward(x_eval, pesos, predict=True)
    proporcion = (objetivo_eval == np.round(output_modelo_entrenado)).sum() / len(
        output_modelo_entrenado
    )
    print(f"proporción de aciertos: {proporcion}")


validacion = evaluacion(x_v, W, z_v)
# }}}

# plot {{{
plt.figure()
# plt.plot(costo_epoca)
plt.plot(cost_epoca, linewidth=4)
plt.plot(error_val, label="valid", linewidth=4)
plt.xlabel("épocas")
plt.xticks(fontsize=20)
plt.ylabel("costo")
plt.yticks(fontsize=20)
plt.legend(prop={"size": 30})
plt.show()
# }}}

# {{{ exportar modelo (lista de pesos W) a archivo filename_modelo
if args.exportar == True:
    np.savez(f"{args.filename_modelo}.npz", W=np.array(W, dtype=object))
else:
    pass


"""

para cargar las matrices que queden almacenadas en filename_modelo.npz luego
de entrenar un modelo nuevo:

np.load('filename_modelo.npz',allow_pickle=True)['W']


"""
# }}}
