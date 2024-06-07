#!/usr/bin/env python3

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
    default="tp1_ej2_training.csv",
    help="los datos son procesados considerando un .csv con los \
                    outputs en las últimas dos columnas",
)
parser.add_argument(
    "--filename_modelo",
    default="weights_ej_2",
    help="nombre \
                    del archivo exportado que contiene el modelo entrenado",
)
parser.add_argument(
    "--S",
    help="Nodos por capa sin contar entrada ni salida,\
                    separados por coma, sin espacios ni []",
    default="2",
)
parser.add_argument("--lr", help="learning rate", type=float, default=0.01)
parser.add_argument("--activation", help="tanh o sigmoid", default="sigmoid")
parser.add_argument("--alfa_momento", help="entre 0 y 1", default=0, type=float)
parser.add_argument("--epocas", default=4000, type=int)
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

data = np.random.permutation(np.array(data))


# {{{ separar en train y valid
def train_valid_split(datos):  # {{{
    datos_train = datos[: int(3 * len(datos) / 4), :]
    datos_valid = datos[int(3 * len(datos) / 4) :, :]
    return datos_train, datos_valid  # }}}


datos_train, datos_valid = train_valid_split(data)
# }}}

# separar en x y z {{{
x_train, z_train = datos_train[:, :-2], datos_train[:, -2:]
x_v, z_v = datos_valid[:, :-2], datos_valid[:, -2:]
# }}}

# normalizaciones {{{
# vectores utiles {{{
minimos_x_train = x_train.min(axis=0)
maximos_x_train = x_train.max(axis=0)
minimos_z_train = z_train.min(axis=0)
maximos_z_train = z_train.max(axis=0)

minimos_x_v = x_v.min(axis=0)
maximos_x_v = x_v.max(axis=0)
minimos_z_v = z_v.min(axis=0)
maximos_z_v = z_v.max(axis=0)
# }}}


def min_max_norm(datos, minimo, maximo):  # {{{
    data_normalizada = (datos - minimo) / (maximo - minimo)
    return data_normalizada  # }}}


# subsets normalizados {{{
x_train_norm = min_max_norm(x_train, minimos_x_train, maximos_x_train)
z_train_norm = min_max_norm(z_train, minimos_z_train, maximos_z_train)

x_v_norm = min_max_norm(x_v, minimos_x_v, maximos_x_v)
z_v_norm = min_max_norm(z_v, minimos_z_v, maximos_z_v)
# }}}


def desnormalizar(datos, minimos, maximos):  # {{{
    return datos * (maximos - minimos) + minimos  # }}}


# }}}


# }}}

# arq {{{
P = x_train.shape[0]
S = [int(i) for i in args.S.split(",")]
S.insert(0, x_train.shape[1])
S.append(2)
# S = [x.shape[1],20,10,5,15,1] #89%
# S = [x.shape[1],20,1]
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


# funcs {{{
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

    for k in range(1, L - 1):  # en vez de L, asi hago la ultima unidad lineal
        Y[k - 1][:] = bias_add(Y_temp)
        Y_temp = activation(Y[k - 1] @ W[k])
    Y[L - 2][:] = bias_add(Y_temp)
    Y_temp = Y[L - 2] @ W[L - 1]  # unidad lineal
    # Y[L-1] = (Y_temp - minimos_objetivo)/(maximos_objetivo-minimos_objetivo)
    Y[L - 1] = min_max_norm(Y_temp, minimos_z_train, maximos_z_train)
    # output normalizado
    if predict:
        return Y_temp  # porque Y_temp no está normalizado.

    return Y  # }}}


# }}}


# backprop {{{
# funcs {{{
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
    # dY_output = d_activation(Y[L-1])
    dY_output = 1  # unidad lineal
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

# init training {{{
B = P
lr = args.lr
costo_epoca = []
error_val = []
t = 0
cost_batch = []
cost_epoca = []
# }}}

# batch train{{{

"""
ambos objetivos tienen un desvío estándar y rangos de valores muy similares,
calculados con:
z_train.std(0)
z_train.min(0)
z_train.max(0)
Esto puede ser utilizado para determinar el corte del entrenamiento: cuando el
error medio sea al menos menor que el desvío estándar.
"""

while t < args.epocas:
    H = np.random.permutation(P)
    for batch in range(0, P, B):
        x_batch_norm = x_train_norm[H[batch : batch + B]]
        z_batch_norm = z_train_norm[H[batch : batch + B]]
        z_batch = z_train[H[batch : batch + B]]

        Y = forward(x_batch_norm, W)

        output_train = forward(x_batch_norm, W, True)  # no normalizado

        # guardo el error cuadrado medio no normalizado del batch
        cost_batch.append(estimation(z_batch, output_train))

        dw = backprop_momento(Y, z_batch_norm, W, dw)
        W = adaptation(W, dw)

    cost_epoca.append(np.mean(cost_batch))  # media de los errores cuadrados
    # medios de cada batch

    cost_batch = []

    # a continuación un paso forward obteniendo outputs sin normalizar
    output_valid = forward(x_v_norm, W, True)

    error_val.append(estimation(z_v, output_valid))

    t += 1
# }}}

# }}}

# evaluacion {{{
"""
el uso de estas funciones con datos de testeo y las matrices exportadas del
entrenamiento se ejemplifica al final del script.
"""


"""
La siguiente función puede ser utilizada para evaluar el modelo entregado con
los datos de testeo (separando objetivos de variables independientes en "x_eval"
y "objetivo_eval") con inputs normalizados y objetivos no normalizados, dado que
el output de la función es el error cuadrado promedio medido con outputs de la
red (y objetivos) no normalizados.
"""


def evaluacion(x_eval, pesos, objetivo_eval):  # {{{
    output_modelo_entrenado = forward(x_eval, pesos, predict=True)
    error_cuadrado_promedio = estimation(objetivo_eval, output_modelo_entrenado)
    return error_cuadrado_promedio  # }}}


error_cuadrado_no_normalizado_validacion = evaluacion(x_v_norm, W, z_v)

print(
    f"error cuadrado con outputs no normalizados: \
{error_cuadrado_no_normalizado_validacion}"
)


"""
esta función es análoga a evaluacion, pero utilizando outputs de la red (y por
ende objetivos) normalizados. Al igual que en el caso anterior, inputs
normalizados.
"""


def evaluacion_normalizada(x_eval, pesos, objetivo_eval):  # {{{
    output_modelo_entrenado_norm = forward(x_eval, pesos)[L - 1]
    error_cuadrado_promedio_norm = estimation(
        objetivo_eval, output_modelo_entrenado_norm
    )
    return error_cuadrado_promedio_norm  # }}}


validacion_normalizada = evaluacion_normalizada(x_v_norm, W, z_v_norm)

print(f"error cuadrado con outputs normalizados: {validacion_normalizada}")


# }}}

# plot {{{
plt.figure()
# plt.plot(costo_epoca)
plt.plot(cost_epoca, linewidth=4)
plt.plot(error_val, label="valid", linewidth=2.5)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("épocas")
plt.ylabel("costo")
plt.legend(prop={"size": 30})
plt.show()
# }}}

# {{{ exportar modelo (lista de pesos W) a archivo filename_modelo \
# y utilizar las funciones de evaluación del modelo
if args.exportar == True:
    np.savez(f"{args.filename_modelo}.npz", W=np.array(W, dtype=object))
else:
    pass


"""

para cargar las matrices que queden almacenadas en filename_modelo.npz luego
de entrenar un modelo nuevo:

np.load('filename_modelo.npz',allow_pickle=True)['W']

Si, entonces, se desea evaluar W con datos de testeo, se puede hacer:
evaluacion_normalizada(x_testeo_normalizado,W,z_testeo_normalizado)

donde W puede ser ya sea una lista de matrices producto del entrenamiento
llevado a cabo en este script, como las matrices entrenadas que fueron
entregadas aparte.

Si, en vez de evaluar el rendimiento, se desea tener el output del modelo con
ciertos datos nuevos:
forward(x_datos_normalizados,W,True)

"""
# }}}
