# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
=============================================================================
 NEWTON-RAPHSON APLICADO AL MODELO ZESTIMATE -- KC HOUSE DATA
 Metodos Numericos 2026 . Camelino, Funes, Ziletti -- Grupo 5
 Universidad Catolica de Cordoba, Facultad de Ingenieria

 CONTEXTO (del informe):
   Zillow Group Inc. colapso en Q4-2021 por errores sistematicos en su
   modelo Zestimate. Este script aplica Newton-Raphson para optimizar la
   funcion de perdida cuadratica sobre propiedades reales del KC House Data
   (King County, WA -- 21.613 propiedades).

 DIFERENCIA CLAVE CON EL CASO DEL INFORME:
   Informe  : 1 propiedad piloto, L(w) cuadratica exacta -> 1 iteracion
              (Hessiana constante, ec. 5 del informe)
   Este codigo:
     Seccion A -> 5 propiedades reales del CSV.
                 Se optimiza UN UNICO factor de correccion global w* para
                 el grupo, minimizando el MSE sobre las 5 casas juntas.
                 Se introduce alpha = 0.25 (amortiguamiento) en la
                 actualizacion N-R para forzar convergencia iterativa visible:
                   w_{i+1} = w_i - alpha * L'(w_i) / L''(w_i)
                 Cada entrada de `resultados` guarda el historial del w global
                 evaluado individualmente sobre cada casa, para mantener
                 compatibilidad con Figuras 1 y 2 del PASO 7.
     Seccion B -> Modelo multivariado real con 5 features del dataset
                 N-R coordinado -> varias iteraciones necesarias
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

# -- Configuracion visual -----------------------------------------------------
plt.rcParams.update({
    'font.family'     : 'DejaVu Sans',
    'axes.spines.top' : False,
    'axes.spines.right': False,
    'axes.grid'       : True,
    'grid.alpha'      : 0.3,
    'figure.dpi'      : 110,
})
COLORES = ['#2E86AB', '#E84855', '#F4A261', '#2A9D8F', '#9B5DE5']

def fmt_usd(n):
    """Formatea numero como $XXX.XXX USD"""
    return f"${n:,.0f}".replace(",", ".")

# =============================================================================
#  PASO 1 -- CARGAR KC HOUSE DATA
# =============================================================================

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kc_house_data.csv")

if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(
        f"\n  Archivo no encontrado: {CSV_PATH}\n"
        "  -> Coloca kc_house_data.csv en la misma carpeta que este .py\n"
        "  -> O edita la variable CSV_PATH con la ruta completa."
    )

df = pd.read_csv(CSV_PATH)

print("=" * 65)
print("  NEWTON-RAPHSON -- MODELO ZESTIMATE -- KC HOUSE DATA")
print("  Metodos Numericos 2026 . UCC . Grupo 5")
print("=" * 65)
print(f"\n[1] Dataset cargado: {len(df):,} propiedades, {len(df.columns)} variables\n")

# =============================================================================
#  PASO 2 -- SELECCION DE 5 PROPIEDADES PILOTO DEL CSV
# =============================================================================

PROP_IDX   = [20,   13,    80,    3,     5   ]
Y_REAL_USD = np.array([385000, 400000, 390000, 604000, 1225000], dtype=float)
ERROR_PCT  = np.array([0.18,   0.12,   0.22,   0.15,   0.20  ])

Y_ZEST_USD = np.round(Y_REAL_USD * (1 - ERROR_PCT))

# w0 inicial fijo para el optimizador global
W0_GLOBAL  = 0.70   # punto de inicio unico para la busqueda del w global

print("-" * 65)
print("  PROPIEDADES PILOTO -- KC House Data")
print("-" * 65)
header = f"  {'Prop':<6} {'Y_real':>12} {'Zestimate':>12} {'Error $':>12} {'Error %':>8}"
print(header)
print("  " + "-" * 58)
for k in range(5):
    err_usd = Y_REAL_USD[k] - Y_ZEST_USD[k]
    print(f"  P{k+1:<5} {fmt_usd(Y_REAL_USD[k]):>12} {fmt_usd(Y_ZEST_USD[k]):>12} "
          f"{fmt_usd(err_usd):>12} {ERROR_PCT[k]*100:>7.1f}%")
print(f"\n  w0 global (punto de inicio N-R): {W0_GLOBAL}\n")

# =============================================================================
#  PASO 3 -- FUNCIONES DE PERDIDA GLOBAL (MSE sobre M=5 propiedades)
#
#  Modelo escalar: y_k(w) = w * Y_zest,k
#
#  L(w)   = (1/M) * sum_k (Y_real,k - w * Y_zest,k)^2          [MSE global]
#  L'(w)  = -(2/M) * sum_k Y_zest,k * (Y_real,k - w * Y_zest,k)
#  L''(w) = (2/M) * sum_k Y_zest,k^2   (Hessiana -- constante)
#
#  Actualizacion N-R amortiguada (alpha = 0.25):
#    w_{i+1} = w_i - alpha * L'(w_i) / L''(w_i)
#
#  El amortiguamiento impide el salto directo al minimo en 1 sola iteracion,
#  generando una trayectoria iterativa numericamente visible y pedagogicamente
#  significativa. alpha=1 recuperaria Newton-Raphson puro (1 iter exacta).
# =============================================================================

M = len(Y_REAL_USD)   # 5 propiedades

def L_global(w, Yr_vec, Yz_vec):
    """MSE global sobre el grupo de M propiedades."""
    return np.mean((Yr_vec - w * Yz_vec) ** 2)

def dL_global(w, Yr_vec, Yz_vec):
    """Primera derivada del MSE global respecto a w."""
    return -(2 / M) * np.sum(Yz_vec * (Yr_vec - w * Yz_vec))

def d2L_global(Yz_vec):
    """Segunda derivada (Hessiana) del MSE global -- constante en w."""
    return (2 / M) * np.sum(Yz_vec ** 2)

# Funciones individuales (para historial por casa, compatibilidad con Figuras 1 y 2)
def L_ind(w, Yr, Yz):
    return (Yr - w * Yz) ** 2

def err_rel_ind(w, Yr, Yz):
    return abs(Yr - w * Yz) / Yr * 100

# =============================================================================
#  PASO 4 -- NEWTON-RAPHSON GLOBAL CON AMORTIGUAMIENTO
#
#  Se optimiza un UNICO w* para las 5 casas juntas minimizando el MSE global.
#  alpha = 0.25 amortigua el paso de N-R: fuerza varias iteraciones visibles.
#  Al final se reconstruye la lista `resultados` evaluando el w global en
#  cada propiedad individualmente, para compatibilidad con los graficos.
# =============================================================================

print("=" * 65)
print("  SECCION A: Newton-Raphson Global con amortiguamiento (alpha=0.25)")
print("  Funcion de perdida: MSE sobre 5 propiedades piloto")
print("=" * 65)

ALPHA    = 0.25    # factor de amortiguamiento (0 < alpha <= 1)
TOL      = 1e-7    # tolerancia sobre |L'(w)|
MAX_ITER = 200

Hess = d2L_global(Y_ZEST_USD)   # constante para toda la iteracion

w        = W0_GLOBAL
w_hist   = [w]
Lg_hist  = [L_global(w, Y_REAL_USD, Y_ZEST_USD)]
dLg_hist = [dL_global(w, Y_REAL_USD, Y_ZEST_USD)]

print(f"\n  alpha (amortiguamiento) = {ALPHA}")
print(f"  TOL (|L'(w)|)           = {TOL}")
print(f"  Hessiana global (cte.)  = {Hess:.4e}\n")
print(f"  {'Iter':<6} {'w_i':>12} {'MSE_global':>16} {'L\'(w)':>16} {'|delta_w|':>12}")
print("  " + "-" * 66)
print(f"  {'0':<6} {w:>12.6f} {Lg_hist[0]:>16.6e} {dLg_hist[0]:>16.6e} {'---':>12}")

n_iter = 0
for i in range(1, MAX_ITER + 1):
    grad  = dL_global(w, Y_REAL_USD, Y_ZEST_USD)
    delta = ALPHA * grad / Hess          # paso amortiguado
    w_new = w - delta

    Lg_new  = L_global(w_new, Y_REAL_USD, Y_ZEST_USD)
    dLg_new = dL_global(w_new, Y_REAL_USD, Y_ZEST_USD)

    w_hist.append(w_new)
    Lg_hist.append(Lg_new)
    dLg_hist.append(dLg_new)
    n_iter = i

    tag = "  <- w* CONVERGENCIA" if abs(dLg_new) < TOL else ""
    print(f"  {i:<6} {w_new:>12.6f} {Lg_new:>16.6e} {dLg_new:>16.6e} "
          f"{abs(delta):>12.6e}{tag}")

    w = w_new
    if abs(dLg_new) < TOL:
        break

w_star = w_hist[-1]
print(f"\n  -> Convergencia en {n_iter} iteracion(es)")
print(f"     w0 = {W0_GLOBAL:.4f}  ->  w* = {w_star:.8f}")
print(f"\n  Predicciones finales con w* = {w_star:.6f}:")
print(f"  {'Prop':<6} {'Y_real':>12} {'Zestimate':>12} {'Pred(w*·Yz)':>14} {'Error final %':>14}")
print("  " + "-" * 64)
for k in range(M):
    pred_f = w_star * Y_ZEST_USD[k]
    err_f  = abs(Y_REAL_USD[k] - pred_f) / Y_REAL_USD[k] * 100
    print(f"  P{k+1:<5} {fmt_usd(Y_REAL_USD[k]):>12} {fmt_usd(Y_ZEST_USD[k]):>12} "
          f"{fmt_usd(pred_f):>14} {err_f:>13.4f}%")

# -- Reconstruir lista `resultados` evaluando w global en cada casa -----------
#    Cada elemento guarda el historial INDIVIDUAL de la casa k bajo el w
#    global de cada iteracion. Esto permite que Figura 1 muestre L(w) por
#    casa y Figura 2 muestre como converge el error relativo por casa.
#    w_hist, L_hist y err_hist tienen la misma longitud (n_iter+1) para todos.

w_hist_arr = np.array(w_hist)

resultados = []
for k in range(M):
    Yr = Y_REAL_USD[k]
    Yz = Y_ZEST_USD[k]
    L_hist_k   = np.array([L_ind(w, Yr, Yz)      for w in w_hist_arr])
    err_hist_k = np.array([err_rel_ind(w, Yr, Yz) for w in w_hist_arr])

    resultados.append({
        "Yr"      : Yr,
        "Yz"      : Yz,
        "w0"      : W0_GLOBAL,
        "w0_fijo" : W0_GLOBAL,
        "wstar"   : w_star,
        "n_iter"  : n_iter,
        "w_hist"  : w_hist_arr,         # w global en cada iteracion
        "L_hist"  : L_hist_k,           # L individual de la casa k
        "err_hist": err_hist_k,         # error relativo individual
        "err0"    : ERROR_PCT[k] * 100,
    })

# =============================================================================
#  PASO 5 -- MODELO MULTIVARIADO CON NEWTON-RAPHSON COORDINADO
# =============================================================================
# =============================================================================
#  PASO 5 -- MODELO MULTIVARIADO MEJORADO (Feature Engineering y Log-Transform)
# =============================================================================

print("\n" + "=" * 65)
print("  SECCION B: N-R multivariado coordinado -- Datos Optimizados")
print("=" * 65)

# --- 1. Filtro de Outliers Básicos ---
# Copiamos el dataframe para no afectar otras partes del código
df_clean = df[(df['bedrooms'] > 0) & (df['bedrooms'] < 10)].copy()
limite_precio = df_clean['price'].quantile(0.99) # Quitamos el 1% de propiedades extremas
df_clean = df_clean[df_clean['price'] < limite_precio].copy()

# --- 2. Selección de features ampliadas (Ubicación y condición) ---
FEAT_NAMES = ['sqft_living', 'grade', 'bathrooms', 'bedrooms', 'yr_built', 
              'lat', 'long', 'waterfront', 'view', 'condition']
X_raw = df_clean[FEAT_NAMES].values.astype(float)

# --- 3. Transformación logarítmica del Target ---
# Convierte la distribución asimétrica de precios en una distribución normal (clave para ML)
Y_raw = np.log1p(df_clean['price'].values.astype(float))
N = len(Y_raw)

# --- 4. Normalización mapminmax [-1, 1] ---
X_min, X_max = X_raw.min(axis=0), X_raw.max(axis=0)
Y_min_v, Y_max_v = Y_raw.min(), Y_raw.max()

rango_X = X_max - X_min
rango_X[rango_X == 0] = 1.0 # Evitar división por cero

X_n = 2 * (X_raw - X_min) / rango_X - 1
Y_n = 2 * (Y_raw - Y_min_v) / (Y_max_v - Y_min_v) - 1

# Anadir columna de bias
X_b = np.column_stack([np.ones(N), X_n])
p = X_b.shape[1]   # Ahora son 11 pesos (bias + 10 features)

# --- 5. Particion 70 / 15 / 15 ---
rng = np.random.default_rng(42)
idx = rng.permutation(N)
n_tr = int(0.70 * N)
n_vl = int(0.15 * N)

idx_tr = idx[:n_tr]
idx_te = idx[n_tr + n_vl:]

Xtr, Ytr = X_b[idx_tr], Y_n[idx_tr]
Xte, Yte = X_b[idx_te], Y_n[idx_te]

print(f"\n  Features ({len(FEAT_NAMES)}): {', '.join(FEAT_NAMES)}")
print(f"  Target:       log1p(price) [Corrige asimetría]")
print(f"  Normalizacion: mapminmax [-1, 1]")
print(f"  Train: {len(idx_tr):,}  |  Test: {len(idx_te):,}\n")

# --- 6. Newton-Raphson coordinado ---
W = np.zeros(p)
TOL_W   = 1e-8
MAX_W   = 300
loss_hist_mv  = []
grad_hist_mv  = []
n_conv_mv     = 0

print(f"  {'Iter':<6} {'MSE train (norm.)':>20} {'||grad||':>16}  Estado")
print("  " + "-" * 58)

for it in range(1, MAX_W + 1):
    resid = Ytr - Xtr @ W
    mse   = np.mean(resid ** 2)
    g     = -(2 / len(Ytr)) * (Xtr.T @ resid)

    loss_hist_mv.append(mse)
    grad_hist_mv.append(np.linalg.norm(g))

    # Actualizacion coordinada
    for j in range(p):
        xj      = Xtr[:, j]
        resid_j = Ytr - Xtr @ W
        dL_j    = -(2 / len(Ytr)) * np.dot(xj, resid_j)
        d2L_j   =  (2 / len(Ytr)) * np.dot(xj, xj)
        W[j]   -= dL_j / d2L_j

    n_conv_mv = it
    estado = ""
    if grad_hist_mv[-1] < TOL_W:
        estado = "<-- CONVERGENCIA"
        if it == 1 or it % 20 == 0 or estado:
            print(f"  {it:<6} {mse:>20.6e} {grad_hist_mv[-1]:>16.6e}  {estado}")
        break
    if it == 1 or it % 20 == 0:
        print(f"  {it:<6} {mse:>20.6e} {grad_hist_mv[-1]:>16.6e}  {estado}")

# -- 7. Metricas en USD --------------------------------------------------------
def desnorm(y_norm):
    """ Revierte mapminmax [-1, 1] y luego revierte log1p """
    y_log = (y_norm + 1) / 2 * (Y_max_v - Y_min_v) + Y_min_v
    return np.expm1(y_log) # Convierte el logaritmo de vuelta a dólares reales

Ypred_te_usd = desnorm(Xte @ W)
Yreal_te_usd = desnorm(Yte)

rmse = np.sqrt(np.mean((Yreal_te_usd - Ypred_te_usd) ** 2))
mae  = np.mean(np.abs(Yreal_te_usd - Ypred_te_usd))
r2   = 1 - np.sum((Yreal_te_usd - Ypred_te_usd)**2) / \
           np.sum((Yreal_te_usd - Yreal_te_usd.mean())**2)

print(f"\n  Iteraciones totales: {n_conv_mv}")
print(f"  RMSE Test: {fmt_usd(rmse)}")
print(f"  MAE  Test: {fmt_usd(mae)}")
print(f"  R2   Test: {r2:.4f}\n")

# =============================================================================
#  PASO 6 -- COMPARATIVA N-R vs SGD vs L-BFGS
# =============================================================================

print("=" * 65)
print("  TABLA COMPARATIVA -- Optimizadores (Tabla 2 del informe)")
print("=" * 65)

W_sgd    = np.zeros(p)
LR_SGD   = 0.05
TOL_SGD  = 1e-8
sgd_conv = 0
for it in range(5000):
    g_sgd = -(2 / len(Ytr)) * (Xtr.T @ (Ytr - Xtr @ W_sgd))
    W_sgd -= LR_SGD * g_sgd
    sgd_conv = it + 1
    if np.linalg.norm(g_sgd) < TOL_SGD:
        break

rmse_sgd = np.sqrt(np.mean((desnorm(Xte @ W_sgd) - Yreal_te_usd) ** 2))

print(f"\n  {'Criterio':<30} {'Newton-Raphson':<18} {'SGD':<18} {'L-BFGS*'}")
print("  " + "-" * 72)
rows = [
    ("Orden convergencia",    "Cuadratico (2deg)",  "Lineal (1deg)",    "Cuasi-2deg"),
    ("Iteraciones (KC mv.)",  str(n_conv_mv),      str(sgd_conv),    "3-8"),
    ("Tasa de aprendizaje",   "No requiere",       f"alpha = {LR_SGD}",  "Busq. linea"),
    ("RMSE Test (USD)",       fmt_usd(rmse),        fmt_usd(rmse_sgd), "~NR"),
    ("Hiperparametros",       "Ninguno",           "alpha (critico)",    "m (memoria)"),
]
for r in rows:
    print(f"  {r[0]:<30} {r[1]:<18} {r[2]:<18} {r[3]}")
print("\n  * L-BFGS estimado teorico (Nocedal & Wright, 2006)\n")

# =============================================================================
#  SECCION C -- RED NEURONAL fitnet(10) -- PyTorch
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# -- Importacion de PyTorch ---------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except ImportError:
    raise ImportError(
        "\n  PyTorch no esta instalado.\n"
        "  Ejecuta: pip install torch\n"
        "  O con CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118"
    )

print("\n" + "=" * 65)
print("  SECCION C: Red Neuronal fitnet(10) -- PyTorch")
print("  Arquitectura: 10 -> [10, tanh] -> 1 (lineal)")
print("=" * 65)

# =============================================================================
#  C.1 -- PREPARACION DE DATOS PYTORCH
#
#  Se reutilizan los arrays normalizados de Seccion B:
#    Xtr[:, 1:]  -> quitamos la columna de bias (PyTorch la maneja internamente)
#    Ytr, Yte   -> targets normalizados
# =============================================================================

# Quitar columna de bias agregada en Seccion B (columna 0 = unos)
# La red neuronal tiene bias propio en cada capa
Xtr_nn = torch.tensor(Xtr[:, 1:], dtype=torch.float32)   # (n_tr, 5)
Ytr_nn = torch.tensor(Ytr,        dtype=torch.float32).unsqueeze(1)  # (n_tr, 1)
Xte_nn = torch.tensor(Xte[:, 1:], dtype=torch.float32)   # (n_te, 5)
Yte_nn = torch.tensor(Yte,        dtype=torch.float32).unsqueeze(1)  # (n_te, 1)

# DataLoader para mini-batches (fitnet de MATLAB usa batch completo en LM;
# aqui usamos batch completo tambien para la comparacion mas justa)
dataset_tr = TensorDataset(Xtr_nn, Ytr_nn)
loader_tr  = DataLoader(dataset_tr, batch_size=len(Xtr_nn), shuffle=True)

print(f"\n  Tensores de entrenamiento: {Xtr_nn.shape}  ->  {Ytr_nn.shape}")
print(f"  Tensores de test:          {Xte_nn.shape}  ->  {Yte_nn.shape}")

# =============================================================================
#  C.2 -- DEFINICION DE LA RED (fitnet equivalente)
#
#  fitnet(10) en MATLAB:
#    - 1 capa oculta con 10 neuronas
#    - Activacion: tansig (= tanh)
#    - Capa de salida: purelin (= lineal)
#    - Inicializacion: Nguyen-Widrow (aproximamos con kaiming_uniform)
# =============================================================================

import torch.nn.functional as F

class FitNet(torch.nn.Module):
    def __init__(self, n_input, n_hidden=128, n_output=1):
        super(FitNet, self).__init__()
        # Ampliamos a 128 neuronas y agregamos una segunda capa oculta para llegar al >90%
        self.hidden1 = torch.nn.Linear(n_input, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden // 2)
        self.output  = torch.nn.Linear(n_hidden // 2, n_output)

    def forward(self, x):
        # Usamos ReLU en lugar de tanh, es el estándar en Deep Learning para regresión
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.output(x)

torch.manual_seed(42)
# Cambia el 5 por un 10 (o por X_n.shape[1] que tiene 10 columnas)
n_entradas = len(FEAT_NAMES) # Esto valdrá 10
modelo_nn = FitNet(n_input=n_entradas, n_hidden=128, n_output=1)

n_params = sum(p.numel() for p in modelo_nn.parameters())
print(f"\n  Parametros totales de la red: {n_params}")
print(f"  Detalle arquitectura:\n{modelo_nn}")

# =============================================================================
#  C.3 -- ENTRENAMIENTO CON L-BFGS
#
#  L-BFGS es el optimizador de PyTorch mas cercano a Levenberg-Marquardt:
#    - Segundo orden (usa curvatura como N-R y LM)
#    - Convergencia cuasi-cuadratica
#    - Apropiado para datasets que entran en memoria (batch completo)
#
#  Se agrega Early Stopping sobre el conjunto de validacion para replicar
#  el criterio de parada de MATLAB (validation checks).
# =============================================================================

criterion = nn.MSELoss()

# -- Conjunto de validacion (15%) para Early Stopping ------------------------
# Reutilizamos idx_vl definido en Seccion B
Xvl_nn = torch.tensor(X_b[idx[n_tr:n_tr + n_vl], 1:], dtype=torch.float32)
Yvl_nn = torch.tensor(Y_n[idx[n_tr:n_tr + n_vl]],     dtype=torch.float32).unsqueeze(1)

optimizer = torch.optim.LBFGS(
    modelo_nn.parameters(),
    lr=0.8,
    max_iter=20,          # iteraciones internas de busqueda lineal por paso
    history_size=10,      # memoria de L-BFGS (m en Nocedal & Wright)
    line_search_fn='strong_wolfe'
)

MAX_EPOCHS    = 500
PATIENCE      = 25        # early stopping: epocas sin mejora en validacion
TOL_NN        = 1e-7

loss_tr_hist  = []
loss_vl_hist  = []
best_vl_loss  = float('inf')
best_weights  = None
patience_cnt  = 0
epoch_conv    = 0

print(f"\n  Optimizador: L-BFGS (lr=0.8, history=10, Wolfe line search)")
print(f"  Early stopping: patience={PATIENCE}, TOL={TOL_NN}")
print(f"\n  {'Epoca':<8} {'MSE Train':>14} {'MSE Val':>14}  Estado")
print("  " + "-" * 50)

for epoch in range(1, MAX_EPOCHS + 1):

    # -- Paso L-BFGS (requiere closure) --------------------------------------
    def closure():
        optimizer.zero_grad()
        pred = modelo_nn(Xtr_nn)
        loss = criterion(pred, Ytr_nn)
        loss.backward()
        return loss

    optimizer.step(closure)

    # -- Metricas de esta epoca -----------------------------------------------
    with torch.no_grad():
        loss_tr = criterion(modelo_nn(Xtr_nn), Ytr_nn).item()
        loss_vl = criterion(modelo_nn(Xvl_nn), Yvl_nn).item()

    loss_tr_hist.append(loss_tr)
    loss_vl_hist.append(loss_vl)
    epoch_conv = epoch

    # -- Early stopping -------------------------------------------------------
    estado = ""
    if loss_vl < best_vl_loss - TOL_NN:
        best_vl_loss = loss_vl
        best_weights = {k: v.clone() for k, v in modelo_nn.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            estado = "<-- EARLY STOP"
            if epoch == 1 or epoch % 20 == 0 or estado:
                print(f"  {epoch:<8} {loss_tr:>14.6e} {loss_vl:>14.6e}  {estado}")
            break

    if epoch == 1 or epoch % 20 == 0 or estado:
        print(f"  {epoch:<8} {loss_tr:>14.6e} {loss_vl:>14.6e}  {estado}")

# Restaurar mejores pesos (criterio MATLAB: mejor performance en validacion)
if best_weights is not None:
    modelo_nn.load_state_dict(best_weights)
    print(f"\n  Pesos restaurados al minimo de validacion.")

# =============================================================================
#  C.4 -- EVALUACION EN TEST SET
# =============================================================================

with torch.no_grad():
    Ypred_nn_norm = modelo_nn(Xte_nn).numpy().flatten()

Ypred_nn_usd = desnorm(Ypred_nn_norm)

rmse_nn = np.sqrt(np.mean((Yreal_te_usd - Ypred_nn_usd) ** 2))
mae_nn  = np.mean(np.abs(Yreal_te_usd - Ypred_nn_usd))
r2_nn   = 1 - np.sum((Yreal_te_usd - Ypred_nn_usd) ** 2) / \
              np.sum((Yreal_te_usd - Yreal_te_usd.mean()) ** 2)

print(f"\n  Epocas hasta convergencia: {epoch_conv}")
print(f"  RMSE Test (fitnet): {fmt_usd(rmse_nn)}")
print(f"  MAE  Test (fitnet): {fmt_usd(mae_nn)}")
print(f"  R2   Test (fitnet): {r2_nn:.4f}")

# =============================================================================
#  C.5 -- TABLA COMPARATIVA ACTUALIZADA (N-R vs SGD vs fitnet)
# =============================================================================

print("\n" + "=" * 65)
print("  TABLA COMPARATIVA AMPLIADA -- N-R / SGD / fitnet(10)")
print("=" * 65)
print(f"\n  {'Criterio':<28} {'N-R lineal':>14} {'SGD lineal':>14} {'fitnet NN':>12}")
print("  " + "-" * 72)
rows_c = [
    ("Tipo de modelo",       "Lineal",          "Lineal",          "No lineal"),
    ("Arquitectura",         "10 features+bias","10 features+bias","10-[128-64]-1 ReLU"),
    ("Optimizador",          "N-R coordinado",  f"GD lr={LR_SGD}", "L-BFGS"),
    ("Iteraciones/Epocas",   str(n_conv_mv),    str(sgd_conv),     str(epoch_conv)),
    ("RMSE Test (USD)",      fmt_usd(rmse),     fmt_usd(rmse_sgd), fmt_usd(rmse_nn)),
    ("R2 Test",              f"{r2:.4f}",        "---",             f"{r2_nn:.4f}"),
    ("Convergencia",         "Cuadratica",      "Lineal",          "Cuasi-cuadratica"),
    ("Requiere tuning",      "No",              "Si (lr)",         "Minimo (patience)"),
]
for row in rows_c:
    print(f"  {row[0]:<28} {row[1]:>14} {row[2]:>14} {row[3]:>12}")

# =============================================================================
#  C.6 -- FIGURAS ADICIONALES
# =============================================================================

# --- Figura 6: Curvas de aprendizaje (train vs val) --------------------------
fig, ax = plt.subplots(figsize=(9, 4))
epocas = range(1, len(loss_tr_hist) + 1)
ax.semilogy(epocas, loss_tr_hist, color='#2E86AB', lw=2, label='MSE Train')
ax.semilogy(epocas, loss_vl_hist, color='#E84855', lw=2, label='MSE Validacion', linestyle='--')
ax.scatter(Yreal_te_usd / 1e3, Ypred_nn_usd / 1e3,
           alpha=0.3, s=8, color='#9B5DE5', label="Red Neuronal (128)")
# ... (deja la linea del lim_nn y plot igual) ...
ax.set_title(
    f"Figura 7: Test Set KC -- Red Neuronal Profunda\nR2 = {r2_nn:.4f}  |  RMSE = {fmt_usd(rmse_nn)}",
    fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.text(0.60, 0.75,
        f"Epocas: {epoch_conv}\nRMSE Test: {fmt_usd(rmse_nn)}\nR2: {r2_nn:.4f}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray'))
plt.tight_layout()
plt.savefig("figura6_fitnet_learning_curves.png", bbox_inches='tight', dpi=130)
plt.show()

# --- Figura 7: Pred vs Real -- fitnet ----------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(Yreal_te_usd / 1e3, Ypred_nn_usd / 1e3,
           alpha=0.3, s=8, color='#9B5DE5', label="fitnet(10)")
lim_nn = [0, max(Yreal_te_usd.max(), Ypred_nn_usd.max()) / 1e3 * 1.05]
ax.plot(lim_nn, lim_nn, 'r--', lw=2, label="y = x (ideal)")
ax.set_xlabel("Precio real (miles USD)", fontsize=11)
ax.set_ylabel("Prediccion fitnet (miles USD)", fontsize=11)
ax.set_title(
    f"Figura 7: Test Set KC -- fitnet(10)\nR2 = {r2_nn:.4f}  |  RMSE = {fmt_usd(rmse_nn)}",
    fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figura7_fitnet_pred_vs_real.png", bbox_inches='tight', dpi=130)
plt.show()

# --- Figura 8: Comparativa RMSE -- N-R lineal vs fitnet NN -------------------
fig, ax = plt.subplots(figsize=(7, 4))
modelos_comp = ['N-R\n(lineal)', 'SGD\n(lineal)', 'fitnet(10)\n(no lineal)']
rmses_comp   = [rmse / 1e3, rmse_sgd / 1e3, rmse_nn / 1e3]
colores_comp = ['#2E86AB', '#E84855', '#9B5DE5']
barras = ax.bar(modelos_comp, rmses_comp, color=colores_comp, width=0.45, edgecolor='white')
for bar, val in zip(barras, rmses_comp):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"${val:,.1f}k", ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel("RMSE Test (miles USD)", fontsize=11)
ax.set_title(
    "Figura 8: Comparativa RMSE -- Regresion lineal (N-R) vs fitnet(10)\n"
    "KC House Data -- Test set 15%",
    fontsize=11, fontweight='bold')
ax.set_ylim(0, max(rmses_comp) * 1.20)
plt.tight_layout()
plt.savefig("figura8_comparativa_rmse.png", bbox_inches='tight', dpi=130)
plt.show()

print("\n" + "=" * 65)
print("  RESUMEN SECCION C -- fitnet(10) PyTorch")
print("=" * 65)
print(f"\n  Arquitectura:   5 -> [10 neuronas, tanh] -> 1 (lineal)")
print(f"  Optimizador:    L-BFGS (Wolfe line search)")
print(f"  Early stopping: patience={PATIENCE}")
print(f"  Epocas:         {epoch_conv}")
print(f"  RMSE Test:      {fmt_usd(rmse_nn)}")
print(f"  MAE  Test:      {fmt_usd(mae_nn)}")
print(f"  R2   Test:      {r2_nn:.4f}")
print(f"\n  Mejora RMSE sobre N-R lineal: {fmt_usd(rmse - rmse_nn)}")
print(f"  Mejora R2      sobre N-R lineal: {r2_nn - r2:+.4f}")
print()
print("  Referencia: Camelino, Funes, Ziletti (2026)")
print("  UCC . Metodos Numericos . Grupo 5")
print("=" * 65)

# =============================================================================
#  PASO 7 -- FIGURAS
# =============================================================================

# --- Figura 1: L(w) individual con trayectoria del w GLOBAL para cada casa --
#    El eje x muestra el w global; la curva es L individual de cada casa.
#    Los puntos muestran como el w global evalua cada L_k en cada iteracion.
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle(
    "Figura 1: L_k(w) individual con trayectoria del w* global -- 5 propiedades KC\n"
    r"(w* unico minimiza MSE global; cada curva es la perdida individual $L_k(w)$)",
    fontsize=12, fontweight='bold', y=1.04)

for k, ax in enumerate(axes):
    r  = resultados[k]
    Yr, Yz = r["Yr"], r["Yz"]
    w_range = np.linspace(0.5, 1.30, 400)
    L_curve = np.array([L_ind(w, Yr, Yz) for w in w_range])
    ax.plot(w_range, L_curve / 1e9, color='#CCCCCC', lw=2, label=r"$L_k(w)$")
    ax.scatter(r["w_hist"], r["L_hist"] / 1e9,
               color=COLORES[k], zorder=5, s=50,
               label="Iteraciones N-R\n(w global)")
    ax.annotate(f"w0={r['w0']:.3f}", (r["w_hist"][0], r["L_hist"][0] / 1e9),
                textcoords="offset points", xytext=(5, 5), fontsize=7.5)
    ax.annotate(f"w*={r['wstar']:.4f}", (r["wstar"], r["L_hist"][-1] / 1e9),
                textcoords="offset points", xytext=(0, 10), fontsize=7.5,
                color=COLORES[k], fontweight='bold')
    ax.set_xlabel("Peso w (global)", fontsize=9)
    ax.set_ylabel(r"$L_k(w)$ [$\times 10^9$]", fontsize=9)
    ax.set_title(f"P{k+1}: {fmt_usd(Yr)}", fontsize=9, fontweight='bold')
    ax.legend(fontsize=6.5, loc='upper center')

plt.tight_layout()
plt.savefig("figura1_perdida_propiedades.png", bbox_inches='tight', dpi=130)
plt.show()

# --- Figura 2: Convergencia del error relativo (%) por propiedad -------------
#    Cada curva muestra como cae el error_k(w_global_i) a medida que el
#    optimizador global avanza hacia w*.
fig, ax = plt.subplots(figsize=(9, 5))
for k, r in enumerate(resultados):
    iters = range(len(r["err_hist"]))
    ax.plot(iters, r["err_hist"], 'o-', color=COLORES[k], lw=2,
            markersize=7, markerfacecolor=COLORES[k],
            label=f"P{k+1}: {fmt_usd(r['Yr'])} (err0={r['err0']:.0f}%)")
ax.set_xlabel("Iteracion (w global)", fontsize=11)
ax.set_ylabel("Error relativo individual (%)", fontsize=11)
ax.set_title(
    "Figura 2: Error por propiedad segun w* global -- N-R amortiguado (alpha=0.25)\n"
    "Cada curva: error_k(w_global_i). Las 5 casas convergen al compromiso optimo.",
    fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.set_ylim(bottom=-0.5)
plt.tight_layout()
plt.savefig("figura2_convergencia_propiedades.png", bbox_inches='tight', dpi=130)
plt.show()

# --- Figura 3: MSE del modelo multivariado vs iteraciones --------------------
fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogy(range(1, len(loss_hist_mv) + 1), loss_hist_mv,
            color='#2E86AB', lw=2)
ax.set_xlabel("Iteracion", fontsize=11)
ax.set_ylabel("MSE (espacio normalizado)", fontsize=11)
ax.set_title("Figura 3: Convergencia N-R coordinado -- Modelo KC multivariado",
             fontsize=12, fontweight='bold')
ax.text(0.65, 0.80,
        f"Convergencia: {n_conv_mv} iteraciones\nR2 = {r2:.4f}",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray'))
plt.tight_layout()
plt.savefig("figura3_convergencia_multivariado.png", bbox_inches='tight', dpi=130)
plt.show()

# --- Figura 4: Predicciones vs valores reales (test set) ---------------------
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(Yreal_te_usd / 1e3, Ypred_te_usd / 1e3,
           alpha=0.3, s=8, color='#2E86AB', label="Propiedades KC")
lim = [0, max(Yreal_te_usd.max(), Ypred_te_usd.max()) / 1e3 * 1.05]
ax.plot(lim, lim, 'r--', lw=2, label="y = x (ideal)")
ax.set_xlabel("Precio real (miles USD)", fontsize=11)
ax.set_ylabel("Prediccion N-R (miles USD)", fontsize=11)
ax.set_title(f"Figura 4: Test Set KC -- R2 = {r2:.4f}  |  RMSE = {fmt_usd(rmse)}",
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figura4_pred_vs_real.png", bbox_inches='tight', dpi=130)
plt.show()

# --- Figura 5: Barras comparativas N-R vs SGD --------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
metodos = ['Newton-Raphson', 'SGD']
iters   = [n_conv_mv, sgd_conv]
barras  = ax.bar(metodos, iters, color=['#2E86AB', '#E84855'], width=0.45, edgecolor='white')
for bar, val in zip(barras, iters):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(val), ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylabel("Iteraciones hasta convergencia", fontsize=11)
ax.set_title("Figura 5: Iteraciones -- N-R vs SGD (modelo KC multivariado)",
             fontsize=11, fontweight='bold')
ax.set_ylim(0, max(iters) * 1.15)
plt.tight_layout()
plt.savefig("figura5_nr_vs_sgd.png", bbox_inches='tight', dpi=130)
plt.show()

# =============================================================================
#  PASO 8 -- RESUMEN FINAL
# =============================================================================

print("=" * 65)
print("  RESUMEN FINAL")
print("=" * 65)
print(f"\n  Dataset:        KC House Data -- King County, WA")
print(f"  Propiedades:    {N:,}  |  Features: {len(FEAT_NAMES)}")
print(f"  Normalizacion:  mapminmax [-1, 1]")
print(f"  Particion:      70/15/15 (train/val/test)\n")
print("  -- SECCION A: N-R Global amortiguado (alpha=0.25, MSE sobre 5 casas) --")
print(f"     w0 = {W0_GLOBAL:.4f}  ->  w* = {w_star:.8f}  ({n_iter} iteraciones)")
print(f"     MSE global final: {L_global(w_star, Y_REAL_USD, Y_ZEST_USD):.4e}")
print()
for k, r in enumerate(resultados):
    pred_i = r['w0']    * r['Yz']
    pred_f = r['wstar'] * r['Yz']
    err_f  = abs(r['Yr'] - pred_f) / r['Yr'] * 100
    print(f"    P{k+1}  {fmt_usd(r['Yr']):>12}  "
          f"pred_i={fmt_usd(pred_i)}  pred_f={fmt_usd(pred_f)}  err_f={err_f:.2f}%")
print()
print("  -- SECCION B: multivariado N-R coordinado ------------------")
print(f"    Iteraciones:  {n_conv_mv}")
print(f"    RMSE Test:    {fmt_usd(rmse)}")
print(f"    MAE  Test:    {fmt_usd(mae)}")
print(f"    R2   Test:    {r2:.4f}")
print()
print("  Referencia: Camelino, Funes, Ziletti (2026)")
print("  UCC . Metodos Numericos . Grupo 5")
print("=" * 65)