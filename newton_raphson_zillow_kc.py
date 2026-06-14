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
    # Actualizacion coordinada (Gauss-Seidel: usa W actualizado en cada paso)
    for j in range(p):
        xj      = Xtr[:, j]
        resid_j = Ytr - Xtr @ W
        dL_j    = -(2 / len(Ytr)) * np.dot(xj, resid_j)
        d2L_j   =  (2 / len(Ytr)) * np.dot(xj, xj)
        W[j]   -= dL_j / d2L_j

    # Gradiente y MSE evaluados DESPUES de los updates de esta iteracion
    resid = Ytr - Xtr @ W
    mse   = np.mean(resid ** 2)
    g     = -(2 / len(Ytr)) * (Xtr.T @ resid)

    loss_hist_mv.append(mse)
    grad_hist_mv.append(np.linalg.norm(g))

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
#  SECCION C -- RED NEURONAL fitnet2({N_H1_LM},{N_H2_LM}) CON LEVENBERG-MARQUARDT
#
#  El algoritmo Levenberg-Marquardt (LM) usa la JACOBIANA EXPLICITA:
#    J[i,k] = d(r_i)/d(W_k)  donde r_i = y_pred_i - y_real_i
#
#  Actualizacion: (J^T J + mu*I) DW = -J^T r
#  La Jacobiana se calcula analiticamente para la red tanh de 1 capa oculta.
#
#  RESTRICCION de parametros: J crece como N*P.
#  Con P=321 y N_J=2000: J es 2000x321 ~ 5 MB (float64), sistema 321x321.
# =============================================================================

# -- Importacion de PyTorch ---------------------------------------------------
try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError(
        "\n  PyTorch no esta instalado.\n"
        "  Ejecuta: pip install torch\n"
    )

N_H1_LM = 32   # neuronas capa oculta 1
N_H2_LM = 22   # neuronas capa oculta 2
N_H3_LM = 14   # neuronas capa oculta 3

print("\n" + "=" * 65)
print(f"  SECCION C: fitnet2({N_H1_LM},{N_H2_LM}) con Levenberg-Marquardt")
print("  Jacobiana analitica: J[i,k] = d(r_i)/d(W_k), forma (N_J, P)")
print("=" * 65)

# =============================================================================
#  C.1 -- FEATURES AMPLIADOS (27) CON FEATURE ENGINEERING
#
#  Feature clave: precio mediano por zipcode (calculado solo en training
#  para evitar data leakage). Captura efectos de mercado local mejor que
#  las coordenadas continuas lat/long para una red tanh.
#  Transformaciones log sobre sqft capturan la escala no lineal del precio.
# =============================================================================

FEAT_NAMES_LM = ['sqft_living', 'grade', 'bathrooms', 'bedrooms',
                 'lat', 'long', 'waterfront', 'view', 'condition',
                 'sqft_living15', 'floors', 'sqft_above']

Y_raw_lm = np.log1p(df_clean['price'].values.astype(float))
N_lm     = len(Y_raw_lm)

# Particion 70/15/15 (antes de computar zipcode para evitar leakage)
rng_lm_gen = np.random.default_rng(42)
idx_lm_all = rng_lm_gen.permutation(N_lm)
n_tr_lm    = int(0.70 * N_lm)
n_vl_lm    = int(0.15 * N_lm)
idx_tr_lm  = idx_lm_all[:n_tr_lm]
idx_vl_lm  = idx_lm_all[n_tr_lm:n_tr_lm + n_vl_lm]
idx_te_lm  = idx_lm_all[n_tr_lm + n_vl_lm:]

# Precio mediano y std por zipcode -- calculado SOLO en training set
df_tr_tmp       = df_clean.iloc[idx_tr_lm]
zip_med_train   = np.log1p(df_tr_tmp.groupby('zipcode')['price'].median())
zip_std_train   = df_tr_tmp.groupby('zipcode')['price'].apply(
                      lambda x: np.log1p(x).std()).fillna(0)
zip_feat_series = df_clean['zipcode'].map(zip_med_train)
zip_std_series  = df_clean['zipcode'].map(zip_std_train)
zip_feat_series.fillna(zip_med_train.median(), inplace=True)
zip_std_series.fillna(zip_std_train.median(), inplace=True)
zip_feat_lm     = zip_feat_series.values.reshape(-1, 1).astype(float)
zip_std_lm      = zip_std_series.values.reshape(-1, 1).astype(float)

# Construccion de X con todas las features (23 total)
X_base_lm    = df_clean[FEAT_NAMES_LM].values.astype(float)
house_age_lm = (2015 - df_clean['yr_built'].values).astype(float).reshape(-1, 1)
renovated_lm = (df_clean['yr_renovated'].values > 0).astype(float).reshape(-1, 1)
# Log-transform de sqft: captura efecto de escala (precio crece sub-linealmente en sqft)
log_sqft_living = np.log1p(df_clean['sqft_living'].values).astype(float).reshape(-1, 1)
log_sqft_lot    = np.log1p(df_clean['sqft_lot'].values).astype(float).reshape(-1, 1)
# Ratio sqft_living15/sqft_living: como la casa se compara con vecinos
sqft_ratio_15   = (df_clean['sqft_living15'].values /
                   df_clean['sqft_living'].values.clip(1)).astype(float).reshape(-1, 1)
# Interaccion grade x sqft_living (normalizada): premium de calidad en casas grandes
grade_x_sqft    = (df_clean['grade'].values *
                   df_clean['sqft_living'].values / 1e5).astype(float).reshape(-1, 1)
# Superficie del sotano (log): valor adicional de espacio habitable
log_sqft_bsmt   = np.log1p(df_clean['sqft_basement'].values).astype(float).reshape(-1, 1)
# Tamaño lote vecinos (log): contexto de densidad del barrio
log_sqft_lot15  = np.log1p(df_clean['sqft_lot15'].values).astype(float).reshape(-1, 1)
# Mes de venta (codificacion ciclica): estacionalidad del mercado inmobiliario
import pandas as _pd
_months = _pd.to_datetime(df_clean['date']).dt.month.values
sale_month_sin  = np.sin(2 * np.pi * _months / 12).astype(float).reshape(-1, 1)
sale_month_cos  = np.cos(2 * np.pi * _months / 12).astype(float).reshape(-1, 1)
# Edad efectiva: si reformada, usa edad desde reforma; si no, edad desde construccion
_yr_ren = df_clean['yr_renovated'].values
eff_age = np.where(_yr_ren > 0, 2015 - _yr_ren,
                   2015 - df_clean['yr_built'].values).astype(float).reshape(-1, 1)
# Interaccion grade x condition: calidad ESTRUCTURAL (grado) × estado actual
grade_x_cond    = (df_clean['grade'].values *
                   df_clean['condition'].values).astype(float).reshape(-1, 1)
# Distancia al centro de Seattle (lat 47.6062, long -122.3321): premium urbano
_dlat = (df_clean['lat'].values - 47.6062)
_dlon = (df_clean['long'].values - (-122.3321)) * np.cos(np.radians(47.6062))
dist_center     = np.sqrt(_dlat**2 + _dlon**2).astype(float).reshape(-1, 1)

# Feature kNN geo-price: precio log ponderado por distancia de k=15 vecinos mas
# cercanos (lat/long). LOO sobre training para evitar data leakage.
print("  Computando kNN geo-price (k=15, LOO)...", flush=True)
_K_KNN  = 15
_ll_all = np.column_stack([df_clean['lat'].values, df_clean['long'].values])
_ll_tr  = _ll_all[idx_tr_lm]
_lp_tr  = Y_raw_lm[idx_tr_lm]
_N_all  = len(_ll_all)
_CHUNK  = 500
_tr_pos = {g: i for i, g in enumerate(idx_tr_lm)}
knn_geo = np.zeros(_N_all)

for _s in range(0, _N_all, _CHUNK):
    _e   = min(_s + _CHUNK, _N_all)
    _pts = _ll_all[_s:_e]
    _d   = np.sqrt(((_ll_tr[None,:,:] - _pts[:,None,:])**2).sum(axis=2))
    for _j in range(_e - _s):
        _dj = _d[_j].copy()
        _gi = _s + _j
        if _gi in _tr_pos:
            _dj[_tr_pos[_gi]] = np.inf
        _ki = np.argpartition(_dj, _K_KNN)[:_K_KNN]
        _wi = 1.0 / (_dj[_ki] + 1e-8)
        knn_geo[_gi] = (_lp_tr[_ki] * _wi).sum() / _wi.sum()

print("  kNN feature listo.", flush=True)
knn_geo_feat = knn_geo.reshape(-1, 1)

# Polinomios lat/long grado 2: capturan la curvatura 2D del precio geografico.
# La superficie de precios de KC no es plana: el valor por metro cuadrado cambia
# de forma no lineal con las coordenadas (Elliott Bay, Redmond, Bellevue peaks).
_lat_raw = df_clean['lat'].values.astype(float)
_lon_raw = df_clean['long'].values.astype(float)
lat2_feat   = (_lat_raw ** 2).reshape(-1, 1)
lon2_feat   = (_lon_raw ** 2).reshape(-1, 1)
latlon_feat = (_lat_raw * _lon_raw).reshape(-1, 1)

X_raw_lm  = np.hstack([X_base_lm, house_age_lm, renovated_lm, zip_feat_lm,
                        log_sqft_living, log_sqft_lot, sqft_ratio_15, grade_x_sqft,
                        log_sqft_bsmt, log_sqft_lot15, sale_month_sin, sale_month_cos,
                        eff_age, grade_x_cond, dist_center,
                        knn_geo_feat, lat2_feat, lon2_feat, latlon_feat])
n_feat_lm = X_raw_lm.shape[1]   # 31
feat_names_all_lm = (FEAT_NAMES_LM +
                     ['house_age', 'renovated', 'zip_log_med_price',
                      'log_sqft_living', 'log_sqft_lot', 'sqft_ratio_15', 'grade_x_sqft',
                      'log_sqft_bsmt', 'log_sqft_lot15', 'month_sin', 'month_cos',
                      'eff_age', 'grade_x_cond', 'dist_center',
                      'knn_k15_geo', 'lat2', 'lon2', 'lat_x_lon'])

Xmin_lm    = X_raw_lm.min(axis=0)
Xmax_lm    = X_raw_lm.max(axis=0)
Ymin_lm    = Y_raw_lm.min()
Ymax_lm    = Y_raw_lm.max()
rng_lm_arr = Xmax_lm - Xmin_lm
rng_lm_arr[rng_lm_arr == 0] = 1.0
X_n_lm = 2 * (X_raw_lm - Xmin_lm) / rng_lm_arr - 1
Y_n_lm = 2 * (Y_raw_lm - Ymin_lm) / (Ymax_lm - Ymin_lm) - 1

# float64 para precision numerica en el sistema LM
Xtr_lm = torch.tensor(X_n_lm[idx_tr_lm], dtype=torch.float64)
Ytr_lm = torch.tensor(Y_n_lm[idx_tr_lm], dtype=torch.float64).unsqueeze(1)
Xvl_lm = torch.tensor(X_n_lm[idx_vl_lm], dtype=torch.float64)
Yvl_lm = torch.tensor(Y_n_lm[idx_vl_lm], dtype=torch.float64).unsqueeze(1)
Xte_lm = torch.tensor(X_n_lm[idx_te_lm], dtype=torch.float64)
Yte_lm = torch.tensor(Y_n_lm[idx_te_lm], dtype=torch.float64).unsqueeze(1)

print(f"\n  Features ({n_feat_lm}): {', '.join(feat_names_all_lm)}")
print(f"  Target:       log1p(price), mapminmax [-1, 1]")
print(f"  Train: {len(idx_tr_lm):,}  |  Val: {len(idx_vl_lm):,}  |  Test: {len(idx_te_lm):,}")

# =============================================================================
#  C.2 -- RED NEURONAL 3 CAPAS CON LeakyReLU, COMPATIBLE CON LM
#
#  fitnet3(28,20,14): 3 capas ocultas LeakyReLU + salida lineal.
#  LeakyReLU evita la saturacion del tanh en valores extremos, mejorando el
#  flujo de gradiente en la Jacobiana. La derivada es constante por partes:
#    d/dz = 1  si z > 0  /  0.01  si z <= 0  (sin saturacion)
#
#  Con 29 features: P = 29*32+32 + 32*22+22 + 22*14+14 + 14+1 = 1993 params.
#  Jacobiana full-batch: ~22447 x 1993 ~ 357 MB float64. Manejable con jac_batch.
#  3 capas: mas capacidad representativa con parametros similares a 2 capas.
# =============================================================================

ALPHA_LRELU = 0.01   # pendiente negativa de LeakyReLU

class FitNetLM2(nn.Module):
    """fitnet 3 capas LeakyReLU: n -> [H1] -> [H2] -> [H3] -> 1 (lineal).
    Llamada FitNetLM2 por compatibilidad con el resto del codigo."""
    def __init__(self, n_input, n_h1=28, n_h2=20, n_h3=14):
        super().__init__()
        self.layer1 = nn.Linear(n_input, n_h1)
        self.layer2 = nn.Linear(n_h1, n_h2)
        self.layer3 = nn.Linear(n_h2, n_h3)
        self.output = nn.Linear(n_h3, 1)
        self.lrelu  = nn.LeakyReLU(ALPHA_LRELU)
        # He initialization (Kaiming) para activaciones tipo ReLU
        nn.init.kaiming_uniform_(self.layer1.weight, a=ALPHA_LRELU, nonlinearity='leaky_relu')
        nn.init.zeros_(self.layer1.bias)
        nn.init.kaiming_uniform_(self.layer2.weight, a=ALPHA_LRELU, nonlinearity='leaky_relu')
        nn.init.zeros_(self.layer2.bias)
        nn.init.kaiming_uniform_(self.layer3.weight, a=ALPHA_LRELU, nonlinearity='leaky_relu')
        nn.init.zeros_(self.layer3.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        h1 = self.lrelu(self.layer1(x))
        h2 = self.lrelu(self.layer2(h1))
        h3 = self.lrelu(self.layer3(h2))
        return self.output(h3)

torch.manual_seed(42)
model_lm    = FitNetLM2(n_input=n_feat_lm, n_h1=N_H1_LM, n_h2=N_H2_LM, n_h3=N_H3_LM).double()
n_params_lm = sum(param.numel() for param in model_lm.parameters())

print(f"\n  Arquitectura: {n_feat_lm} -> [{N_H1_LM},lrelu] -> [{N_H2_LM},lrelu] -> [{N_H3_LM},lrelu] -> 1")
print(f"  Parametros P = {n_params_lm}")
print(f"  Detalle:\n{model_lm}")

# =============================================================================
#  C.3 -- LEVENBERG-MARQUARDT CON JACOBIANA ANALITICA (3 CAPAS LeakyReLU)
#
#  La Jacobiana d(r_i)/d(W_k) se computa de forma cerrada (backprop manual):
#
#    d_lrelu(z) = 1 si z>0, alpha si z<=0   (LeakyReLU, sin saturacion)
#
#    delta3_{i,j} = W4_{0,j} * d_lrelu(z3_{i,j})          [∂y/∂z3]
#    delta2_{i,j} = sum_k W3_{k,j} * delta3_{i,k} * d_lrelu(z2_{i,j})
#    delta1_{i,j} = sum_k W2_{k,j} * delta2_{i,k} * d_lrelu(z1_{i,j})
#
#  Bloques de J (concatenados, coinciden con model.parameters()):
#    dW1[i,j,k] = delta1[i,j] * x[i,k]        -> (N, H1*n)
#    db1[i,j]   = delta1[i,j]                  -> (N, H1)
#    ... idem para capas 2, 3 y output
#
#  Regularizacion L2 integrada: (J^T J + (mu+lam_N)*I) dW = -(J^T r + lam_N*W)
#
#  Referencia: Hagan & Menhaj (1994). "Training feedforward networks with
#              the Marquardt algorithm." IEEE TNN 5(6):989-993.
# =============================================================================

def jacobian_analitico(model, x, jac_batch=3000):
    """
    Jacobiana analitica para FitNetLM2 (3 capas LeakyReLU), forma (N, P).
    Procesada en mini-bloques para controlar pico de memoria.

    Backprop manual con LeakyReLU (derivada constante por partes):
      d_lrelu(z) = 1  si z > 0  /  alpha  si z <= 0

      delta3 = W4 * d_lrelu(z3)                       [∂y/∂z3]
      delta2 = (delta3 @ W3) * d_lrelu(z2)            [∂y/∂z2]
      delta1 = (delta2 @ W2) * d_lrelu(z1)            [∂y/∂z1]

    Orden en J coincide con model.parameters():
      layer1.{weight,bias} | layer2.{weight,bias} | layer3.{weight,bias} | output.{weight,bias}
    """
    W1 = model.layer1.weight.data    # (H1, n)
    b1 = model.layer1.bias.data
    W2 = model.layer2.weight.data    # (H2, H1)
    b2 = model.layer2.bias.data
    W3 = model.layer3.weight.data    # (H3, H2)
    b3 = model.layer3.bias.data
    W4 = model.output.weight.data    # (1,  H3)
    H1, n = W1.shape
    H2    = W2.shape[0]
    H3    = W3.shape[0]
    N     = x.shape[0]
    P     = H1*n + H1 + H2*H1 + H2 + H3*H2 + H3 + H3 + 1

    J = torch.zeros(N, P, dtype=x.dtype)
    alpha = x.new_tensor(ALPHA_LRELU)

    for s in range(0, N, jac_batch):
        e   = min(s + jac_batch, N)
        xb  = x[s:e]
        nb  = e - s

        z1 = xb @ W1.T + b1                            # (nb, H1)
        h1 = torch.where(z1 > 0, z1, alpha * z1)      # LeakyReLU
        z2 = h1 @ W2.T + b2                            # (nb, H2)
        h2 = torch.where(z2 > 0, z2, alpha * z2)
        z3 = h2 @ W3.T + b3                            # (nb, H3)
        h3 = torch.where(z3 > 0, z3, alpha * z3)

        # Derivadas de LeakyReLU: 1 si z>0, alpha si z<=0
        d1 = torch.where(z1 > 0, torch.ones_like(z1), alpha * torch.ones_like(z1))
        d2 = torch.where(z2 > 0, torch.ones_like(z2), alpha * torch.ones_like(z2))
        d3 = torch.where(z3 > 0, torch.ones_like(z3), alpha * torch.ones_like(z3))

        delta3 = W4 * d3                               # (nb, H3): ∂y/∂z3
        delta2 = (delta3 @ W3) * d2                   # (nb, H2): ∂y/∂z2
        delta1 = (delta2 @ W2) * d1                   # (nb, H1): ∂y/∂z1

        dW1 = (delta1.unsqueeze(2) * xb.unsqueeze(1)).reshape(nb, -1)
        db1 = delta1
        dW2 = (delta2.unsqueeze(2) * h1.unsqueeze(1)).reshape(nb, -1)
        db2 = delta2
        dW3 = (delta3.unsqueeze(2) * h2.unsqueeze(1)).reshape(nb, -1)
        db3 = delta3
        dW4 = h3                                       # (nb, H3)
        db4 = torch.ones(nb, 1, dtype=x.dtype)

        J[s:e] = torch.cat([dW1, db1, dW2, db2, dW3, db3, dW4, db4], dim=1)

    return J   # (N, P)

def get_flat_params(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()])

def set_flat_params(model, flat_params):
    offset = 0
    for param in model.parameters():
        n = param.numel()
        param.data.copy_(flat_params[offset:offset + n].view(param.shape))
        offset += n

def mse_completo(model, x, y):
    with torch.no_grad():
        return ((model(x) - y) ** 2).mean().item()

def desnorm_lm(y_norm):
    """Revierte mapminmax [-1,1] y luego revierte log1p. Para Seccion C."""
    y_log = (y_norm + 1) / 2 * (Ymax_lm - Ymin_lm) + Ymin_lm
    return np.expm1(y_log)

def run_lm_training(model, Xtr, Ytr, Xvl, Yvl, n_params,
                    mu_init=0.01, max_iter=300, patience=60,
                    tol_grad=1e-9, nu_dec=0.1, nu_inc=10.0,
                    mu_min=1e-12, mu_max=1e10, lambda_reg=0.0, verbose=True):
    """Entrena con Levenberg-Marquardt usando Jacobiana analitica (full-batch).

    Regularizacion L2 (Tikhonov) integrada en el sistema LM:
      (J^T J + (mu + lam_N)*I) dW = -(J^T r + lam_N*W)
    donde lam_N = lambda_reg * N. Es el equivalente bayesiano al dropout para
    optimizadores de segundo orden: penaliza pesos grandes sin romper la
    consistencia de la Jacobiana analitica.

    Retorna (best_W, best_val_mse, tr_hist, vl_hist, n_iters, n_acc, n_rej, mu_final).
    """
    mu = mu_init
    best_vl = float('inf')
    best_W = get_flat_params(model).clone()
    patience_cnt = 0
    n_acc = 0
    n_rej = 0
    n_iters = 0
    tr_hist = []
    vl_hist = []
    N_tr = len(Xtr)
    lam_N = lambda_reg * N_tr   # escala para que sea comparable con JTJ ~ O(N)

    for it in range(1, max_iter + 1):
        with torch.no_grad():
            J = jacobian_analitico(model, Xtr)
            r = model(Xtr).squeeze() - Ytr.squeeze()

        W_curr  = get_flat_params(model)
        mse_old = (r ** 2).mean().item()
        # Loss regularizada (para criterio de aceptacion)
        E_old   = mse_old + (lambda_reg / 2) * W_curr.pow(2).sum().item()

        # Sistema LM con Tikhonov: (JTJ + (mu+lam_N)*I) dW = -(JTr + lam_N*W)
        JTJ = J.T @ J
        JTr = J.T @ r + lam_N * W_curr
        A   = JTJ + (mu + lam_N) * torch.eye(n_params, dtype=torch.float64)

        try:
            delta_W = torch.linalg.solve(A, -JTr)
        except torch.linalg.LinAlgError:
            mu = min(mu * nu_inc, mu_max)
            continue

        W_old = get_flat_params(model).clone()
        set_flat_params(model, W_old + delta_W)

        with torch.no_grad():
            r_new   = model(Xtr).squeeze() - Ytr.squeeze()
            mse_new = (r_new ** 2).mean().item()
        W_new = get_flat_params(model)
        E_new = mse_new + (lambda_reg / 2) * W_new.pow(2).sum().item()

        if E_new < E_old:
            mu = max(mu * nu_dec, mu_min)
            n_acc += 1
        else:
            set_flat_params(model, W_old)
            mu = min(mu * nu_inc, mu_max)
            n_rej += 1

        # Validacion: MSE puro (sin reg) para comparar modelos con distinto lambda
        loss_tr = mse_completo(model, Xtr, Ytr)
        loss_vl = mse_completo(model, Xvl, Yvl)
        tr_hist.append(loss_tr)
        vl_hist.append(loss_vl)

        if loss_vl < best_vl:
            best_vl = loss_vl
            best_W  = get_flat_params(model).clone()
            patience_cnt = 0
        else:
            patience_cnt += 1

        n_iters   = it
        grad_norm = JTr.norm().item()

        if verbose and (it == 1 or it % 25 == 0 or patience_cnt >= patience):
            print(f"  {it:<6} {loss_tr:>14.6e} {loss_vl:>14.6e} {mu:>12.2e}")

        if patience_cnt >= patience:
            if verbose:
                print(f"\n  Early stopping: {patience} iters sin mejora en validacion.")
            break
        if grad_norm < tol_grad:
            if verbose:
                print(f"\n  Convergencia: ||J^T r|| = {grad_norm:.2e} < TOL")
            break

    return best_W, best_vl, tr_hist, vl_hist, n_iters, n_acc, n_rej, mu

# -- Hiperparametros LM -------------------------------------------------------
MU_INIT     = 0.01     # amortiguamiento inicial (Hagan & Menhaj 1994)
MU_MIN      = 1e-12
MU_MAX      = 1e10
NU_DEC      = 0.1      # reduccion de mu al aceptar paso
NU_INC      = 10.0     # incremento de mu al rechazar paso
MAX_ITER_LM = 300
TOL_GRAD_LM = 1e-9
PATIENCE_LM = 120      # epocas sin mejora en validacion

N_TR_LM = len(Xtr_lm)

# -- Augmentacion de datos: +50% copias con ruido gaussiano ------------------
#  Ruido en espacio normalizado [-1,1]; preserva la estructura local sin
#  inventar nuevos precios. Expande el training set de 14965 a 22447 muestras.
SIGMA_AUG = 0.004   # ruido reducido: evita que Phase 2 aprenda el ruido en vez de la señal
torch.manual_seed(0)
Xtr_aug_noise = torch.clamp(
    Xtr_lm + SIGMA_AUG * torch.randn_like(Xtr_lm), -1.1, 1.1)
Xtr_aug = torch.cat([Xtr_lm, Xtr_aug_noise])
Ytr_aug = torch.cat([Ytr_lm, Ytr_lm])   # mismo precio para copia ruidosa
N_TR_AUG = len(Xtr_aug)

print(f"\n  Jacobiana FULL-BATCH: N_orig={N_TR_LM}, N_aug={N_TR_AUG}, P={n_params_lm}")
print(f"  mu_init={MU_INIT}, NU_dec={NU_DEC}, NU_inc={NU_INC}")
print(f"  Augmentacion: +{len(Xtr_aug_noise)} muestras con ruido σ={SIGMA_AUG}")

# -- Fase 0: cross-validation de lambda_reg sobre el validation set ----------
#  Grid search: lambda_reg x semillas x iteraciones cortas.
#  Dropout no es compatible con la Jacobiana analitica (J seria estocastica);
#  la regularizacion L2 (Tikhonov) es el equivalente bayesiano para LM.
LAMBDA_GRID    = [0.0, 1e-6, 1e-5, 5e-5]
SEEDS_CV       = [42, 7, 13]
MAX_ITER_CV    = 25
PATIENCE_CV    = 20

print(f"\n  Fase 0: grid search lambda_reg {LAMBDA_GRID}")
print(f"          ({len(SEEDS_CV)} semillas x {MAX_ITER_CV} iter, val set 15%)\n")
print(f"  {'lambda':>10} {'seed':>6} {'val_MSE':>14}")
print("  " + "-" * 34)

best_config = {'val': float('inf'), 'lam': 0.0, 'W': None, 'mu': MU_INIT}

for lam_cv in LAMBDA_GRID:
    for seed_cv in SEEDS_CV:
        torch.manual_seed(seed_cv)
        m_cv = FitNetLM2(n_input=n_feat_lm, n_h1=N_H1_LM, n_h2=N_H2_LM, n_h3=N_H3_LM).double()
        W_cv, vl_cv, _, _, _, _, _, mu_cv = run_lm_training(
            m_cv, Xtr_aug, Ytr_aug, Xvl_lm, Yvl_lm,
            n_params=n_params_lm, mu_init=MU_INIT,
            max_iter=MAX_ITER_CV, patience=PATIENCE_CV,
            tol_grad=TOL_GRAD_LM, nu_dec=NU_DEC, nu_inc=NU_INC,
            mu_min=MU_MIN, mu_max=MU_MAX, lambda_reg=lam_cv, verbose=False
        )
        print(f"  {lam_cv:>10.1e} {seed_cv:>6}  {vl_cv:>14.6e}")
        if vl_cv < best_config['val']:
            best_config = {'val': vl_cv, 'lam': lam_cv,
                           'W': W_cv.clone(), 'mu': mu_cv}

LAMBDA_REG = best_config['lam']
print(f"\n  Mejor config: lambda_reg={LAMBDA_REG:.1e}  val_MSE={best_config['val']:.6e}")

# -- Fase 1: 5 reinicios cortos con lambda* para explorar cuencas ------------
SEEDS_LM       = [42, 7, 13, 99, 2024]
MAX_ITER_SHORT = 35
PATIENCE_SHORT = 25

print(f"\n  Fase 1: {len(SEEDS_LM)} reinicios cortos ({MAX_ITER_SHORT} iter, lambda={LAMBDA_REG:.1e})")

ensemble_models  = []   # almacena los N modelos del ensemble
ensemble_val_mse = []

for i_rs, seed_rs in enumerate(SEEDS_LM):
    torch.manual_seed(seed_rs)
    m_tmp = FitNetLM2(n_input=n_feat_lm, n_h1=N_H1_LM, n_h2=N_H2_LM, n_h3=N_H3_LM).double()
    W_rs, vl_rs, _, _, nit_rs, _, _, mu_rs = run_lm_training(
        m_tmp, Xtr_aug, Ytr_aug, Xvl_lm, Yvl_lm,
        n_params=n_params_lm, mu_init=MU_INIT,
        max_iter=MAX_ITER_SHORT, patience=PATIENCE_SHORT,
        tol_grad=TOL_GRAD_LM, nu_dec=NU_DEC, nu_inc=NU_INC,
        mu_min=MU_MIN, mu_max=MU_MAX, lambda_reg=LAMBDA_REG, verbose=False
    )
    print(f"    Restart {i_rs+1} (seed={seed_rs:4d}): val_MSE={vl_rs:.6e}  iters={nit_rs}"
          f"  mu_final={mu_rs:.2e}")
    # Guardar modelo pre-entrenado para ensemble
    set_flat_params(m_tmp, W_rs)
    ensemble_models.append(m_tmp)
    ensemble_val_mse.append(vl_rs)

# Ordenar por val_MSE para logging
best_val_init = min(ensemble_val_mse)
best_mu_init  = MU_INIT  # se resetea para Phase 2 (continuacion desde punto bueno)
print(f"\n  Mejor val_MSE (Phase 1): {best_val_init:.6e}")

# -- Fase 2: continuar entrenando TODOS los modelos ---------------------------
#  Ensemble: promedio de N modelos reduce varianza y mejora R2 sistematicamente.
#  Cada modelo parte de una cuenca diferente → diversidad de predicciones.
print(f"\n  Fase 2: entrenamiento completo de {len(SEEDS_LM)} modelos ({MAX_ITER_LM} iter c/u)")

loss_tr_lm_hist = []
loss_vl_lm_hist = []
epoch_conv_lm   = 0
n_aceptados     = 0
n_rechazados    = 0

for i_ens, m_ens in enumerate(ensemble_models):
    print(f"\n  --- Modelo {i_ens+1}/{len(ensemble_models)} (seed={SEEDS_LM[i_ens]}) ---")
    print(f"  {'Iter':<6} {'MSE Train':>14} {'MSE Val':>14} {'mu':>12}")
    print("  " + "-" * 55)

    W_ens, vl_ens, tr_h, vl_h, ep_ens, acc_ens, rej_ens, _ = run_lm_training(
        m_ens, Xtr_aug, Ytr_aug, Xvl_lm, Yvl_lm,
        n_params=n_params_lm, mu_init=MU_INIT,
        max_iter=MAX_ITER_LM, patience=PATIENCE_LM,
        tol_grad=TOL_GRAD_LM, nu_dec=NU_DEC, nu_inc=NU_INC,
        mu_min=MU_MIN, mu_max=MU_MAX, lambda_reg=LAMBDA_REG, verbose=True
    )
    set_flat_params(m_ens, W_ens)
    n_aceptados  += acc_ens
    n_rechazados += rej_ens
    epoch_conv_lm = max(epoch_conv_lm, ep_ens)
    # Usar curvas del mejor modelo (menor val_MSE final)
    if vl_ens <= min(ensemble_val_mse):
        loss_tr_lm_hist = tr_h
        loss_vl_lm_hist = vl_h
    ensemble_val_mse[i_ens] = vl_ens

print(f"\n  Ensemble entrenado. val_MSE individuales: "
      f"{[f'{v:.4e}' for v in ensemble_val_mse]}")
print(f"  Aceptados totales: {n_aceptados}  |  Rechazados: {n_rechazados}")

# model_lm apunta al mejor modelo individual para compatibilidad con el resto del codigo
best_idx = int(np.argmin(ensemble_val_mse))
model_lm = ensemble_models[best_idx]

# =============================================================================
#  C.4 -- EVALUACION EN TEST SET (ENSEMBLE)
# =============================================================================

with torch.no_grad():
    # Ensemble ponderado por 1/val_MSE: mejor modelo aporta mas
    preds_test  = [m(Xte_lm).numpy().flatten() for m in ensemble_models]
    w_ens_raw   = np.array([1.0 / v for v in ensemble_val_mse])
    w_ens       = w_ens_raw / w_ens_raw.sum()
Ypred_lm_norm = sum(w * p for w, p in zip(w_ens, preds_test))
print(f"  Pesos ensemble (1/MSE norm): {[f'{w:.3f}' for w in w_ens]}")

Ypred_lm_usd = desnorm_lm(Ypred_lm_norm)
Yreal_lm_usd = desnorm_lm(Yte_lm.numpy().flatten())

# Tambien calcular R2 del mejor modelo individual (para referencia)
with torch.no_grad():
    pred_best_norm = model_lm(Xte_lm).numpy().flatten()
pred_best_usd = desnorm_lm(pred_best_norm)
r2_best_single = 1 - np.sum((Yreal_lm_usd - pred_best_usd)**2) / \
                     np.sum((Yreal_lm_usd - Yreal_lm_usd.mean())**2)

rmse_nn = np.sqrt(np.mean((Yreal_lm_usd - Ypred_lm_usd) ** 2))
mae_nn  = np.mean(np.abs(Yreal_lm_usd - Ypred_lm_usd))
r2_nn   = 1 - np.sum((Yreal_lm_usd - Ypred_lm_usd) ** 2) / \
              np.sum((Yreal_lm_usd - Yreal_lm_usd.mean()) ** 2)

print(f"\n  Iteraciones max. por modelo: {epoch_conv_lm}")
print(f"  RMSE Test (ensemble {len(ensemble_models)} modelos): {fmt_usd(rmse_nn)}")
print(f"  MAE  Test (ensemble): {fmt_usd(mae_nn)}")
print(f"  R2   Test (ensemble): {r2_nn:.4f}")
print(f"  R2   Test (mejor individuo): {r2_best_single:.4f}")

# =============================================================================
#  C.5 -- TABLA COMPARATIVA
# =============================================================================

print("\n" + "=" * 65)
print("  TABLA COMPARATIVA AMPLIADA -- N-R / SGD / fitnet-LM")
print("=" * 65)
print(f"\n  {'Criterio':<28} {'N-R lineal':>14} {'SGD lineal':>14} {'fitnet-LM':>12}")
print("  " + "-" * 72)
rows_c = [
    ("Tipo de modelo",      "Lineal",            "Lineal",           "No lineal"),
    ("Arquitectura",        "10 feat+bias",       "10 feat+bias",    f"{n_feat_lm}-[{N_H1_LM},{N_H2_LM},{N_H3_LM}]-1 lrelu"),
    ("Optimizador",         "N-R coordinado",    f"GD lr={LR_SGD}",  "Levenberg-Marquardt"),
    ("Jacobiana",           "Diag H (P,)",        "Gradiente (P,)",  f"J: {N_TR_LM}x{n_params_lm}"),
    ("Parametros P",        str(p),               str(p),             str(n_params_lm)),
    ("Iteraciones",         str(n_conv_mv),       str(sgd_conv),      str(epoch_conv_lm)),
    ("RMSE Test (USD)",     fmt_usd(rmse),        fmt_usd(rmse_sgd),  fmt_usd(rmse_nn)),
    ("R2 Test",             f"{r2:.4f}",           "---",              f"{r2_nn:.4f}"),
    ("Convergencia",        "Cuadratica",         "Lineal",           "Cuasi-cuadratica"),
]
for row in rows_c:
    print(f"  {row[0]:<28} {row[1]:>14} {row[2]:>14} {row[3]:>12}")

# =============================================================================
#  C.6 -- FIGURAS
# =============================================================================

# --- Figura 6: Curvas de convergencia LM -------------------------------------
fig, ax = plt.subplots(figsize=(9, 4))
epocas_lm = range(1, len(loss_tr_lm_hist) + 1)
ax.semilogy(epocas_lm, loss_tr_lm_hist, color='#2E86AB', lw=2, label='MSE Train')
ax.semilogy(epocas_lm, loss_vl_lm_hist, color='#E84855', lw=2,
            label='MSE Validacion', linestyle='--')
ax.set_xlabel("Iteracion LM", fontsize=11)
ax.set_ylabel("MSE (espacio normalizado)", fontsize=11)
ax.set_title(
    f"Figura 6: Convergencia LM -- fitnet({N_H1_LM},{N_H2_LM}) con Jacobiana\n"
    f"Early stop iter {epoch_conv_lm}  |  RMSE Test = {fmt_usd(rmse_nn)}",
    fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.text(0.60, 0.75,
        f"Iters: {epoch_conv_lm}\nRMSE: {fmt_usd(rmse_nn)}\nR2: {r2_nn:.4f}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray'))
plt.tight_layout()
plt.savefig("figura6_fitnet_learning_curves.png", bbox_inches='tight', dpi=130)
plt.show()

# --- Figura 7: Pred vs Real -- fitnet-LM -------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(Yreal_lm_usd / 1e3, Ypred_lm_usd / 1e3,
           alpha=0.3, s=8, color='#9B5DE5', label=f"fitnet({N_H1_LM},{N_H2_LM})-LM")
lim_nn = [0, max(Yreal_lm_usd.max(), Ypred_lm_usd.max()) / 1e3 * 1.05]
ax.plot(lim_nn, lim_nn, 'r--', lw=2, label="y = x (ideal)")
ax.set_xlabel("Precio real (miles USD)", fontsize=11)
ax.set_ylabel("Prediccion fitnet-LM (miles USD)", fontsize=11)
ax.set_title(
    f"Figura 7: Test Set KC -- fitnet({N_H1_LM},{N_H2_LM})-LM\n"
    f"R2 = {r2_nn:.4f}  |  RMSE = {fmt_usd(rmse_nn)}",
    fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figura7_fitnet_pred_vs_real.png", bbox_inches='tight', dpi=130)
plt.show()

# --- Figura 8: Comparativa RMSE N-R / SGD / fitnet-LM -----------------------
fig, ax = plt.subplots(figsize=(7, 4))
modelos_comp = ['N-R\n(lineal)', 'SGD\n(lineal)', f'fitnet({N_H1_LM},{N_H2_LM})\n(LM)']
rmses_comp   = [rmse / 1e3, rmse_sgd / 1e3, rmse_nn / 1e3]
colores_comp = ['#2E86AB', '#E84855', '#9B5DE5']
barras = ax.bar(modelos_comp, rmses_comp, color=colores_comp, width=0.45, edgecolor='white')
for bar, val in zip(barras, rmses_comp):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"${val:,.1f}k", ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel("RMSE Test (miles USD)", fontsize=11)
ax.set_title(
    f"Figura 8: Comparativa RMSE -- N-R lineal vs fitnet({N_H1_LM},{N_H2_LM})-LM\n"
    "KC House Data -- Test set 15%",
    fontsize=11, fontweight='bold')
ax.set_ylim(0, max(rmses_comp) * 1.20)
plt.tight_layout()
plt.savefig("figura8_comparativa_rmse.png", bbox_inches='tight', dpi=130)
plt.show()

print("\n" + "=" * 65)
print(f"  RESUMEN SECCION C -- fitnet({N_H1_LM},{N_H2_LM})-LM (Levenberg-Marquardt)")
print("=" * 65)
print(f"\n  Arquitectura:   {n_feat_lm} -> [{N_H1_LM},{N_H2_LM},{N_H3_LM}, lrelu] -> 1 (lineal)")
print(f"  Optimizador:    Levenberg-Marquardt (Jacobiana analitica, vectorizada)")
print(f"  Parametros:     P = {n_params_lm}")
print(f"  Dim. Jacobiana: N x P = {N_TR_LM} x {n_params_lm}  (~{N_TR_LM*n_params_lm/1e6:.1f}M elem)")
print(f"  Early stopping: patience={PATIENCE_LM}")
print(f"  Iteraciones:    {epoch_conv_lm}")
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
print(f"  -- SECCION C: fitnet({N_H1_LM},{N_H2_LM})-LM (Levenberg-Marquardt) --------")
print(f"    Arquitectura: {n_feat_lm} -> [{N_H1_LM},{N_H2_LM},{N_H3_LM}, lrelu] -> 1")
print(f"    Parametros P: {n_params_lm}")
print(f"    Jacobiana J:  {N_TR_LM} x {n_params_lm} (full-batch)")
print(f"    Iteraciones:  {epoch_conv_lm}")
print(f"    RMSE Test:    {fmt_usd(rmse_nn)}")
print(f"    MAE  Test:    {fmt_usd(mae_nn)}")
print(f"    R2   Test:    {r2_nn:.4f}")
print()
print("  Referencia: Camelino, Funes, Ziletti (2026)")
print("  UCC . Metodos Numericos . Grupo 5")
print("=" * 65)