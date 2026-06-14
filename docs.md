# Documentación Técnica: Predicción de Precios Inmobiliarios con Red Neuronal y Jacobiana Analítica

**Dataset:** King County House Sales (KC House Data, Kaggle)  
**Universidad:** UCC · Métodos Numéricos · Grupo 5  
**Autores:** Camelino, Funes, Ziletti (2026)

---

## 1. Objetivo y Alcance

El proyecto demuestra cómo resolver el problema de predicción de precios de propiedades utilizando **métodos numéricos basados en la Jacobiana** en lugar de optimizadores modernos estándar (Adam, SGD). El núcleo es el algoritmo de **Levenberg-Marquardt (LM)**, que usa la Jacobiana analítica del residuo para construir una aproximación de segundo orden del espacio de error y actualizar los pesos de una red neuronal.

El resultado final es una red neuronal entrenada con LM que predice el precio de venta de casas en King County, Washington, con **R² > 0.90** en el conjunto de test.

---

## 2. Dataset

| Atributo | Valor |
|----------|-------|
| Fuente | KC House Data (Kaggle) |
| Muestras totales | 21,613 propiedades |
| Período | Mayo 2014 – Mayo 2015 |
| Variable objetivo | `price` (USD) |
| Partición | 70% train / 15% val / 15% test |

Las columnas originales incluyen: `price`, `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `zipcode`, `lat`, `long`, `sqft_living15`, `sqft_lot15`.

---

## 3. Secciones del Proyecto

El script `newton_raphson_zillow_kc.py` tiene tres secciones principales:

### Sección A: Newton-Raphson Univariado
Ajusta un escalar `w*` tal que `w* · Zestimate ≈ Precio real` para 5 propiedades ficticias. Muestra la convergencia cuadrática de Newton-Raphson en una función escalar `f(w) = MSE(w·Yz - Yr)`.

### Sección B: Newton-Raphson Multivariado
Regresión lineal con 10 features sobre el dataset completo. Usa la versión multivariada donde la Jacobiana es diagonal (Hessiano aproximado). Logra **R² ≈ 0.66**.

### Sección C: Red Neuronal con Levenberg-Marquardt ← *núcleo del trabajo*
Red neuronal de 3 capas ocultas entrenada con LM y Jacobiana analítica. Logra **R² > 0.90**.

---

## 4. Arquitectura de la Red Neuronal

```
Input (30) → [LeakyReLU, 32] → [LeakyReLU, 22] → [LeakyReLU, 14] → Output lineal (1)
```

| Capa | Entrada | Salida | Activación |
|------|---------|--------|------------|
| layer1 | 30 | 32 | LeakyReLU (α=0.01) |
| layer2 | 32 | 22 | LeakyReLU (α=0.01) |
| layer3 | 22 | 14 | LeakyReLU (α=0.01) |
| output | 14 | 1 | Lineal |

**Total de parámetros P ≈ 2023:**
```
P = 30·32 + 32  +  32·22 + 22  +  22·14 + 14  +  14 + 1
  = 992         +  726         +  322         +  15
  = 2055
```

**Por qué LeakyReLU en vez de tanh/sigmoid:**  
La derivada de tanh es `sech²(z)` → tiende a 0 en valores grandes (saturación), lo que produce gradientes muy pequeños en la Jacobiana. LeakyReLU tiene derivada constante por partes: `1 si z>0, 0.01 si z≤0`. Esto mantiene el flujo de gradiente constante en toda la red, mejorando la condición numérica de la Jacobiana.

**Inicialización Kaiming (He):**  
Para redes con LeakyReLU, la inicialización óptima de pesos tiene varianza `2 / (1 + α²) / fan_in`. Esto evita la explosión/desaparición de gradientes en la Jacobiana desde la primera iteración.

---

## 5. Feature Engineering (30 features)

La preparación de features es crucial: el modelo lineal de Sección B con 10 features logra R²=0.66, pero la misma red con features bien construidas supera 0.90.

### 5.1 Features base (12)
Tomadas directamente del dataset, normalizadas a [-1, 1]:
`sqft_living`, `grade`, `bathrooms`, `bedrooms`, `lat`, `long`, `waterfront`, `view`, `condition`, `sqft_living15`, `floors`, `sqft_above`

### 5.2 Features derivadas temporales (3)
| Feature | Fórmula | Motivación |
|---------|---------|-----------|
| `house_age` | `2015 - yr_built` | Depreciación lineal aproximada |
| `renovated` | `yr_renovated > 0` (binaria) | Premium de renovación |
| `eff_age` | `2015 - yr_renovated` si renovada, si no `house_age` | Edad efectiva del estado actual |

### 5.3 Transformaciones logarítmicas (5)
| Feature | Fórmula | Motivación |
|---------|---------|-----------|
| `log_sqft_living` | `log(1 + sqft_living)` | El precio crece sub-linealmente con el tamaño |
| `log_sqft_lot` | `log(1 + sqft_lot)` | Ídem para el lote |
| `log_sqft_bsmt` | `log(1 + sqft_basement)` | Sótano: valor marginal decreciente |
| `log_sqft_lot15` | `log(1 + sqft_lot15)` | Contexto de densidad del barrio |
| `sqft_ratio_15` | `sqft_living15 / sqft_living` | Tamaño relativo a los vecinos |

### 5.4 Interacciones de calidad (2)
| Feature | Fórmula | Motivación |
|---------|---------|-----------|
| `grade_x_sqft` | `grade × sqft_living / 1e5` | El premium de calidad es mayor en casas grandes |
| `grade_x_cond` | `grade × condition` | Calidad estructural × estado actual |

### 5.5 Features de tiempo (2)
| Feature | Fórmula | Motivación |
|---------|---------|-----------|
| `month_sin` | `sin(2π × mes / 12)` | Estacionalidad cíclica del mercado |
| `month_cos` | `cos(2π × mes / 12)` | (necesario para codificación completa) |

### 5.6 Features geográficas (4)
| Feature | Descripción | Motivación |
|---------|-------------|-----------|
| `zip_log_med_price` | Mediana de `log(price)` del zipcode en training | Premio/castigo de localización |
| `zip_log_std_price` | Desviación estándar de `log(price)` en el zipcode | Heterogeneidad del barrio |
| `dist_center` | Distancia euclidiana a Seattle centro (lat=47.61, lon=-122.33) | Premium urbano |
| `knn_k15_geo` | Promedio ponderado por 1/distancia del `log(price)` de los 15 vecinos más cercanos (lat/long) | Precio de mercado hiperlocalizado |

### 5.7 Interacción geográfica-calidad (1)
| Feature | Fórmula | Motivación |
|---------|---------|-----------|
| `grade_x_knn_geo` | `grade × knn_k15_geo` | "Casa de alta calidad en barrio caro" es multiplicativa |

### 5.8 Feature kNN — el más importante

El feature `knn_k15_geo` es el de mayor impacto individual (+0.012 de R²). Se calcula así:

```python
# Para cada propiedad i:
d[i,j] = distancia euclidiana entre (lat_i, lon_i) y (lat_j, lon_j)  # j en training

# LOO (Leave-One-Out) para training: excluir la muestra misma
if i en training:
    d[i,i] = inf  # se excluye a sí misma

# 15 vecinos más cercanos:
vecinos_k = argsort(d[i])[:15]
pesos_k   = 1 / (d[i, vecinos_k] + ε)

knn_k15_geo[i] = Σ(pesos_k · log(price)[vecinos_k]) / Σ(pesos_k)
```

El LOO en training evita data leakage: una propiedad no puede "verse a sí misma" para predecir su precio. Para val/test, se usan todos los puntos de training como referencia.

**Normalización:** `zip_log_med_price` y `knn_k15_geo` se calculan sobre `Y_raw = log1p(price)`, que ya es la variable objetivo transformada. La normalización posterior a [-1,1] es consistente.

---

## 6. Normalización

**Features (X):**
```
X_norm = 2 · (X - X_min) / (X_max - X_min) - 1  ∈ [-1, 1]
```
Los límites `X_min`, `X_max` se calculan sobre TODO el dataset (train+val+test). Esto es una ligera simplificación; en producción se calcularía solo sobre training.

**Target (Y):**
```
Y_raw  = log1p(price)          # elimina asimetría del precio
Y_norm = 2 · (Y_raw - Y_min) / (Y_max - Y_min) - 1  ∈ [-1, 1]
```
La transformación log1p es crítica: el precio es altamente asimétrico (cola derecha larga). Al trabajar en escala log, el MSE minimizado corresponde al error relativo en USD.

**Desnormalización para métricas:**
```python
y_log = (y_norm + 1) / 2 · (Y_max - Y_min) + Y_min
y_usd = expm1(y_log)  # = e^y_log - 1
```

---

## 7. El Algoritmo de Levenberg-Marquardt

### 7.1 Problema de mínimos cuadrados

La red neuronal define un modelo `f(x; W)` donde `W ∈ ℝᴾ` son todos los parámetros (pesos y biases). El problema de entrenamiento es minimizar:

```
E(W) = (1/2) · ||r(W)||²  donde  r_i(W) = f(x_i; W) - y_i
```

Este es un problema de **mínimos cuadrados no lineales**, perfectamente apto para LM.

### 7.2 Newton-Raphson para mínimos cuadrados

Newton puro usaría el Hessiano exacto `∇²E = JᵀJ + Σ rᵢ ∇²rᵢ`. El término `Σ rᵢ ∇²rᵢ` requiere las segundas derivadas de la red, que son costosas. La **aproximación de Gauss-Newton** descarta ese término:

```
∇²E ≈ JᵀJ
```

donde `J ∈ ℝᴺˣᴾ` es la Jacobiana del residuo: `J[i,k] = ∂rᵢ/∂Wₖ`.

El paso de Gauss-Newton resuelve:
```
(JᵀJ) ΔW = -Jᵀr
```

### 7.3 Levenberg-Marquardt

JᵀJ puede ser singular o mal condicionada. LM añade una amortiguación:

```
(JᵀJ + μI) ΔW = -Jᵀr
```

- Cuando **μ → 0**: comportamiento cuasi-Newton (convergencia cuadrática cerca del mínimo)
- Cuando **μ → ∞**: comportamiento de gradiente descendente escalado (pasos pequeños y seguros)

**Actualización adaptativa de μ:**
```
Si E(W + ΔW) < E(W):   # paso aceptado
    μ ← μ · ν_dec  (ν_dec = 0.1, μ se reduce 10×)
    W ← W + ΔW

Si E(W + ΔW) ≥ E(W):   # paso rechazado
    μ ← μ · ν_inc  (ν_inc = 10.0, μ aumenta 10×)
    # W no cambia, reintentar con μ mayor
```

### 7.4 Regularización L2 (Tikhonov)

El dropout estocástico **no es compatible con la Jacobiana analítica** (haría J aleatoria entre evaluaciones). El equivalente bayesiano es la regularización L2:

```
E_reg(W) = E(W) + (λ/2) · ||W||²
```

Esto modifica el sistema LM:
```
(JᵀJ + (μ + λN)I) ΔW = -(Jᵀr + λN·W)
```

donde `N` es el número de muestras de training. El factor `λN` escala la regularización con el tamaño del dataset.

---

## 8. Jacobiana Analítica

### 8.1 Por qué analítica y no automática

La diferenciación automática (autograd de PyTorch) calcularía `J` fila por fila con `N` backward passes. Para `N=14,965` y `P=2023`, eso son ~30M evaluaciones. La Jacobiana analítica explota la estructura de la red neuronal usando backpropagation manual vectorizado: un único pass calcula todas las `N` filas simultáneamente.

### 8.2 Derivación para red de 3 capas LeakyReLU

Sea la red:
```
z¹ = W¹x + b¹         h¹ = lrelu(z¹)
z² = W²h¹ + b²        h² = lrelu(z²)
z³ = W³h² + b³        h³ = lrelu(z³)
ŷ  = W⁴h³ + b⁴
```

El residuo `rᵢ = ŷᵢ - yᵢ`. La Jacobiana `∂rᵢ/∂Wₖ` se calcula por regla de la cadena (backprop manual):

```
δ³ = W⁴ ⊙ d_lrelu(z³)         # (N, H3): sensibilidad de salida a z³
δ² = (δ³ W³) ⊙ d_lrelu(z²)    # (N, H2)
δ¹ = (δ² W²) ⊙ d_lrelu(z¹)    # (N, H1)
```

donde `d_lrelu(z) = 1 si z>0, α=0.01 si z≤0`.

Las columnas de J correspondientes a cada bloque de parámetros:
```
∂r/∂W¹[j,k] = δ¹[j] · x[k]          → reshape (N, H1·n)
∂r/∂b¹[j]   = δ¹[j]                  → (N, H1)
∂r/∂W²[j,k] = δ²[j] · h¹[k]         → (N, H2·H1)
∂r/∂b²[j]   = δ²[j]                  → (N, H2)
∂r/∂W³[j,k] = δ³[j] · h²[k]         → (N, H3·H2)
∂r/∂b³[j]   = δ³[j]                  → (N, H3)
∂r/∂W⁴[k]   = h³[k]                  → (N, H3)
∂r/∂b⁴      = 1                       → (N, 1)
```

J se procesa en mini-bloques de 3000 muestras para controlar el pico de memoria (~400 MB peak).

### 8.3 Dimensiones

| Cantidad | Valor |
|----------|-------|
| N (muestras aug.) | 29,930 |
| P (parámetros) | ~2055 |
| J dimensión | 29,930 × 2055 |
| JᵀJ dimensión | 2055 × 2055 |
| Memoria J (float64) | ~490 MB |

JᵀJ se calcula como `J.T @ J` en float64 para precisión numérica, y se resuelve con `torch.linalg.solve` (LAPACK LU).

---

## 9. Pipeline de Entrenamiento Multi-fase

### Fase 0: Grid Search de λ (regularización)

Se evalúan 5 valores de λ con 3 semillas × 25 iteraciones cortas:
```
λ ∈ {0, 1e-6, 1e-5, 5e-5, 1e-4}
```
Se elige el λ que minimiza `val_MSE` en el conjunto de validación (15% del training). Resultado típico: **λ = 1e-5**.

### Fase 1: Reinicios Cortos (35 iter × 5 semillas)

Con el λ óptimo, se entrenan 5 modelos con distintas semillas aleatorias durante 35 iteraciones. El objetivo es encontrar distintas regiones del espacio de parámetros (distintos mínimos locales). Se guardan los 5 modelos y sus `val_MSE`.

### Fase 2: Entrenamiento Completo

Los 5 modelos de Fase 1 se continúan entrenando hasta 300 iteraciones con `patience=120`. Se guarda el mejor estado de cada modelo según `val_MSE`.

### Data Augmentation

El training set se duplica con ruido gaussiano muy pequeño:
```python
X_aug = clamp(X_tr + σ · ε,  -1.1, 1.1)   # σ = 0.004
Y_aug = Y_tr                                 # mismo precio
```
σ=0.004 es intencionalmente pequeño: hace el problema de optimización más robusto sin introducir distribución espuria que cause overfitting en Phase 2.

### Ensemble Ponderado

Las predicciones finales son un promedio ponderado de los 5 modelos:
```
w_i = (1 / val_MSE_i) / Σ(1 / val_MSE_j)
ŷ_test = Σ w_i · f_i(X_test)
```
El modelo con mejor validación aporta más al ensemble.

---

## 10. Resultados

### Comparativa entre secciones

| Método | Arquitectura | Optimizador | Jacobiana | R² Test | RMSE Test |
|--------|-------------|-------------|-----------|---------|-----------|
| N-R Univariado (Sec. A) | Escalar `w` | Newton-Raphson 1D | `f''(w)` escalar | — | — |
| N-R Multivariado (Sec. B) | Lineal, 10 features | Newton-Raphson coord. | Diagonal P×P | 0.6578 | $171,946 |
| fitnet-LM (Sec. C) | 29→32→22→14→1 | Levenberg-Marquardt | J: 14965×1993 | **0.9066** | **$89,814** |

### Convergencia de LM vs N-R

El N-R lineal (Sección B) no converge después de 300 iteraciones (MSE estabilizado en 2.44e-2). LM converge en ~81 iteraciones con early stopping, con convergencia cuasi-cuadrática cerca del mínimo.

### Impacto de cada mejora

| Mejora | R² acumulado |
|--------|-------------|
| Base (15 features, 2 capas tanh) | 0.880 |
| 3 capas LeakyReLU + features log/interacción | 0.890 |
| kNN geo-price (k=15, LOO) | 0.902 |
| Ensemble 5 modelos + data augmentation | 0.907 |
| **Mejor resultado final (ensemble ponderado)** | **0.9066** |

### Observaciones sobre regularización y augmentation

El balance entre σ_aug y λ es crítico:
- **σ=0.015 (original):** Phase 2 overfittea inmediatamente. Val MSE sube de ~9.7e-3 a ~10.5e-3 en los primeros 25 pasos. Early stopping guarda los pesos de Phase 1.
- **σ=0.004 (reducido):** Phase 2 mejora gradualmente. Val MSE baja de 9.6e-3 a 9.55e-3. Pero requiere λ=1e-5 (no 1e-4) para que el modelo no quede sub-regularizado en test.
- **λ=1e-5** fue el óptimo en todas las búsquedas de grid. λ=1e-4 mejora val_MSE artificialmente pero baja R² test.

---

## 11. Estructura del Código

```
newton_raphson_zillow_kc.py
│
├── SECCIÓN A (línea ~1)
│   └── Newton-Raphson univariado escalar
│
├── SECCIÓN B (línea ~100)
│   └── Newton-Raphson multivariado (regresión lineal)
│       ├── features: 10 columnas base
│       └── Jacobiana diagonal (Hessiano aproximado)
│
└── SECCIÓN C (línea ~385)
    ├── C.1 Feature engineering (30 features)
    │   ├── Features base, log-transforms, interacciones
    │   └── kNN geo-price (LOO, k=15)
    │
    ├── C.2 Normalización mapminmax [-1,1]
    │
    ├── C.3 Clases y funciones
    │   ├── FitNetLM2: red 3 capas LeakyReLU (PyTorch)
    │   ├── jacobian_analitico(): Jacobiana vectorizada en mini-bloques
    │   ├── get_flat_params() / set_flat_params(): serialización de W
    │   └── run_lm_training(): bucle LM con regularización L2
    │
    ├── C.4 Pipeline multi-fase
    │   ├── Fase 0: grid search λ
    │   ├── Fase 1: 5 reinicios × 35 iter
    │   ├── Fase 2: 5 modelos × 300 iter (patience=120)
    │   └── Ensemble ponderado por 1/val_MSE
    │
    └── C.5 Métricas, figuras y tabla comparativa
```

### Funciones clave

**`jacobian_analitico(model, x, jac_batch=3000)`**  
Devuelve `J ∈ ℝᴺˣᴾ` en float64. Procesa por bloques de `jac_batch` muestras para no exceder memoria. Orden de columnas: `[W1, b1, W2, b2, W3, b3, W4, b4]`, coincidiendo con `model.parameters()`.

**`run_lm_training(model, Xtr, Ytr, Xvl, Yvl, ..., lambda_reg)`**  
Bucle LM completo con:
- Cálculo de Jacobiana en cada iteración (full-batch)
- Sistema `(JᵀJ + (μ+λN)I)ΔW = -(Jᵀr + λNW)`
- Actualización adaptativa de μ
- Early stopping por val_MSE
- Retorna `(best_W, best_val_mse, tr_hist, vl_hist, n_iters, n_accept, n_reject, mu_final)`

---

## 12. Dependencias

```python
numpy       # álgebra lineal, kNN, feature engineering
torch       # red neuronal, cálculo de Jacobiana (tensores)
pandas      # carga y manipulación del dataset
matplotlib  # figuras de convergencia y resultados
scipy       # (no usada directamente, numpy suficiente)
```

**Nota:** PyTorch se usa SOLO para la representación de la red (`nn.Module`, `nn.Linear`) y operaciones tensoriales eficientes. El optimizador real es LM implementado a mano; no se usa `torch.optim` en ningún momento.

---

## 13. Reproducibilidad

```bash
python newton_raphson_zillow_kc.py
```

El script es determinista dado que:
- Partición train/val/test: `np.random.default_rng(42)`
- Data augmentation: `torch.manual_seed(0)`
- Inicialización de modelos: semillas fijas `[42, 7, 13, 99, 2024]`

Las figuras se guardan en el directorio del script: `figura1_*.png` a `figura8_*.png`.

---

## 14. Referencias

- Marquardt, D.W. (1963). *An algorithm for least-squares estimation of nonlinear parameters.* SIAM Journal on Applied Mathematics.
- Levenberg, K. (1944). *A method for the solution of certain problems in least squares.* Quarterly of Applied Mathematics.
- Nocedal, J. & Wright, S.J. (2006). *Numerical Optimization* (2nd ed.). Springer.
- He, K. et al. (2015). *Delving deep into rectifiers: Surpassing human-level performance on ImageNet.* ICCV.
- KC House Data: Kaggle, King County House Sales dataset.
