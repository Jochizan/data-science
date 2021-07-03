# Tratamiento de datos
import pandas as pd
import numpy as np
from scipy.stats import trim_mean

# Gráficos
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style

# Configuración de matplotlib
style.use('ggplot') or plt.style.use('ggplot')
# config InlineBackend.figure_format = 'retina'

# Configuración warnings
import warnings

warnings.filterwarnings('ignore')

# Varios
from tqdm import tqdm

# Datos
# ==============================================================================
datos = np.array([
    81.372918, 25.700971, 4.942646, 43.020853, 81.690589, 51.195236,
    55.659909, 15.153155, 38.745780, 12.610385, 22.415094, 18.355721,
    38.081501, 48.171135, 18.462725, 44.642251, 25.391082, 20.410874,
    15.778187, 19.351485, 20.189991, 27.795406, 25.268600, 20.177459,
    15.196887, 26.206537, 19.190966, 35.481161, 28.094252, 30.305922
])

# Gráficos distribución observada
# ==============================================================================
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

axs[0].hist(datos, bins=30, density=True, color='#3182bd', alpha=0.5, label='muestra_1')
axs[0].plot(datos, np.full_like(datos, -0.001), '|k', markeredgewidth=1)
axs[0].set_title('Distribución de valores observados')
axs[0].set_xlabel('valor')
axs[0].set_ylabel('densidad')

pd.Series(datos).plot.kde(ax=axs[1], color='#3182bd')
axs[1].plot(datos, np.full_like(datos, 0), '|k', markeredgewidth=1)
axs[1].set_title('Distribución densidad estimada')
axs[1].set_xlabel('valor')
axs[1].set_ylabel('densidad')

fig.tight_layout()

plt.show()


def calcular_estadistico(x):
    '''
    Función para calcular el estadístico de interés.

    Parameters
    ----------
    x : numpy array
         valores de la muestra.

    Returns
    -------
    estadístico: float
        valor del estadístico.
    '''
    estadistico = np.mean(x)

    return estadistico


def bootstraping(x, fun_estadistico, n_iteraciones=9999):
    '''
    Función para calcular el valor del estadístico en múltiples muestras generadas
    mediante muestreo repetido con reposición (bootstrapping).

    Parameters
    ----------
    x : numpy array
         valores de la muestra.

    fun_estadistico : function
        función que recibe como argumento una muestra y devuelve el valor
        del estadístico.

    n_iteraciones : int
        número iteraciones (default `9999`).

    Returns
    -------
    distribuciones: numpy array
        valor del estadístico en cada muestra de bootstrapping.
    '''

    n = len(x)
    dist_boot = np.full(shape=n_iteraciones, fill_value=np.nan)

    for i in tqdm(range(n_iteraciones)):
        resample = np.random.choice(x, size=n, replace=True)
        dist_boot[i] = fun_estadistico(resample)

    return dist_boot


dist_boot = bootstraping(
    x=datos,
    fun_estadistico=calcular_estadistico,
    n_iteraciones=9999
)

# Distribución de bootstrapping
# ==============================================================================
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 3.3))
ax.hist(dist_boot, bins=30, density=True, color='#3182bd', alpha=0.5)
ax.set_title('Distribución de bootstrapping')
ax.set_xlabel('media')
ax.set_ylabel('densidad')

# Intervalo IC basado en percentiles de la distribución bootstrapping
# ==============================================================================
# Un IC del 95% debe abarcar desde el cuantil 0.025 al 0.975
cuantiles = np.quantile(a=dist_boot, q=[0.025, 0.975])
print('-------------------------------')
print('Intervalo basado en percentiles')
print('-------------------------------')
print(cuantiles)

# Gráfico intervalo de confianza del 95%
# ==============================================================================
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 4))
ax.hist(dist_boot, bins=30, density=True, color='#3182bd', alpha=0.5)
ax.axvline(x=datos.mean(), color='firebrick', label='media observada')
ax.axvline(x=cuantiles[0], color='black', linestyle='--', label='IC 95%')
ax.axvline(x=cuantiles[1], color='black', linestyle='--')
ax.hlines(y=0.001, xmin=cuantiles[0], xmax=cuantiles[1], color='black')
ax.set_title('Intervalo bootstrapping basados en percentiles')
ax.set_xlabel('media')
ax.set_ylabel('densidad')
ax.legend()

plt.show()
