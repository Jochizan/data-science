import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

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

    return(estadistico)


def bootstraping_empirico(x, fun_estadistico, n_iteraciones=9999):
    '''
    Función para calcular la diferencia entre el valor del estadístico en la
    muestra original y el valor del estadístico en múltiples muestras generadas
    por muestreo con reposición.

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
        valor de la diferencia del estadístico en cada muestra de bootstrapping.
    '''

    n = len(x)
    estadistico_muestra = fun_estadistico(x)
    dist_boot = np.full(shape=n_iteraciones, fill_value=np.nan)

    for i in tqdm(range(n_iteraciones)):
        resample = np.random.choice(x, size=n, replace=True)
        dist_boot[i] = fun_estadistico(resample) - estadistico_muestra

    return dist_boot


datos = np.array([
    81.372918, 25.700971, 4.942646, 43.020853, 81.690589, 51.195236,
    55.659909, 15.153155, 38.745780, 12.610385, 22.415094, 18.355721,
    38.081501, 48.171135, 18.462725, 44.642251, 25.391082, 20.410874,
    15.778187, 19.351485, 20.189991, 27.795406, 25.268600, 20.177459,
    15.196887, 26.206537, 19.190966, 35.481161, 28.094252, 30.305922
])

dist_boot = bootstraping_empirico(
                x = datos,
                fun_estadistico = calcular_estadistico,
                n_iteraciones   = 9999
            )

# Distribución de bootstrapping
# ==============================================================================
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7,3.3))
ax.hist(dist_boot, bins=30, density=True, color='#3182bd', alpha=0.5)
ax.set_title('Distribución de bootstrapping')
ax.set_xlabel(r'$\hat{X}^{b} - \hat{X}$')
ax.set_ylabel('densidad')



# Intervalo IC basado en bootstrapping empírico.
# ==============================================================================
# Un IC del 95% debe abarcar desde el cuantil 0.025 al 0.975
cuantiles = np.quantile(a = dist_boot, q = [0.025, 0.975])
estadistico_muestra = calcular_estadistico(datos)
print('------------------')
print('Intervalo empírico')
print('------------------')
intervalo = [estadistico_muestra - cuantiles[1] , estadistico_muestra - cuantiles[0]]
print(intervalo)

from statsmodels.stats.weightstats import DescrStatsW
d1 = DescrStatsW(datos)
print(d1.tconfint_mean(alpha=0.05, alternative='two-sided'))
