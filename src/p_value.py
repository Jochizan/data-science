import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración warnings
import warnings

warnings.filterwarnings('ignore')

# Varios
from tqdm import tqdm

# Datos
# ==============================================================================
url = 'https://raw.githubusercontent.com/JoaquinAmatRodrigo/Estadistica-con-R/' \
      + 'master/datos/LatinoEd.csv'
datos = pd.read_csv(url)
datos = datos[['Achieve', 'Mex']]
datos = datos.rename(columns={'Achieve': 'nota', 'Mex': 'nacionalidad'})
datos['nacionalidad'] = np.where(datos['nacionalidad'] == 0, 'pais_1', 'pais_2')
datos.info()

# Estadísticos descriptivos por grupo
# ==============================================================================
datos.groupby(by='nacionalidad').describe()

dif_observada = (datos.nota[datos.nacionalidad == 'pais_1'].mean()
                 - datos.nota[datos.nacionalidad == 'pais_2'].mean())

print(f"Diferencia de medias observada: {dif_observada}")

# Gráficos distribución observada
# ==============================================================================
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 3.5))
sns.violinplot(
    x=datos.nota,
    y=datos.nacionalidad,
    color=".8",
    ax=axs[0]
)
sns.stripplot(
    x=datos.nota,
    y=datos.nacionalidad,
    data=datos,
    size=4,
    jitter=0.1,
    palette='tab10',
    ax=axs[0]
)
axs[0].set_title('Distribución de valores por grupo')
axs[0].set_ylabel('nacionalidad')
axs[0].set_xlabel('nota')

for nacionalidad in datos.nacionalidad.unique():
    datos_temp = datos[datos.nacionalidad == nacionalidad]['nota']
    datos_temp.plot.kde(ax=axs[1], label=nacionalidad)
    axs[1].plot(datos_temp, np.full_like(datos_temp, 0), '|k', markeredgewidth=1)

axs[1].set_title('Distribución de valores por grupo')
axs[1].set_xlabel('nota')
axs[1].legend()

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

    return (estadistico)


def bootstraping_2_grupos(x1, x2, fun_estadistico, n_iteraciones=9999):
    '''
    Función para calcular la diferencia de un estadístico entre dos grupos en
    múltiples muestras generadas mediante muestreo repetido con reposición
    (bootstrapping).

    Parameters
    ----------
    x1 : numpy array
         valores de la muestra del grupo 1.

    x2 : numpy array
         valores de la muestra del grupo 2.

    fun_estadistico : function
        función que recibe como argumento una muestra y devuelve el valor
        del estadístico.

    n_iteraciones : int
        número iteraciones (default `9999`).

    Returns
    -------
    distribuciones: numpy array
        diferencia entre ambos grupos en cada muestra de bootstrapping.
    '''

    n1 = len(x1)
    n2 = len(x2)
    pool = np.hstack((x1, x2))
    dist_boot = np.full(shape=n_iteraciones, fill_value=np.nan)

    for i in tqdm(range(n_iteraciones)):
        # Se crea una nueva muestra
        resample = np.random.choice(pool, size=n1 + n2, replace=True)
        # Se reparten las observaciones en dos grupos y se calcula el estadístico
        estadistico_1 = fun_estadistico(resample[:n1])
        estadistico_2 = fun_estadistico(resample[n1:])
        # Diferencia entre estadísticos
        dist_boot[i] = estadistico_1 - estadistico_2

    return dist_boot


dist_boot = bootstraping_2_grupos(
    x1=datos.nota[datos.nacionalidad == 'pais_1'],
    x2=datos.nota[datos.nacionalidad == 'pais_2'],
    fun_estadistico=calcular_estadistico,
    n_iteraciones=9999
)

# Distribución de bootstrapping
# ==============================================================================
dif_observada = datos.nota[datos.nacionalidad == 'pais_1'].mean() \
                - datos.nota[datos.nacionalidad == 'pais_2'].mean()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 4))
ax.hist(dist_boot, bins=30, density=True, color='#3182bd', alpha=0.5)
ax.axvline(x=dif_observada, color='red', label='diferencia observada')
ax.axvline(x=-dif_observada, color='red')

ax.set_title('Distribución de bootstrapping')
ax.set_xlabel('diferencia de medias')
ax.set_ylabel('densidad')
ax.legend()

pd.Series(dist_boot).describe()

# P-value empírico con y sin corrección
# ==============================================================================
p_value = (sum(np.abs(dist_boot) > np.abs(dif_observada))) / len(dist_boot)
p_value_correc = (sum(np.abs(dist_boot) > np.abs(dif_observada)) + 1) / len(dist_boot + 1)
print(f"p-value sin corrección: {p_value}")
print(f"p-value con corrección: {p_value_correc}")
