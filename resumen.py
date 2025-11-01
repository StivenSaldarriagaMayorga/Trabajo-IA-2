import pandas as pd
from knn import metricas_knn
from desicion_tree import metricas_dt
from svm import metricas_svm
from red_neuronal import metricas_nn

datos_figura1 = {
    "ÁRBOLES DE DECISIÓN": metricas_dt,
    "K VECINOS MÁS CERCANOS (KNN)": metricas_knn,
    "MÁQUINAS DE VECTORES DE SOPORTE (SVM)": metricas_svm,
    "REDES NEURONALES": metricas_nn,
}

dfs = [pd.DataFrame(v) for v in datos_figura1.values()]

for k, df in zip(datos_figura1.keys(), dfs):
    df.columns = pd.MultiIndex.from_product([[k], df.columns])

figura1 = pd.concat(dfs, axis=1)
figura1 = figura1.round(2)
figura1.index = range(1, 9)
figura1.to_csv("figura1.csv")

print(figura1)
