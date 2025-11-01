import pandas as pd
from knn import metricas_knn
from desicion_tree import metricas_dt
from svm import metricas_svm
from red_neuronal import metricas_nn


def generar_figura1():
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
    # figura1.to_csv("figura1.csv")

    return figura1


def plot_f1_max_por_algoritmo(f1):
    return f1.max().plot.bar(rot=0, title="F1-score máximo por algoritmo (general)", ylabel="F1-score")


def plot_f1_medio_por_normalizacion_y_algoritmo(f1):
    ed_no = f1.iloc[:4, :].mean()
    ed_si = f1.iloc[:4, :].mean()
    pd.DataFrame({"ED (NO)": ed_no, "ED (SI)": ed_si}).plot.bar(rot=0, title="F1-score medio por normalización y algoritmo", ylabel="F1-score")


def plot_f1_medio_por_outliers_y_algoritmo(f1):
    outliers_no = f1.iloc[[0, 1, 4, 5], :].mean()
    outliers_si = f1.iloc[[2, 3, 6, 7], :].mean()
    pd.DataFrame({"Outliers (NO)": outliers_no, "Outliers (SI)": outliers_si}).plot.bar(rot=0, title="F1-score medio por outliers y algoritmo", ylabel="F1-score")


def plot_f1_medio_por_balanceo_y_algoritmo(f1):
    balanceo_no = f1.iloc[[0, 2, 4, 6], :].mean()
    balanceo_si = f1.iloc[[1, 3, 5, 7], :].mean()
    pd.DataFrame({"Outliers (NO)": balanceo_no, "Outliers (SI)": balanceo_si}).plot.bar(rot=0, title="F1-score medio por balanceo y algoritmo", ylabel="F1-score")


figura1 = generar_figura1()
f1 = figura1.xs("f1", axis=1, level=1)

plot_f1_max_por_algoritmo(f1)
plot_f1_medio_por_normalizacion_y_algoritmo(f1)
plot_f1_medio_por_outliers_y_algoritmo(f1)
plot_f1_medio_por_balanceo_y_algoritmo(f1)
