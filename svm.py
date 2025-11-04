from pathlib import Path

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight
from dataset import calcular_metricas, dataframes, generar_resumen_pruebas, le, preprocesadores, probar_modelo
import numpy as np
import pandas as pd
from scipy.sparse import vstack
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.decomposition import PCA


imgs_dir = Path(f"resultados/imagenes/svm")


# Función para plotear los límites de decisión
def plot_decision_boundary(idx, titulo, X, y, model):
    plt.figure()
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max),
                         np.linspace(y_min, y_max))

    # Cada punto de la malla está en 2D (PC1, PC2)
    # Necesitamos llevarlo al espacio original para usar el modelo
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_original = pca.inverse_transform(grid_points)

    Z = model.predict(grid_points_original)
    # Z = le.transform(Z)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, vmin=y.min(), vmax=y.max(), alpha=0.8, cmap=plt.cm.RdYlBu)

    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors="k")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"Regiones de decisión SVM: Caso {idx + 1} {titulo}")

    plt.legend(
        handles=scatter.legend_elements()[0], labels=list(le.classes_), title="Clases"
    )

    if imgs_dir.exists():
        plt.savefig(imgs_dir / f"{idx + 1}-{titulo}.png")
    plt.show()


def plot_roc_pr(modelo, X_test, y_test):
    n_classes = len(list(le.classes_))-1

    # Binarizar etiquetas (necesario para curvas ROC/PR multiclase)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    # Predicciones probabilísticas
    y_score = modelo.predict_proba(X_test)

    # Curva ROC (micro)
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC micro (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC Micro')
    plt.legend()
    if imgs_dir.exists():
        plt.savefig(imgs_dir / f"roc-caso8.png")
    plt.show()

    # Curva Precision-Recall (micro)
    precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_score.ravel())
    avg_precision = average_precision_score(y_test_bin, y_score, average="micro")

    plt.figure()
    plt.plot(recall, precision, color='green', lw=2,
             label=f'Precision-Recall micro (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall Micro')
    plt.legend()
    if imgs_dir.exists():
        plt.savefig("pr-caso8.png")
    plt.show()

def buscar_hiperparametros():
    for idx in range(len(dataframes)):
        print(f"====== Caso {idx + 1} ======")
        param_grid = {
          'estimator__C': [0.01, 0.1, 1, 10, 100],
          'estimator__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        }
        grid = GridSearchCV(OneVsRestClassifier(SVC(kernel="rbf")), param_grid, scoring='f1_macro', cv=5, n_jobs=-1)
        X_train, _, y_train, _ = dataframes[idx]
        grid.fit(X_train, y_train)
        print(grid.best_params_, grid.best_score_)


def entrenar_y_evaluar(idx, titulo, classifier, kernel, *, C, **kwargs):
    X_train, X_test, y_train, y_test = dataframes[idx]
    modelo = classifier(SVC(kernel=kernel, C=C, probability=idx == 7, **kwargs))
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    metricas = calcular_metricas(y_test, y_pred)

    # casos de prueba
    pruebas = probar_modelo(modelo, preprocesadores[idx])

    # gráfico región de decisión
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    plot_decision_boundary(idx, titulo, X, y, modelo)

    # gráfico curvas roc y pr para el caso 8
    if idx == 7:  # caso 8: curvas ROC y PR
        plot_roc_pr(modelo, X_test, y_test)

    return metricas, pruebas


# buscar_hiperparametros()
hiperparametros = [
    {'C': 10, 'gamma': 0.001},
    {'C': 100, 'gamma': 0.01},
    {'C': 1, 'gamma': 1},
    {'C': 100, 'gamma': 0.001},
    {'C': 10, 'gamma': 'scale'},
    {'C': 10, 'gamma': 1},
    {'C': 100, 'gamma': 1},
    {'C': 100, 'gamma': 1}
]

metricas_svm = []
pruebas_svm = []
for idx in range(len(dataframes)):
    print(f"====== Caso {idx + 1} ======")
    metricas, pruebas = entrenar_y_evaluar(
        idx, "RBF OvR", OneVsRestClassifier, "rbf",
        **hiperparametros[idx]
    )
    print("RBF OvR:", metricas)
    # rbf_ovo = entrenar_y_evaluar(
    #     idx, "RBF OvO", OneVsOneClassifier, "rbf",
    #     **hiperparametros[idx]
    # )
    # print("RBF OvO:", rbf_ovo)
    # lineal_ovr = entrenar_y_evaluar(
    #     idx, "Lineal OvR", OneVsRestClassifier, "linear", C=1.0
    # )
    # print("Lineal OvR:", lineal_ovr)
    # lineal_ovo = entrenar_y_evaluar(
    #     idx, "Lineal OvO", OneVsOneClassifier, "linear", C=1.0
    # )
    # print("Lineal OvO:", lineal_ovo)

    # mejor = max((rbf_ovr, rbf_ovo, lineal_ovr, lineal_ovo), key=lambda x: x["f1"])
    # metricas_svm.append(mejor)
    metricas_svm.append(metricas)
    pruebas_svm.append(pruebas)
