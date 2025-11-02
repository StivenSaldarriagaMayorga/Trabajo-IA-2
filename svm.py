from datetime import datetime
from dataset import calcular_metricas, dataframes, generar_caso_de_prueba, le, preprocesadores
import numpy as np
import pandas as pd
from scipy.sparse import vstack
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.decomposition import PCA


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

    # plt.savefig(f"images/{idx + 1}-{titulo}.png")
    plt.show()


def plot_roc_pr(modelo, X_test, y_test):
    # Predecir probabilidades
    y_scores = modelo.predict_proba(X_test)[:, 1]

    # Calcular curvas
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    fpr, tpr, _ = roc_curve(y_test, y_scores)

    # Calcular áreas bajo la curva
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y_test, y_scores)

    print(f"roc_auc:  {roc_auc:.2f}")
    print(f"pr_auc:  {pr_auc:.2f}")

    plt.figure(figsize=(12, 5))

    # Curva Precisión vs Recall
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, label=f"AP = {pr_auc:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision vs Recall")
    plt.grid(True)
    plt.legend()

    # Curva ROC
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")  # Línea diagonal
    plt.xlabel("FPR (1 - Specificity)")
    plt.ylabel("TPR (Recall)")
    plt.title("Curva ROC")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def entrenar_y_evaluar(idx, titulo, classifier, kernel, *, C, **kwargs):
    X_train, X_test, y_train, y_test = dataframes[idx]
    modelo = classifier(SVC(kernel=kernel, C=C, probability=idx == 7, **kwargs))
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    metricas = calcular_metricas(y_test, y_pred)

    # casos de prueba
    casos = []
    for i in range(3):
        c = generar_caso_de_prueba()
        cn = preprocesadores[idx].transform(c)
        prediccion = modelo.predict(cn)
        prediccion = le.inverse_transform(prediccion)
        c["Predicción"] = prediccion
        casos.append(c)
        print(f"> Caso de prueba {i+1}:", c)
        print(">> Predicción:", prediccion)
    casos = pd.concat(casos)
    casos.index = pd.Series(range(1, 4), name="Prueba #")
    casos = casos.reset_index()
    casos = casos.round(2)
    print(casos.T)
    # casos.T.to_csv(f"casos-svm/caso-{idx+1}.csv", index=False)

    # gráfico región de decisión
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    plot_decision_boundary(idx, titulo, X, y, modelo)

    # gráfico curvas roc y pr para el caso 8
    if idx == 7:  # caso 8: curvas ROC y PR
        plot_roc_pr(modelo, X_test, y_test)

    return metricas


metricas_svm = []
for idx, datos in enumerate(dataframes):
    print(f"====== Caso {idx + 1} ======")
    rbf_ovr = entrenar_y_evaluar(
        idx, "RBF OvR", OneVsRestClassifier, "rbf", C=1.0, gamma="auto"
    )
    print("RBF OvR:", rbf_ovr)
    # rbf_ovo = entrenar_y_evaluar(
    #     idx, "RBF OvO", OneVsOneClassifier, "rbf", C=1.0, gamma="auto"
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
    metricas_svm.append(rbf_ovr)
