from datetime import datetime
from prueba import dataframes
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
from sklearn.preprocessing import LabelEncoder


# Función para plotear los límites de decisión
def plot_decision_boundary(idx, titulo, X, y, model):
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

    le = LabelEncoder()
    y = le.fit_transform(y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max),
                         np.linspace(y_min, y_max))

    # Cada punto de la malla está en 2D (PC1, PC2)
    # Necesitamos llevarlo al espacio original para usar el modelo
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_original = pca.inverse_transform(grid_points)

    Z = model.predict(grid_points_original)
    Z = le.transform(Z)
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

    # Paso 8: Calcular áreas bajo la curva
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


def entrenar_y_evaluar(idx, titulo, datos, classifier, kernel, *, C, **kwargs):
    X_train, X_test, y_train, y_test = datos
    modelo = classifier(SVC(kernel=kernel, C=C, probability=idx == 7, **kwargs))
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="weighted", zero_division=np.nan
    )
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    metricas = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    X = vstack((X_train, X_test))
    y = pd.concat((y_train, y_test))

    plot_decision_boundary(idx, titulo, X, y, modelo)

    if idx == 7:  # caso 8: curvas ROC y PR
        plot_roc_pr(modelo, X_test, y_test)

    return metricas


for idx, datos in enumerate(dataframes):
    print(f"====== Caso {idx + 1} ======")
    rbf_ovr = entrenar_y_evaluar(
        idx, "RBF OvR", datos, OneVsRestClassifier, "rbf", C=1.0, gamma="auto"
    )
    print("RBF OvR:", rbf_ovr)
    rbf_ovo = entrenar_y_evaluar(
        idx, "RBF OvO", datos, OneVsOneClassifier, "rbf", C=1.0, gamma="auto"
    )
    print("RBF OvO:", rbf_ovo)
    lineal_ovr = entrenar_y_evaluar(
        idx, "Lineal OvR", datos, OneVsRestClassifier, "linear", C=1.0
    )
    print("Lineal OvR:", lineal_ovr)
    lineal_ovo = entrenar_y_evaluar(
        idx, "Lineal OvO", datos, OneVsOneClassifier, "linear", C=1.0
    )
    print("Lineal OvO:", lineal_ovo)
