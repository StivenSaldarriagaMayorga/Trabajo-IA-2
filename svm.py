from datetime import datetime
from prueba import dataframes
import numpy as np
import pandas as pd
from scipy.sparse import vstack
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


# Función para plotear los límites de decisión
def plot_decision_boundary(X, y, model):
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

    le = LabelEncoder()
    y = le.fit_transform(y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # Cada punto de la malla está en 2D (PC1, PC2)
    # Necesitamos llevarlo al espacio original para usar el modelo
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_original = pca.inverse_transform(grid_points)

    Z = model.predict(grid_points_original)
    Z = le.transform(Z)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, vmin=y.min(), vmax=y.max(), alpha=0.8, cmap=plt.cm.RdYlBu)

    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors="k")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Regiones de decisión SVM")

    plt.legend(
        handles=scatter.legend_elements()[0], labels=list(le.classes_), title="Clases"
    )

    plt.show()


def entrenar_y_evaluar(datos, classifier, kernel, *, C, **kwargs):
    X_train, X_test, y_train, y_test = datos
    modelo = classifier(SVC(kernel=kernel, C=C, **kwargs))
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="weighted", zero_division=np.nan
    )
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    X = vstack((X_train, X_test))
    y = pd.concat((y_train, y_test))

    plot_decision_boundary(X, y, modelo)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


for idx, datos in enumerate(dataframes):
    print(f"====== Caso {idx + 1} ======")
    rbf_ovr = entrenar_y_evaluar(datos, OneVsRestClassifier, "rbf", C=1.0, gamma="auto")
    print("RBF OvR:", rbf_ovr)
    rbf_ovo = entrenar_y_evaluar(datos, OneVsOneClassifier, "rbf", C=1.0, gamma="auto")
    print("RBF OvO:", rbf_ovo)
    lineal_ovr = entrenar_y_evaluar(datos, OneVsRestClassifier, "linear", C=1.0)
    print("Lineal OvR:", lineal_ovr)
    lineal_ovo = entrenar_y_evaluar(datos, OneVsOneClassifier, "linear", C=1.0)
    print("Lineal OvO:", lineal_ovo)
