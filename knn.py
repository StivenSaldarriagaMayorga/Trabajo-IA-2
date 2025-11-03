from dataset import calcular_metricas, dataframes
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


# Función para graficar regiones de decisión de KNN usando PCA
def plot_decision_boundary_knn(idx, X, y, model, le=None):
    """
    Genera un gráfico 2D de las regiones de decisión del modelo KNN
    proyectando los datos con PCA a 2 componentes.
    """
    plt.figure()
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Crear una malla 2D sobre el espacio proyectado
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    # Cada punto de la malla está en 2D (PCA1, PCA2)
    # Lo llevamos de vuelta al espacio original
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_original = pca.inverse_transform(grid_points)

    # Predicciones del modelo sobre la malla
    Z = model.predict(grid_points_original)
    Z = Z.reshape(xx.shape)

    # Fondo coloreado (regiones)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)

    # Puntos originales en espacio PCA
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors="k", s=30
    )

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"Regiones de decisión KNN: Caso {idx}")

    # Leyenda con etiquetas de clase
    if le is not None:
        labels = list(le.classes_)
    else:
        labels = [str(cls) for cls in np.unique(y)]

    plt.legend(
        handles=scatter.legend_elements()[0], labels=labels, title="Clases", loc="best"
    )
    plt.grid(True)
    plt.show()


metricas_knn = []

for i, (X_train, X_test, y_train, y_test) in enumerate(dataframes, start=1):
    print(f"\n===== Caso {i} =====")

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    metricas = calcular_metricas(y_test, y_pred)
    print(metricas)
    metricas_knn.append(metricas)

    plot_decision_boundary_knn(i, X_train, y_train, knn)

