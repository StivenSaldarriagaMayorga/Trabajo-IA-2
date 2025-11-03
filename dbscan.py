from dataset import calcular_metricas, dataframes, SEED as seed, le, preprocesadores
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import os

sns.set(style="whitegrid")

def Decision_DBSCAN(i, dfcase, *, min_samples=16):
    X_train, X_test, _, _ = dfcase

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    # Concatenamos los datos de entrenamiento y prueba
    X_train = np.concatenate((X_train, X_test))





    # === Estimación visual y automática de eps ===
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X_train)
    distances, indices = neighbors_fit.kneighbors(X_train)
    distances = np.sort(distances[:, min_samples - 1])

    # Normalizamos para evitar escalas absurdas
    distances = distances / np.max(distances)

    # Detección automática del codo
    kneedle = KneeLocator(
        range(len(distances)),
        distances,
        curve='convex',
        direction='increasing'
    )
    eps_val = distances[kneedle.knee] if kneedle.knee is not None else np.median(distances)

    plt.figure(figsize=(8, 5))
    plt.plot(distances, label='Distancias ordenadas')
    if kneedle.knee is not None:
        plt.axvline(kneedle.knee, color='r', linestyle='--', label=f'Codo (eps ≈ {eps_val:.3f})')
    plt.title(f"K-dist plot - Caso {i+1}")
    plt.xlabel("Puntos ordenados")
    plt.ylabel(f"Distancia normalizada al {min_samples}° vecino")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\n===== CASO {i+1}: MODELO DBSCAN =====")
    print(f"Valor sugerido de eps: {eps_val:.4f}")

    # === Modelo DBSCAN ===
    model = DBSCAN(eps=eps_val, min_samples=min_samples)
    clusters = model.fit_predict(X_train)

    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    print(f"Clusters detectados (sin contar ruido): {n_clusters}")

    # === Reducción PCA para graficar ===
    pca = PCA(n_components=2, random_state=seed)
    X_pca = pca.fit_transform(X_train)

    plt.figure(figsize=(8, 6))
    for label in np.unique(clusters):
        if label == -1:
            plt.scatter(X_pca[clusters == label, 0], X_pca[clusters == label, 1],
                        c="gray", label="Ruido", alpha=0.5)
        else:
            plt.scatter(X_pca[clusters == label, 0], X_pca[clusters == label, 1],
                        label=f"Cluster {label}")
    plt.title(f"Distribución de Clusters - Caso {i+1}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Evaluación comparativa (solo referencial) ===
    if len(np.unique(clusters)) > 1:
        metricas = silhouette_score(X_train, clusters)
        print(f"Silhouette Score: {metricas:.4f}")
    else:
        metricas = {"silhouette": 0,}
        print("No se pueden calcular métricas: solo se detectó un grupo o ruido total.")



    return metricas

# === Ejecución ===
for i, dfcase in enumerate(dataframes):
    print(f"\n===== CASO {i+1} =====")
    Decision_DBSCAN(i, dfcase, min_samples=16)
