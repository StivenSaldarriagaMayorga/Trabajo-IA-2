from dataset import SEED as seed, dataframes_no_supervisado 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


def estimar_eps(X, min_samples, caso):
    # Esta función calcula el parámetro "eps" de DBSCAN de forma automática usando el método del codo (KneeLocator).
    # Se calculan las distancias al k-ésimo vecino más cercano y se grafica el K-dist plot para visualizar el cambio de concavidad.
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, min_samples - 1])
    distances = distances / np.max(distances)  # normalizar

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
    plt.title(f"K-dist plot - Caso {caso+1}")
    plt.xlabel("Puntos ordenados")
    plt.ylabel(f"Distancia normalizada al {min_samples}° vecino")
    plt.legend()
    plt.grid(True)
    plt.show()

    return eps_val


def entrenar_dbscan(X, eps_val, min_samples):
    # Esta función entrena el modelo DBSCAN con los valores de eps y min_samples calculados.
    # Retorna el modelo entrenado, los clusters generados y el número total de clusters (excluyendo el ruido).
    model = DBSCAN(eps=eps_val, min_samples=min_samples)
    clusters = model.fit_predict(X)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    print(f"Clusters detectados (sin contar ruido): {n_clusters}")
    return model, clusters, n_clusters


def graficar_clusters(X, clusters, caso):
    # Esta función aplica una reducción de dimensionalidad con PCA (a 2 componentes)
    # y genera una gráfica de dispersión que muestra los clusters detectados por DBSCAN.
    # Los puntos con etiqueta -1 se representan como ruido en color gris.
    pca = PCA(n_components=2, random_state=seed)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for label in np.unique(clusters):
        if label == -1:
            plt.scatter(X_pca[clusters == label, 0], X_pca[clusters == label, 1],
                        c="gray", label="Ruido", alpha=0.5)
        else:
            plt.scatter(X_pca[clusters == label, 0], X_pca[clusters == label, 1],
                        label=f"Cluster {label}")
            
    plt.scatter(X_pca[0, 0], X_pca[0, 1], c='red', marker='*', s=250, label='Nuevo punto')

    plt.title(f"Distribución de Clusters - Caso {caso+1}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.show()


def calcular_silhouette(X, clusters):
    # Esta función calcula la métrica de Silhouette Score para evaluar la calidad del clustering.
    # Un valor cercano a 1 indica buena separación de los clusters, mientras que valores negativos indican solapamiento.
    if len(np.unique(clusters)) > 1:
        sil = silhouette_score(X, clusters)
        print(f"Silhouette Score: {sil:.4f}")
        return sil
    else:
        print("No se pueden calcular métricas: solo se detectó un grupo o ruido total.")
        return 0


def Decision_DBSCAN(i, dfcase, *, min_samples=17):
    # Esta función principal coordina el flujo completo del modelo DBSCAN:
    # - Calcula eps automáticamente con el método del codo
    # - Entrena el modelo con los parámetros determinados
    # - Grafica los clusters resultantes con PCA
    # - Evalúa el clustering mediante el Silhouette Score
    X_train = dfcase
    print(f"\n===== CASO {i+1}: MODELO DBSCAN =====")

    eps_val = estimar_eps(X_train, min_samples, i)
    print(f"Valor sugerido de eps: {eps_val:.4f}")

    model, clusters, n_clusters = entrenar_dbscan(X_train, eps_val, min_samples)
    graficar_clusters(X_train, clusters, i)

    metricas = calcular_silhouette(X_train, clusters)
    return metricas

for i, dfcase in enumerate(dataframes_no_supervisado):
    Decision_DBSCAN(i, dfcase, min_samples=17)
