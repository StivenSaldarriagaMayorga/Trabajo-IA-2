from dataset import dataframes, SEED
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np

metricas_kmeans=[]
def Kmeans(dataframe):

    #Tomar los datos del dataframe
    X_train, X_test, _, _ = dataframe

    #método del codo para hallar el k
    inertia = []
    silhouette_scores = []
    K_range = range(2, 11)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = kmeans.fit_predict(X_train)
        inertia.append(kmeans.inertia_) # Métrica Inercia.
        silhouette_scores.append(silhouette_score(X_train, labels))  
    
    #Elegimos el k óptimo usando el silhouette score para que sea elegido 
    # #automáticamente por el algoritmo, ya que usando el método del codo sería ambiguo
    optimal_k = K_range[np.argmax(silhouette_scores)] 
    print(f"Mejor K según Silhouette: {optimal_k}")

    #Graficar los resultados
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, marker='o')
    plt.title('Método del Codo')
    plt.xlabel('Número de clusters (K)')
    plt.ylabel('Inercia')
    plt.xticks(K_range)
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 11), silhouette_scores, marker='o', color='green')
    plt.title('Silhouette Score por número de clusters')
    plt.xlabel('Número de clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()

    #Entrenar el modelo con el mejor k
    #Si bien silhouette define k=2 con el mejor, según el método del codo k=3 
    #suele ser un punto razonable para separar los clústers, y ese usaremos para el método
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=SEED, n_init='auto')
    clusters = kmeans.fit_predict(X_train)

    #Visualizamos usando PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_train)

    plt.figure(figsize=(8, 6))
    for i in range(optimal_k):
        plt.scatter(X_pca[clusters == i, 0], X_pca[clusters == i, 1], label=f'Cluster {i}')

    plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0],
                pca.transform(kmeans.cluster_centers_)[:, 1],
                s=250, c='black', marker='X', label='Centroides')
    plt.title('Visualización de Clusters con PCA')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.grid(True)
    plt.show()

    #Métricas de rendimiento
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_train, clusters)

    metricas_kmeans.append({
        "Inercia": inertia,
        "Silhouette": silhouette,
    })

for dataframe in dataframes:
    Kmeans(dataframe)

print(metricas_kmeans)




