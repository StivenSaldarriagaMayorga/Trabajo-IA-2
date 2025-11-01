from dataset import dataframes, SEED
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import pandas as pd
#import seaborn as sns

def Kmeans(dataframe, metricas_kmeans):

    #Tomar los datos del dataframe
    X_train, X_test, _, _ = dataframe

    #método del codo para hallar el k
    '''inertia = []
    K_range = range(1, 11)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        kmeans.fit(X_train)
        inertia.append(kmeans.inertia_)'''   # Métrica Inercia.

    '''plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, marker='o')
    plt.title('Método del Codo')
    plt.xlabel('Número de clusters (K)')
    plt.ylabel('Inercia')
    plt.xticks(K_range)
    plt.grid(True)
    plt.show()'''

    #Silhouette score para evaluar la calidad de agrupamiento
    '''silhouette_scores = []

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = kmeans.fit_predict(X_train)
        score = silhouette_score(X_train, labels)
        silhouette_scores.append(score)'''

    '''plt.figure(figsize=(8, 5))
    plt.plot(range(2, 11), silhouette_scores, marker='o', color='green')
    plt.title('Silhouette Score por número de clusters')
    plt.xlabel('Número de clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()'''

    #Calcular K-means con el óptimo
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=SEED, n_init=10)
    clusters = kmeans.fit_predict(X_train)

    # Calculamos la distancia de cada punto a cada centroide
    '''distancias = cdist(X_train, kmeans.cluster_centers_, metric='euclidean')'''

    # Distancia del primer punto a todos los centroides
    '''print("Distancia del primer vino a cada centroide:")
    print(distancias[0])'''

    #Visualizamos usando PCA
    pca = PCA(n_components=2)
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
    metricas_kmeans = []
    Kmeans(dataframe, metricas_kmeans)

print(metricas_kmeans)




