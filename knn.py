from dataset import calcular_metricas, dataframes
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt

metricas_knn = []
for i, (X_train, X_test, y_train, y_test) in enumerate(dataframes, start=1):
    print(f"\n===== Caso {i} =====")

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    metricas = calcular_metricas(y_test, y_pred)
    print(metricas)
    metricas_knn.append(metricas)

""" # Colores para aprobado y no aprobado
colors = ["red" if aprobado == 0 else "green" for aprobado in y_test]

# Gráfico de dispersión de los datos de prueba
plt.figure(figsize=(10, 6))
plt.scatter(
    X_test["Edad"],
    X_test["Monto solicitado"],
    c=colors,
    label="Muestra",
    s=80,
    edgecolors="k",
)

plt.xlabel("Edad")
plt.ylabel("Monto solicitado")
plt.title("Clasificación de Aprobación de Créditos (KNN)")
plt.grid(True)
plt.legend()
plt.show() """
