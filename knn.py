from dataset import dataframes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

metricas_knn = []

for i, (X_train, X_test, y_train, y_test) in enumerate(dataframes, start=1):
    print(f"\n===== Caso {i} =====")

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    metricas = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    metricas_knn.append(metricas)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")


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
