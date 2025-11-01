from prueba import dataframes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

for i, (X_train, X_test, y_train, y_test) in enumerate(dataframes, start=1):
    print(f"\n===== Caso {i} =====")

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    """ prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0) """

    print(f"Accuracy: {acc:.4f}")
    """ print(f"Precisión: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}") """


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
