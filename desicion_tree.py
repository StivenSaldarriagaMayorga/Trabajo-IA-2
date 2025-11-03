from dataset import calcular_metricas, dataframes, SEED as seed, le, probar_modelo, preprocesadores
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
import os


# ===================== #
#   PREPROCESAMIENTO    #
# ===================== #

def preparar_datos(dfcase):
    """Convierte los conjuntos de datos a matrices NumPy y elimina NaNs."""
    X_train, X_test, y_train, y_test = dfcase

    # Obtener nombres de características
    if hasattr(X_train, "columns"):
        feature_names = list(X_train.columns)
        X_train = X_train.values
        X_test = X_test.values
    else:
        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()
        if hasattr(X_test, "toarray"):
            X_test = X_test.toarray()
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]

    # Aplanar y limpiar NaNs
    y_train = np.ravel(np.array(y_train))
    y_test = np.ravel(np.array(y_test))
    m_tr = ~np.isnan(y_train)
    m_te = ~np.isnan(y_test)

    X_train, y_train = X_train[m_tr], y_train[m_tr]
    X_test, y_test = X_test[m_te], y_test[m_te]

    return X_train, X_test, y_train, y_test, feature_names


# ===================== #
#   ENTRENAMIENTO       #
# ===================== #

def entrenar_arbol(X_train, y_train, max_depth, random_state):
    """Entrena un árbol de decisión con criterio Gini."""
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


# ===================== #
#   GRÁFICOS Y ANÁLISIS #
# ===================== #

def graficar_arbol(model, feature_names):
    """Grafica el árbol de decisión."""
    plt.figure(figsize=(16, 12))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=list(le.classes_),
        filled=True,
        rounded=True
    )
    plt.title("Árbol de Decisión (criterio: Gini)")
    plt.show()


def importancias_caracteristicas(model, feature_names):
    """Muestra las características más importantes según el modelo (Top 10) con nombres reales."""
    # Mapeo de nombres reales (solo numéricas, sin la variable decisora)
    feature_map = [
        "Administrative", "Administrative_Duration", 
        "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay",
        "OperatingSystems", "Browser", "Region", "TrafficType"
    ]
    
    # Si hay más columnas (p. ej. dummies de Month o Weekend), completa automáticamente
    if len(feature_names) > len(feature_map):
        extras = [f"Extra_{i}" for i in range(len(feature_names) - len(feature_map))]
        feature_map.extend(extras)
    
    # Aplicar los nombres al DataFrame de importancias
    importances = pd.DataFrame({
        "feature": feature_map[:len(feature_names)],
        "importance": np.round(model.feature_importances_, 4)
    }).sort_values("importance", ascending=False)

    top10 = importances.head(10)

    print("\n===== CARACTERÍSTICAS MÁS IMPORTANTES =====")
    print(top10.to_string(index=False))

    # Gráfico mejorado
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(10, 6))
    bars = plt.bar(top10["feature"], top10["importance"], color=plt.cm.Blues(np.linspace(0.5, 1, len(top10))))
    plt.title("Importancia de Características (Top 10)", fontsize=14, weight='bold')
    plt.xlabel("Características", fontsize=12)
    plt.ylabel("Importancia (Gini)", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    # Añadir los valores encima de las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.005, f"{height:.2f}",
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()


# ===================== #
#   GUARDAR RESULTADOS  #
# ===================== #

def guardar_pruebas(i, model):
    """Ejecuta las pruebas y guarda los resultados en CSV."""
    pruebas = probar_modelo(model, preprocesadores[i])

    folder_path = "casos-trees"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, f"caso-{i + 1}.csv")
    pruebas.to_csv(file_path)
    print(pruebas)

    return pruebas


# ===================== #
#   PROCESO PRINCIPAL   #
# ===================== #

def Decision_Tree(i, dfcase, *, max_depth=5, random_state=None):
    """Pipeline completo del árbol de decisión."""
    if random_state is None:
        random_state = seed

    X_train, X_test, y_train, y_test, feature_names = preparar_datos(dfcase)

    model = entrenar_arbol(X_train, y_train, max_depth, random_state)
    y_pred = model.predict(X_test)


    #graficar_arbol(model, feature_names)
    importancias_caracteristicas(model, feature_names)

    guardar_pruebas(i, model)
    return calcular_metricas(y_test, y_pred)



"""Ejecuta el modelo para todos los casos disponibles."""
for i, dfcase in enumerate(dataframes):
    print(f"\n===== CASO {i} =====")
    Decision_Tree(i, dfcase, max_depth=5, random_state=seed)


