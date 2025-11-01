from dataset import dataframes, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

def Decision_Tree(dfcase, *, max_depth=5, random_state=None):
    if random_state is None:
        random_state = seed
    X_train, X_test, y_train, y_test = dfcase
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


    y_train = np.ravel(np.array(y_train))
    y_test = np.ravel(np.array(y_test))
    m_tr = ~np.isnan(y_train)
    m_te = ~np.isnan(y_test)
    X_train, y_train = X_train[m_tr], y_train[m_tr]
    X_test, y_test = X_test[m_te], y_test[m_te]

    model = DecisionTreeClassifier(criterion="gini", max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Accuracy: {acc:.4f}")

    plt.figure(figsize=(16, 12))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=[
            "Free_Churn","Free_NoChurn",
            "Premium_Churn","Premium_NoChurn",
            "Family_Churn","Family_NoChurn",
            "Student_Churn","Student_NoChurn"
        ],
        filled=True,
        rounded=True
    )
    plt.title("Árbol de Decisión (criterio: Gini)")
    plt.show()
    return acc

for dfcase in dataframes:
    Decision_Tree(dfcase, max_depth=5, random_state=seed)
