from dataset import calcular_metricas, dataframes

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.neighbors import KNeighborsClassifier


def plot_knn_regions_pca2d(
    i, X_train, X_test, y_train, y_test, k=3, subsample=1200, p_lo=5, p_hi=95
):

    scaler = RobustScaler().fit(X_train)
    Xtr_s = scaler.transform(X_train)
    Xte_s = scaler.transform(X_test)

    pca = PCA(n_components=2, whiten=True, random_state=0).fit(Xtr_s)
    Xtr_2d = pca.transform(Xtr_s)
    Xte_2d = pca.transform(Xte_s)

    all_classes = np.unique(np.concatenate([y_train, y_test]))
    le = LabelEncoder().fit(all_classes)
    ytr_enc = le.transform(y_train)
    yte_enc = le.transform(y_test)
    n_classes = len(le.classes_)

    knn_2d = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn_2d.fit(Xtr_2d, ytr_enc)

    Xall_2d = np.vstack([Xtr_2d, Xte_2d])
    x_min, x_max = np.percentile(Xall_2d[:, 0], [p_lo, p_hi])
    y_min, y_max = np.percentile(Xall_2d[:, 1], [p_lo, p_hi])
    padx = 0.08 * (x_max - x_min)
    pady = 0.08 * (y_max - y_min)
    x_min, x_max = x_min - padx, x_max + padx
    y_min, y_max = y_min - pady, y_max + pady

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(10, 7))
    cmap = plt.cm.RdYlBu
    levels = np.arange(-0.5, n_classes + 0.5, 1.0)

    plt.contourf(xx, yy, Z, levels=levels, alpha=0.30, cmap=cmap)
    plt.contour(xx, yy, Z, levels=levels, colors="k", linewidths=0.6, alpha=0.7)

    rng = np.random.default_rng(0)
    idx_tr = rng.choice(len(Xtr_2d), size=min(subsample, len(Xtr_2d)), replace=False)
    idx_te = rng.choice(
        len(Xte_2d), size=min(subsample // 3, len(Xte_2d)), replace=False
    )
    sc_tr = plt.scatter(
        Xtr_2d[idx_tr, 0],
        Xtr_2d[idx_tr, 1],
        c=ytr_enc[idx_tr],
        cmap=cmap,
        edgecolors="k",
        s=20,
        alpha=0.9,
    )
    plt.scatter(
        Xte_2d[idx_te, 0],
        Xte_2d[idx_te, 1],
        c=yte_enc[idx_te],
        cmap=cmap,
        edgecolors="k",
        s=35,
        alpha=0.95,
        marker="o",
    )

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    handles, _ = sc_tr.legend_elements()
    plt.legend(
        handles, list(le.classes_), title="Clases", loc="lower right", framealpha=0.95
    )

    plt.title(
        f"Regiones de Decisión KNN (k={k}) – Caso {i} (PCA 2D robusto)", fontsize=14
    )
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
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

    plot_knn_regions_pca2d(i, X_train, X_test, y_train, y_test, k=5)
