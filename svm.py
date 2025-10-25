from dataset import dataframes
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier


def entrenar_y_evaluar(datos, classifier, kernel, *, C, **kwargs):
    X_train, X_test, y_train, y_test = datos
    modelo = classifier(SVC(kernel=kernel, C=C, **kwargs))
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    return accuracy_score(y_test, y_pred)


for idx, datos in enumerate(dataframes):
    print(f"====== Dataframe {idx} ======")
    rbf_ovr = entrenar_y_evaluar(datos, OneVsRestClassifier, "rbf", C=1.0, gamma="auto")
    print("RBF OvR:", rbf_ovr)
    rbf_ovo = entrenar_y_evaluar(datos, OneVsOneClassifier, "rbf", C=1.0, gamma="auto")
    print("RBF OvO:", rbf_ovo)
    lineal_ovr = entrenar_y_evaluar(datos, OneVsRestClassifier, "linear", C=1.0)
    print("Lineal OvR:", lineal_ovr)
    lineal_ovo = entrenar_y_evaluar(datos, OneVsOneClassifier, "linear", C=1.0)
    print("Lineal OvO:", lineal_ovo)
