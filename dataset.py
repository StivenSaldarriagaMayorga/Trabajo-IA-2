import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from ucimlrepo import fetch_ucirepo

DATASET_Y_COLUMN = "VisitorTypeRevenue"
DATASET_CAT_COLS = [
    "Month",
    "Weekend"
]
DATASET_NUM_COLS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

SEED = 852
np.random.seed(SEED)
tf.random.set_seed(SEED)


def obtener_dataset() -> pd.DataFrame:
    """
    Retorna un DataFrame de pandas con 5700 filas del dataset "Online Shoppers Purchasing
    Intention" descargado del repositorio UCI Machine Learning
    """

    dataset = fetch_ucirepo(id=468)

    X = dataset.data.features
    y = dataset.data.targets["Revenue"]

    columna_a_combinar = "VisitorType"
    nuevo_y = X[columna_a_combinar]+y.astype(str)
    nuevo_y.name = DATASET_Y_COLUMN

    df = pd.concat([X.drop(columns=columna_a_combinar), nuevo_y], axis="columns")

    # Usamos el último dígito de la cédula de Stiven Saldarriaga (7)
    df = df.sample(5700, random_state=SEED)

    return df


def make_xy(df: pd.DataFrame):
    """
    Retorna `X` (las columnas features de df) y `y` (la columna a predecir)
    """
    X = df.drop(columns=DATASET_Y_COLUMN)
    y = df[DATASET_Y_COLUMN].copy()

    return X, y


def make_train_test_split(df: pd.DataFrame):
    """
    Separa los datos en conjuntos de entrenamiento y prueba
    """

    X, y = make_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
    )

    return X_train, X_test, y_train, y_test


def balancear_clases(X_train, y_train):
    """
    Balancea las clases del conjunto de datos de tal forma que las clases mayoritarias tengan
    la misma cantidad de filas que las clases minoritarias
    """
    smote = SMOTE(random_state=SEED, k_neighbors=1)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    return X_train, y_train


def make_column_transformer(*, use_scaler=False):
    """
    Retorna un ColumnTransformer que convierte variables categóricas en numéricas mediante
    OneHotEncoding. Si `use_scaler` es True, entonces también realiza escalado mediante
    StandardScaler de las variables numéricas.
    """

    transformers: list = [
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
            DATASET_CAT_COLS,
        )
    ]

    if use_scaler:
        transformers.append(
            (
                "scaler",
                StandardScaler(),
                DATASET_NUM_COLS,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="passthrough")


def preprocess(X_train, X_test, *, use_scaler):
    """
    Convierte variables categóricas en numéricas mediante OneHotEncoding. Si `use_scaler` es True,
    entonces también realiza escalado mediante StandardScaler de las variables numéricas.
    """

    transformer = make_column_transformer(use_scaler=use_scaler)
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)
    return X_train, X_test, transformer


def make_clean_from_outliers_mask(X_train, *, k=1.5):
    """
    Retorna una máscara que al ser aplicada elimina outliers según el método IQR
    """

    Q1 = X_train[DATASET_NUM_COLS].quantile(0.25)
    Q3 = X_train[DATASET_NUM_COLS].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR

    mask = ~((X_train[DATASET_NUM_COLS] < lower) | (X_train[DATASET_NUM_COLS] > upper)).any(axis=1)

    return mask


def sin_outliers_iqr(X_train, y_train, *, k=1.5):
    """
    Retorna X_train y y_train sin outliers según el método IQR
    """

    mask = make_clean_from_outliers_mask(X_train, k=k)

    X_train_clean = X_train[mask]
    y_train_clean = y_train.loc[X_train_clean.index]

    return X_train_clean, y_train_clean


def con_outliers_5(X_train, y_train, *, k=1.5, target=0.05):
    """
    Retorna X_train y y_train con outliers de tal forma que estos son el 5% de los datos. Esto se
    realiza encontrando los outliers mediante el método IQR y seleccionando 95%, de la cantidad
    total de datos, del conjunto sin outliers y 5% del conjunto con outliers. En caso de que no
    hayan suficientes outliers para completar el 5% de la cantidad total de datos, entonces permite
    que hayan filas de outliers duplicadas.
    """

    mask_clean = make_clean_from_outliers_mask(X_train, k=k)

    n_clean = int(len(X_train) * (1 - target))
    n_outlier = int(len(X_train) * target)

    X_train_clean = X_train[mask_clean].sample(n_clean, random_state=SEED, replace=True)
    X_train_outlier = X_train[~mask_clean].sample(
        n_outlier, random_state=SEED, replace=True
    )

    X_train_5 = pd.concat([X_train_clean, X_train_outlier])
    y_train_5 = y_train.loc[X_train_5.index]

    return X_train_5, y_train_5


def calcular_metricas(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=np.nan)
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


df = obtener_dataset()
X_train_orig, X_test_orig, y_train_orig, y_test_orig = make_train_test_split(df)

# `le` codifica la columna objetivo para darle valores enteros a cada clase
le = LabelEncoder()
le.fit(y_train_orig)

dataframes = []
preprocesadores = []
for i in range(8):
    X_train = X_train_orig.copy()
    X_test = X_test_orig.copy()
    y_train = y_train_orig.copy()
    y_test = y_test_orig.copy()

    if i in {0, 1, 4, 5}:
        X_train, y_train = sin_outliers_iqr(X_train, y_train)
    else:
        X_train, y_train = con_outliers_5(X_train, y_train)

    use_scaler = i in {4, 5, 6, 7}
    # preprocess se encarga tanto de convertir variables categóricas en numéricas como del
    # escalado, dependiendo de si la columna es categórica o numérica. Se realiza la conversión de
    # categóricas a numéricas con OneHotEncoding independiende del dataset, y se realiza escalado
    # únicamente para los datasets 5, 6, 7 y 8
    X_train, X_test, preprocesador = preprocess(X_train, X_test, use_scaler=use_scaler)
    preprocesadores.append(preprocesador)

    if i in {1, 3, 5, 7}:
        X_train, y_train = balancear_clases(X_train, y_train)

    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    dataframes.append((X_train, X_test, y_train, y_test))


def generar_caso_de_prueba():
    distributions = {
        "Administrative": "discrete",
        "Administrative_Duration": "exponential",
        "Informational": "discrete",
        "Informational_Duration": "exponential",
        "ProductRelated": "discrete",
        "ProductRelated_Duration": "exponential",
        "BounceRates": "exponential",
        "ExitRates": "exponential",
        "PageValues": "exponential",
        "SpecialDay": "exponential",
        "Month": "discrete",
        "OperatingSystems": "discrete",
        "Browser": "discrete",
        "Region": "discrete",
        "TrafficType": "discrete",
        "Weekend": "discrete"
    }

    result = {}

    for col, dist in distributions.items():
        vc = df[col].value_counts()
        keys = vc.index.values
        values = vc.values
        total = values.sum()

        # 2/3 de probabilidad de introducir ruido
        apply_noise = np.random.random() < 2/3

        if dist == "discrete":
            probabilities = values / total
            choice = np.random.choice(keys, p=probabilities)

            if apply_noise:
                # desplazamiento aleatorio: escoger una categoría distinta si es posible
                if len(keys) > 1 and np.random.random() < 0.5:
                    alt_keys = [k for k in keys if k != choice]
                    choice = np.random.choice(alt_keys)
            result[col] = choice

        elif dist == "exponential":
            mean = df[col].mean()
            val = np.random.exponential(mean)

            if apply_noise:
                # agregar ruido gaussiano (positivos, pero más dispersos)
                noise_factor = np.random.uniform(0.5, 1.5)
                val *= noise_factor

                # ocasionalmente, valores más extremos
                if np.random.random() < 0.1:
                    val *= np.random.uniform(1.5, 3)

            result[col] = val

    return pd.DataFrame([result])


def generar_resumen_pruebas(pruebas):
    pruebas = pd.concat(pruebas)
    pruebas.index = range(1, 4)
    pruebas = pruebas.round(2).T
    pruebas.index.name = "Prueba #"
    return pruebas


def probar_modelo(modelo, preprocesador):
    pruebas = []
    for i in range(3):
        c = generar_caso_de_prueba()
        cn = preprocesador.transform(c)
        prediccion = modelo.predict(cn)
        prediccion = le.inverse_transform(prediccion)
        c["Predicción"] = prediccion
        pruebas.append(c)
        print(f"> Caso de prueba {i+1}:", c)
        print(">> Predicción:", prediccion)
    return generar_resumen_pruebas(pruebas)
