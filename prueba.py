from pathlib import Path
import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from imblearn.under_sampling import RandomUnderSampler

DATASET_KAGGLE_HANDLE = "keyushnisar/dating-app-behavior-dataset"
DATASET_FILE = "dating_app_behavior_dataset.csv"
DATASET_Y_COLUMN = "swipe_right_label"
DATASET_IGNORE_COLUMNS = ["interest_tags"]

SEED=852
np.random.seed(SEED)


def obtener_dataset() -> pd.DataFrame:
    path = kagglehub.dataset_download(DATASET_KAGGLE_HANDLE)
    path = Path(path) / DATASET_FILE

    df = pd.read_csv(path)
    df = df.dropna()

    #Usamos el último dígito de la cédula de Stiven Saldarriaga (7)
    df = df.sample(5700, random_state=SEED)

    df = df.drop(columns=DATASET_IGNORE_COLUMNS)

    return df


def make_xy(df: pd.DataFrame):
    X = df.drop(columns=DATASET_Y_COLUMN)
    y = df[DATASET_Y_COLUMN].copy()
    return X, y


def make_train_test_split(df: pd.DataFrame):
    """
    Separar los datos en conjunto de entrenamiento y prueba
    """

    X, y = make_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
    )

    return X_train, X_test, y_train, y_test


def make_column_transformer(*, use_scaler=False):
    transformers: list = [("encoder", OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_include=object))]

    if use_scaler:
        transformers.append(("scaler", StandardScaler(), make_column_selector(dtype_include=["int64", "float64"])))

    return ColumnTransformer(transformers=transformers)


def preprocess(X_train, X_test, *, use_scaler):
    transformer = make_column_transformer(use_scaler=use_scaler)
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)
    return X_train, X_test


def make_clean_from_outliers_mask(X_train, *, cols, k=1.5):
    Q1 = X_train[cols].quantile(0.25)
    Q3 = X_train[cols].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k*IQR
    upper =  Q3 + k*IQR

    mask = ~((X_train[cols] < lower) |
             (X_train[cols] > upper)).any(axis=1)

    return mask


def sin_outliers_iqr(X_train, y_train, *, cols, k=1.5):
    mask = make_clean_from_outliers_mask(X_train, cols=cols, k=k)

    X_train_clean = X_train[mask]
    y_train_clean = y_train.loc[X_train_clean.index]

    return X_train_clean, y_train_clean


#Balanceo de clases:
#Balanceada:
def balancear_clases(df: pd.DataFrame):
    min_count = df[DATASET_Y_COLUMN].value_counts().min()
    clases_balanceadas = []
    for _, grupo in df.groupby(DATASET_Y_COLUMN):
        grupo_res = grupo.sample(min_count, random_state=SEED)
        clases_balanceadas.append(grupo_res)
    return pd.concat(clases_balanceadas)


def con_outliers_5(X_train, y_train, *, cols, k=1.5, target=0.05):
    mask_clean = make_clean_from_outliers_mask(X_train, cols=cols, k=k)

    n_clean = int(len(X_train)*(1-target))
    n_outlier = int(len(X_train)*target)

    X_train_clean = X_train[mask_clean].sample(n_clean, random_state=SEED)
    X_train_outlier = X_train[~mask_clean].sample(n_outlier, random_state=SEED, replace=True)

    X_train_5 = pd.concat([X_train_clean, X_train_outlier])
    y_train_5 = y_train.loc[X_train_5.index]

    return X_train_5, y_train_5


df = obtener_dataset()

dataframes=[]
for i in range(8):
    X_train, X_test, y_train, y_test = make_train_test_split(df)

    if i in {1, 3, 5, 7}:
        rus = RandomUnderSampler(random_state=SEED)
        X_train, y_train = rus.fit_resample(X_train, y_train)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if i in {0, 1, 4, 5}:
        X_train, y_train = sin_outliers_iqr(X_train, y_train, cols=numeric_cols)
    else:
        X_train, y_train = con_outliers_5(X_train, y_train, cols=numeric_cols)

    # categóricas a numéricas y escalado
    use_scaler = i in {4,5,6,7}
    X_train, X_test = preprocess(X_train, X_test, use_scaler=use_scaler)

    dataframes.append((X_train, X_test, y_train, y_test))
