from pathlib import Path
import pandas as pd
import numpy as np
import kagglehub
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample

DATASET_KAGGLE_HANDLE = "nabihazahid/spotify-dataset-for-churn-analysis"
DATASET_FILE = "spotify_churn_dataset.csv"
DATASET_Y_COLUMN = "churn_plan_class"
DATASET_IGNORE_COLUMNS = ["user_id", "is_churned", "subscription_type"]

SEED=852
np.random.seed(SEED)


def obtener_dataset() -> pd.DataFrame:
    path = kagglehub.dataset_download(DATASET_KAGGLE_HANDLE)
    path = Path(path) / DATASET_FILE

    df = pd.read_csv(path)
    df = df.dropna()

    #Usamos el último dígito de la cédula de Stiven Saldarriaga (7)
    df = df.sample(5700, random_state=SEED)

    df['churn_plan_class']=df['subscription_type']+df['is_churned'].map({1:'_Churn',0:'_NoChurn'})

    df = df.drop(columns=DATASET_IGNORE_COLUMNS)

    print(df[DATASET_Y_COLUMN].value_counts())

    return df


def categoricas_a_numericas(df: pd.DataFrame):
    y_orig = df[DATASET_Y_COLUMN]
    df = pd.get_dummies(df, columns=df.drop(columns=DATASET_Y_COLUMN).select_dtypes(include="object").columns)
    le = LabelEncoder()
    df[DATASET_Y_COLUMN] = le.fit_transform(y_orig)
    return df


#Dividir el conjunto de datos
def obtener_caracteristicas_y_objetivo(df: pd.DataFrame):
    X = df.drop(columns=DATASET_Y_COLUMN)
    y = df[DATASET_Y_COLUMN]
    return X, y


def escalar_datos(X):
    scaler=StandardScaler()
    return scaler.fit_transform(X)


#Separar los datos en conjunto de entrenamiento y prueba
def dividir_datos(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=SEED)
    return X_train,X_test,y_train,y_test


def sin_outliers_iqr(df: pd.DataFrame, k=1.5):
    numeric_df = df.drop(columns=DATASET_Y_COLUMN).select_dtypes("number")
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(.75)
    IQR = Q3 - Q1
    lower = Q1 - k*IQR
    upper =  Q3 + k*IQR
    mask = ~((numeric_df < lower) | (numeric_df > upper)).any(axis=1)
    data_clean = df.loc[mask].reset_index(drop=True)
    X_clean = data_clean.drop(columns=[DATASET_Y_COLUMN])
    y_clean = data_clean[DATASET_Y_COLUMN]
    return X_clean, y_clean


#Balanceo de clases:
#Balanceada:
def balancear_clases(X, y):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name=DATASET_Y_COLUMN)
    df = pd.concat([X,y],axis=1)
    min_count = df[DATASET_Y_COLUMN].value_counts().min()
    clases_balanceadas = []
    for _, grupo in df.groupby(DATASET_Y_COLUMN):
        grupo_res = resample(grupo,replace=False,n_samples=min_count,random_state=SEED)
        clases_balanceadas.append(grupo_res)
    df_balanceado = pd.concat(clases_balanceadas)
    return df_balanceado.drop(columns=[DATASET_Y_COLUMN]), df_balanceado[DATASET_Y_COLUMN]


def con_outliers_5(df, target=0.05, tol=0.002):

    num = df.select_dtypes(include='number').drop(columns=[DATASET_Y_COLUMN], errors='ignore')
    k_lo, k_hi = 0.1, 3.0
    mask = None

    for _ in range(20):
        k = 0.5*(k_lo + k_hi)
        Q1, Q3 = num.quantile(.25), num.quantile(.75)
        IQR = Q3 - Q1
        lo, up = Q1 - k*IQR, Q3 + k*IQR
        mask = (num.lt(lo) | num.gt(up)).any(axis=1)
        rate = mask.mean()
        if abs(rate - target) <= tol:
            break
        k_lo, k_hi = (k, k_hi) if rate > target else (k_lo, k)

    data_5 = df.copy()
    # data_5['is_outlier_5pct'] = mask.astype(int)

    X_5 = data_5.drop(columns=[DATASET_Y_COLUMN])
    y_5 = data_5[DATASET_Y_COLUMN]
    return X_5, y_5


df = obtener_dataset()
df = categoricas_a_numericas(df)


dataframes=[]
for i in range(8):
    df_aux = df.copy()
    X, y = obtener_caracteristicas_y_objetivo(df_aux)
    X_train, X_test, y_train, y_test = dividir_datos(X, y)

    if i in {0, 1, 4, 5}:
        X_train, y_train = sin_outliers_iqr(df_aux)
    else:
        X_train, y_train = con_outliers_5(df_aux)

    if i in {1, 3, 5, 7}:
        X_train, y_train = balancear_clases(X_train, y_train)


    if i in {4,5,6,7}:
        X_train = escalar_datos(X_train)
        X_test = escalar_datos(X_test)

    dataframes.append((X_train, X_test, y_train, y_test))
