#!pip install scikit-learn

#Importar las librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

#Se leen los datos
df=pd.read_csv('https://raw.githubusercontent.com/StivenSaldarriagaMayorga/Trabajo-IA-2/refs/heads/main/files/spotify_churn_dataset.csv')

#Usamos el último dígito de la cédula de Stiven Saldarriaga (7)
df=df.iloc[:5700]

#Preprocesamiento inicial
df=df.set_index('user_id')
df['churn_plan_class']=df['subscription_type']+df['is_churned'].map({1:'_Churn',0:'_NoChurn'})
df=df.drop(columns=['subscription_type','is_churned'])

#Estrablecer semilla
seed=852
np.random.seed(seed)


#preprocesamiento

#Normalización de los datos:
#CC:
def Normalizacion_CC(df):
    CC=df.copy()
    CC['gender']=CC['gender'].map({'Male':0,'Female':1,'Other':2})
    CC['country']=CC['country'].map({'AU':0,'US':1,'DE':2,'UK':3,'IN':4,'PK':5,'FR':6,'CA':7})
    CC['device_type']=CC['device_type'].map({'Desktop':0,'Mobile':1,'Web':2})
    CC['churn_plan_class']=CC['churn_plan_class'].map({'Free_Churn':0,'Free_NoChurn':1,
                                                     'Premium_Churn':2,'Premium_NoChurn':3,
                                                     'Family_Churn':4,'Family_NoChurn':5,'Student_Churn':6,'Student_NoChurn':7})
    return CC

#Dividir el conjunto de datos
def obtener_caracteristicas_y_objetivo(df):
    X = df[['gender','age','country','listening_time','songs_played_per_day','skip_rate','device_type','ads_listened_per_week','offline_listening']]
    y = df['churn_plan_class']
    return X, y

#Separar los datos en conjunto de entrenamiento y prueba
def dividir_datos(X,y,seed):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=seed)
    return X_train,X_test,y_train,y_test

#ED:
def escalar_datos(X_train,X_test):
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    return pd.DataFrame(X_train_scaled, columns=X_train.columns),pd.DataFrame(X_test_scaled, columns=X_test.columns)




#Balanceo de clases:
#Balanceada:
def balancear_clases(X, y):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name='churn_plan_class')
    df = pd.concat([X,y],axis=1)
    min_count = df['churn_plan_class'].value_counts().min()
    clases_balanceadas = []
    for _, grupo in df.groupby('churn_plan_class'):
        grupo_res = resample(grupo,replace=False,n_samples=min_count,random_state=seed)
        clases_balanceadas.append(grupo_res)
    df_balanceado = pd.concat(clases_balanceadas)
    return df_balanceado.drop(columns=['churn_plan_class']), df_balanceado['churn_plan_class']


"""print(X_train_scaled)
print(X_test_scaled)"""
#OUTLAIERS:

#SIN OUTLIERS


def sin_outliers_iqr(df, y='churn_plan_class', k=1.5):
    Q1 = df.drop(columns=y).quantile(0.25)
    Q3 = df.drop(columns=y).quantile(.75)
    IQR = Q3 - Q1
    lower = Q1 - k*IQR
    upper =  Q3 + k*IQR
    mask = ~((df.drop(columns=y) < lower) | (df.drop(columns=y) > upper)).any(axis=1)
    data_clean = df.loc[mask].reset_index(drop=True)
    X_clean = data_clean.drop(columns=[y]) #.values
    y_clean = data_clean[y] #.values
    return X_clean, y_clean

def con_outliers_5(df, y='churn_plan_class', target=0.05, tol=0.002):

    num = df.select_dtypes(include='number').drop(columns=[y], errors='ignore')
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
    data_5['is_outlier_5pct'] = mask.astype(int)

    X_5 = data_5.drop(columns=[y]).values
    y_5 = data_5[y].values
    return X_5, y_5, data_5



dataframes=[]
for i in range(8):
    df_aux = df.copy()
    df_aux = Normalizacion_CC(df_aux)
    X, y = obtener_caracteristicas_y_objetivo(df_aux)
    X_train, X_test, y_train, y_test = dividir_datos(X, y, seed)

    if i in {4,5,6,7}:
        X_train, X_test = escalar_datos(X_train, X_test)
    
    if i in {0, 1, 4, 5}:
        X_train, y_train = sin_outliers_iqr(pd.concat([X_train, y_train], axis=1))
    else:  
        X_train, y_train = con_outliers_5(pd.concat([X_train, y_train], axis=1))
    
    if i in {1, 3, 5, 7}:
        X_train, y_train = balancear_clases(X_train, y_train)
    
    dataframes.append((X_train, X_test, y_train, y_test))

print(dataframes)

   
