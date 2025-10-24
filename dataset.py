#!pip install scikit-learn

#Importar las librerías necesarias
import pandas as pd
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
    return X_train_scaled,X_test_scaled




#Balanceo de clases:
#Balanceada:
def balancear_clases(X, y):
    df = pd.concat([X, y], axis=1)
    min_count = df['churn_plan_class'].value_counts().min()
    clases_balanceadas = []
    for clase, grupo in df.groupby('churn_plan_class'):
        grupo_res = resample(grupo,replace=False,n_samples=min_count,random_state=seed)
        clases_balanceadas.append(grupo_res)
    df_balanceado = pd.concat(clases_balanceadas)
    return df_balanceado.drop('churn_plan_class', axis=1), df_balanceado['churn_plan_class']

