import pandas as pd

df=pd.read_csv('https://raw.githubusercontent.com/StivenSaldarriagaMayorga/Trabajo-IA-2/refs/heads/main/files/spotify_churn_dataset.csv')

#Usamos el último dígito de la cédula de Stiven Saldarriaga (7)
df=df.iloc[:5700]

df=df.set_index('user_id')
df['churn_plan_class']=df['subscription_type']+df['is_churned'].map({1:'_Churn',0:'_NoChurn'})
df=df.drop(columns=['subscription_type','is_churned'])


seed=852

#preprocesamiento
CC=df.copy()

#Normalización de los datos:
#CC:
CC['gender']=CC['gender'].map({'Male':0,'Female':1,'Other':2})
CC['country']=CC['country'].map({'AU':0,'US':1,'DE':2,'UK':3,'IN':4,'PK':5,'FR':6,'CA':7})
CC['device_type']=CC['device_type'].map({'Desktop':0,'Mobile':1,'Web':2})
CC['churn_plan_class']=CC['churn_plan_class'].map({'Free_Churn':0,'Free_NoChurn':1,
                                                     'Premium_Churn':2,'Premium_NoChurn':3,
                                                     'Family_Churn':4,'Family_NoChurn':5,'Student_Churn':6,'Student_NoChurn':7})


#Dividir el conjunto de datos
X = CC[['gender','age','country','listening_time','songs_played_per_day','skip_rate','device_type','ads_listened_per_week','offline_listening']]
y = CC['churn_plan_class']

print(X)
print(y)

