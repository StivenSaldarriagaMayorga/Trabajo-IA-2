import pandas as pd

df=pd.read_csv('https://raw.githubusercontent.com/StivenSaldarriagaMayorga/Trabajo-IA-2/refs/heads/main/files/spotify_churn_dataset.csv')
df=df.set_index('user_id')

print(df)