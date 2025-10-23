import pandas as pd

df=pd.read_csv('files/spotify_churn_dataset.csv')
df=df.set_index('user_id')

print(df)