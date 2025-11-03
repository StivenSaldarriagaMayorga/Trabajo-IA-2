import os
import pandas as pd

def process_casos(dir="svm"):
    for i in range(3):
        dfs_fila = []
        for caso in range(i*3+1, min(i*3+4, 9)):
            print(caso)
            f = f"resultados/casos-de-prueba/{dir}/caso-{caso}.csv"
            df = pd.read_csv(f, index_col=0, header=0)
            # df = df.set_index(df["Prueba #"])
            # df.index.name = None
            # print(df.T)
            dfs_fila.append(df)
            os.remove(f)
        df = pd.concat(dfs_fila, axis=1)

        df.to_csv(f"resultados/casos-de-prueba/{dir}/{i}.csv")
