import os
import pandas as pd

url = 'http://www.liaad.up.pt/kdus/downloads/kdd-cup-10-percent-dataset'

path = 'data/intrusion/'

os.makedirs(path, mode=0o777, exist_ok=True)

df = pd.read_csv(url, header=None, compression='zip').rename(
    columns={41:'class'})
df['class'] = (df['class'] == 'normal.').astype(int)

print(df.head())
print(df.tail())

df.dropna(inplace=True)
print("Dataframe shape {}".format(df.shape))

df.to_csv(path + 'data.csv', index=False)
