import os
import pandas as pd

path = '../data/elec2/'
url = 'https://www.openml.org/data/get_csv/2419/electricity-normalized.arff'

os.makedirs(path, mode=0o777, exist_ok=True)

df = pd.read_csv(url)
df['class'] = (df['class'] == "UP").astype(int)

print(df.head())
print(df.tail())

df.dropna(inplace=True)
print("Dataframe shape {}".format(df.shape))

df.to_csv(path + 'eletricity.csv', index=False)
