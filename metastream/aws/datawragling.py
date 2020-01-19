import os
import pandas as pd

url = 'https://www.openml.org/data/get_csv/20757277/aws-spot-pricing-market.arff'

path = 'data/aws/'

os.makedirs(path, mode=0o777, exist_ok=True)

df = pd.read_csv(url).query("region=='eu-west-1a' and "+
                                         "operating_system=='Linux/UNIX'")\
                                  .drop(['region', 'operating_system'],axis=1)
df = df.head(30000)
df = df[df.instance_type.str.contains('r3.')].sort_values(['month', 'day',
                                                           'hour', 'minute'])\
                                             .reset_index(drop=True)
df.price = (df.price > df.price.std()).astype(int)
df.rename(columns={'price':'class'}, inplace=True)
df = pd.get_dummies(df, columns=['instance_type'], drop_first=True)

print(df.head())
print(df.tail())

df.dropna(inplace=True)
print("Dataframe shape {}".format(df.shape))

df.to_csv(path + 'data.csv', index=False)
