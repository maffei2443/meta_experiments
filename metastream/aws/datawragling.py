import os
import pandas as pd

url = 'https://www.openml.org/data/get_csv/20757277/aws-spot-pricing-market.arff'

path = 'data/aws/'

os.makedirs(path, mode=0o777, exist_ok=True)

df = pd.read_csv(url).query("region=='sa-east-1b' and "
                            "operating_system=='Linux/UNIX' and "
                            "instance_type=='m3.medium'")\
                     .drop(['region', 'operating_system',
                                   'instance_type'],axis=1)
df['year'] = 2017
df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
df.set_index('date', inplace=True)
df.sort_index(inplace=True)
del df['year']
periods = 10
df['mean'] = df.price.rolling('1H', min_periods=periods,
                              closed='right').mean()
df['std'] = df.price.rolling('1H', min_periods=periods,
                             closed='right').std()
df['class'] = (df.price > df['mean'] + df['std']).astype(int)
df = df.iloc[periods - 1:]
del df['price']

print(df.head())
print(df.tail())

df.dropna(inplace=True)
print("Dataframe shape {}".format(df.shape))

df.to_csv(path + 'data.csv', index=False)
