import os
import pandas as pd

urls = [
    'http://stat-computing.org/dataexpo/2009/2007.csv.bz2',
    'http://stat-computing.org/dataexpo/2009/2008.csv.bz2']

path = 'data/flights/'

os.makedirs(path, mode=0o777, exist_ok=True)

flights = [pd.read_csv(url).query('Origin=="ORD" & Dest=="LGA"')\
           for url in urls]
df = pd.concat(flights, axis=0, ignore_index=True)[[
    'Year', 'Month', 'DayofMonth', 'DayOfWeek', 'CRSDepTime', 'CRSArrTime',
    'FlightNum', 'CRSElapsedTime', 'DepDelay'
]]

df.sort_values('CRSDepTime', inplace=True)
df.reset_index(drop=True, inplace=True)
df['class'] = (df['DepDelay'] > 15).astype(int)
df.drop(columns=['DepDelay'], inplace=True)

print(df.head())
print(df.tail())

df.dropna(inplace=True)
print("Dataframe shape {}".format(df.shape))

df.to_csv(path + 'data.csv', index=False)
