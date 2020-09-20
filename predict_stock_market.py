import pandas as pd
from datetime import datetime
from sklearn import datasets, linear_model
from sklearn.metrics import mean_absolute_error

hist = pd.read_csv('sphist.csv',parse_dates=['Date'])
hist.sort_values('Date',ascending=True,inplace=True)
hist['avg_5_days'] = pd.rolling_mean(hist.Close, window=5).shift(1)
hist['avg_30_days'] = pd.rolling_mean(hist.Close, window=30).shift(1)
hist['avg_365_days'] = pd.rolling_mean(hist.Close, window=365).shift(1)

clean_hist = hist[hist['Date'] > datetime(year=1951, month=1, day=2)].copy()
clean_hist.dropna(axis=0,inplace=True)
train = clean_hist[clean_hist['Date'] < datetime(year=2013, month=1, day=1)].copy()
test = clean_hist[clean_hist['Date'] >= datetime(year=2013, month=1, day=1)].copy()

features = ['avg_5_days','avg_30_days','avg_365_days']
lr = linear_model.LinearRegression()
lr.fit(train[features],train['Close'])
predictions = lr.predict(test[features])
test_msa = mean_absolute_error(test['Close'],predictions)
print(test_msa)