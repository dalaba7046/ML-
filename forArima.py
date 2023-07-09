# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 14:11:54 2022

@author: a2793
"""

import numpy as np
import pandas as pd
from tabulate import tabulate
from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.metrics import mean_squared_error
import warnings
import statsmodels.api as sm
warnings.filterwarnings("ignore")

df = pd.read_excel('TW_Sales_V1.xlsx')

data = df[['NET SALES']]

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(311)
fig = plot_acf(data, ax=ax1, title="Autocorrelation")


fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(data.value); axes[0, 0].set_title('Original Series')
plot_acf(data.value, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(data.value.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(data.value.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(data.value.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(data.value.diff().diff().dropna(), ax=axes[2, 1])

plt.show()

#模型建立
model = sm.tsa.arima.ARIMA(data, order=(1,1,2))
result = model.fit()


#切分
price = data.values
length = int(len(price) * 0.8)
train = list(price[0:length])
test =  price[length:len(price)]
date = data.index[length:len(price)]
predictions = []
low_bound = []
up_bound = []

#預測
for i in range(len(test)):
   model = sm.tsa.arima.ARIMA(train, order=(5, 1, 2))
   model_fit = model.fit()
   pred = model_fit.forecast()[0]
   predictions.append(pred)
   real = test[i]
   train.append(real[0])   
   low_bound.append(model_fit.forecast()[0])
   up_bound.append(model_fit.forecast()[0])
 
   print(date[i] ,"|", 'Pred - '+str(round(pred[0],2)) ,"|", 'Real - '+str(real[0]))
 
MSE = mean_squared_error(test, predictions)
print('Mean Squared Error : '+str(round(MSE,4)))

