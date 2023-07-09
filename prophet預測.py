# -*- coding: utf-8 -*-
"""
Created on Thu May 19 22:27:26 2022

@author: a2793
"""

import prophet
import pandas as pd
import numpy 
import math
import time
import matplotlib.pyplot as plt
#from sklearn.cross_validation import train_test_split




df = pd.read_excel('TW_Sales_V1.xlsx')
#把年月欄位格式改為 yyyy-mm
for i in range(len(df)):
    df['年'][i]=df['年'][i].replace('年','')
    df['月'][i] = df['月'][i].replace('月','')
    df['年月'][i] = df['年月'][i].replace('年','-').replace('月','')

for j in range(len(df)):
    df['年月'][j] = str(int(df['年月'][j].split('-')[0])+1911)+'-'+str(int(df['年月'][j].split('-')[1]))
    
    
df['年月'] =pd.to_datetime(df['年月'])

#指定分析欄位
new_df = df[['年月','NET SALES']]
#重新命名
new_df.columns = ['ds', 'y']


#演算法
#訓練資料
train = new_df.drop(new_df.index[-12:])


model = prophet.Prophet()
model.fit(train)

future = list()
for i in range(1, 13):
    date = '2022-%02d' % i
    future.append([date])
    
    
future = pd.DataFrame(future, columns=['ds'])
future['ds']= pd.to_datetime(future['ds'])
future.head()

# do prediction
forecast = model.predict(future)
forecast.head()

# plot results
model.plot(forecast)
plt.scatter(x=new_df.ds, y=new_df.y)
plt.legend(['Actual', 'Predict'])
plt.show()