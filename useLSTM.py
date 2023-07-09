# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 19:06:01 2022

@author: Johnny.Liu
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
import time
import numpy 
import math
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

'''
Year    年度
Scenario    版本
Month    月
Channel    通路
Customer    客戶名
Product    產品
EBIT    銷售淨利
GrsPft    銷售毛利
NET SALES    銷售金額
COGS    銷售成本
TOTSAG    銷售費用
SALES_UNITS    銷售量

目標預測:產品的銷售量,銷售金額,銷售成本,銷售費用
'''

df = pd.read_excel('TW_Sales.xlsx')

#####################ETL Space##################################

'''
資料處理並拆分訓練資料集
'''
#調整資料,年份轉int,月份轉int
for i in range(len(df)):
    print(i)
    df['Year'][i]=df['Year'][i].replace('YR','')
    df['Month'][i] = time.strptime(df['Month'][i],'%b').tm_mon


#先依照prodcut分類
prodcut_group_list=list(df.groupby('Product'))
#觀察資料
group1 = prodcut_group_list[0][1]
'''
年份跟月份需要合併2016-7之類的
重新排序之後取出
'NET SALES','SALES_UNITS','COGS','TOTSAG'
以上欄位

檢驗資料數 index=442的product資料數量最多
整理index442的資料作訓練

'''
df = df[['NET SALES','SALES_UNITS','COGS','TOTSAG']]


dataset = df.values
dataset = dataset.astype('float32')
# 正規化(normalize) 資料，使資料值介於[0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# 2/3 資料為訓練資料， 1/3 資料為測試資料
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# 產生 (X, Y) 資料集, Y 是下一期的乘客數(reshape into X=t and Y=t+1)
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
####################################################################

'''
使用LSTM演算法
'''
# 建立及訓練 LSTM 模型
model = Sequential()
model.add(LSTM(128, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


# 預測
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# 回復預測資料值為原始數據的規模
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate 均方根誤差(root mean squared error)
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# 畫訓練資料趨勢圖
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# 畫測試資料趨勢圖
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# 畫原始資料趨勢圖
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


#更換演算法
