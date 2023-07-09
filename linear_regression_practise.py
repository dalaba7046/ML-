import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np
plt.style.use('ggplot')
# file_path = Path(r'D:\house')

# files = list(file_path.glob("*.csv"))

# df = pd.concat((pd.read_csv(f,encoding="utf8").drop([0]) for f in files))
# df.index = range(len(df))
#p='C:\\Users\\Johnny.Liu\\Desktop\\'
df_train = pd.read_csv(p+'test.csv',encoding='utf-8-sig')
df_train = df_train.sort_values(by='transaction_year_month_and_day')
#歸仁區價格
df = df_train.query('The_villages_and_towns_urban_district == "歸仁區"')
df.index = range(len(df))
#對數去偏
train_Y = np.log1p(df['total_price_NTD'])
#取出數值欄位,排除類別欄位
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
        
df2 = df[num_features]
MMEncoder = MinMaxScaler()
#繪製線型回歸,房價與總評數關係
sns.regplot(x = df2['building_shifting_total_area'], y=train_Y)
train_X = MMEncoder.fit_transform(df2)
estimator = LinearRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()

#去除離群值,將總評數限制在400以內,總價超過10萬
keep_indexs = (df2['building_shifting_total_area']<400) & (df2['total_price_NTD']>100000)
df3 = df2[keep_indexs]

train_Y2 = train_Y[keep_indexs]
sns.regplot(x = df3['building_shifting_total_area'], y=train_Y2)
plt.show()

# 做線性迴歸, 觀察分數
train_X2 = MMEncoder.fit_transform(df3)
estimator2 = LinearRegression()
cross_val_score(estimator2, train_X2, train_Y2, cv=5).mean()