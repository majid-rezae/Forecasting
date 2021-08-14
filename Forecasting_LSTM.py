####import all the packages####
import math
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import  Dense,LSTM,Dropout
from tensorflow.keras import backend  
from tensorflow.keras.models import Sequential
from pandas.tseries.offsets import DateOffset
####import and read database csv file####
path='C:\\Users\\Majid\\Desktop\\dados_ANG1.csv'          
file  = open(path, "r")
df= pd.read_csv(file, delimiter=',')
print(df)

####select Data column as index####
df["ds"] =pd.to_datetime(df.ds)
df=df.set_index ('ds')
#dataset=dataset.sort_values(by='Data')

####filter a select column####
df= df.replace(',','.', regex=True)
#df = dataset.filter(["Velocidade do vento (m/s)"])
#print(df)



# set datas between 0 and 1 for neural network model  
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(df)
X_replace= df.replace(',','.', regex=True)
df_scaled = scaler.fit_transform(df)
# convert it back to numpy array
X_np = X_replace.values
# set the object type as float
X_fa = X_np.astype(float)
# perdict for seven days
forecast_features_set = []
labels = []
for i in range(7,len(df)):
    forecast_features_set.append(df_scaled[i-7:i, 0])
    labels.append(df_scaled[i, 0])


    
forecast_features_set , labels = np.array(forecast_features_set ), np.array(labels)

forecast_features_set = np.reshape(forecast_features_set, (forecast_features_set.shape[0], forecast_features_set.shape[1], 1))
forecast_features_set.shape

# LSTM Model 
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(forecast_features_set.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=100))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

history = model.fit(forecast_features_set, labels, epochs = 10, batch_size = 100)

forecast_list=[]

batch=df_scaled[-forecast_features_set.shape[1]:].reshape((1,forecast_features_set.shape[1],1))

for i in range(forecast_features_set.shape[1]):
    forecast_list.append(model.predict(batch)[0])
    batch = np.append(batch[:,1:,:], [[forecast_list[i]]], axis=1)
df_predict=pd.DataFrame(scaler.inverse_transform(forecast_list),index=df[-forecast_features_set.shape[1]:].index, 
                        columns=["Forecasting"])

df_predict =pd.concat([df,df_predict],axis=1)
 
 
add_dates=[df.index[-1]+DateOffset(days=x) for x in range(0,8)]
future_dates=pd.DataFrame(index=add_dates[1:],columns=df.columns)
df_forecast=pd.DataFrame(scaler.inverse_transform(forecast_list),index=future_dates[-forecast_features_set.shape[1]:].index, 
                        columns=["Forecasting"])

df_forecast =pd.concat([df,df_forecast],axis=1)
df_forecast=df_forecast.drop(['y'], axis=1)
df_forecast=df_forecast.dropna()

#### save data to CSV file ####

#df_forecast.to_csv(r'C:\\Users\\Majid\\Desktop\\Forcasting_Velocidade.csv', index = True, header=True)
print(df_forecast)



     
 