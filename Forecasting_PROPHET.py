from pandas import read_csv
from pandas import to_datetime
from pandas import DataFrame
from fbprophet import Prophet
 
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot
#from google.colab import files
import matplotlib.pyplot as plt
#upload=files.upload()

df = read_csv('dados_ANG1.csv' , delimiter=',')
df= df.replace(',','.', regex=True)
print(df)


#df=df.drop()
# prepare expected column names
df.columns = ['ds', 'y']
df['ds']= to_datetime(df['ds'])
dft=df.sort_values(by='ds')

m = Prophet()
m.fit(df)

# Python
#future = m.make_future_dataframe(periods=-1)
future=m.make_future_dataframe(periods=5, freq = "D", include_history = False)

# Python
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']] 
forecast[['ds', 'yhat' ]] 
forecast=forecast.filter(['ds', 'yhat', 'yhat_lower', 'yhat_upper' ])
forecast=forecast.rename(columns={'ds': 'Data','yhat': 'Predicao', 'yhat_lower':'Minimum', 'yhat_upper': 'Maximum'})
forecast.set_index('Data',drop=True,inplace=True)
print(forecast)