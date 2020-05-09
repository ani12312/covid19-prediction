import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#import random
#import math
#import time
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error

from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
complete=pd.read_csv("complete.csv")
complete.drop("Total Confirmed cases (Indian National)",axis=1,inplace=True)
complete.drop("Total Confirmed cases ( Foreign National )",axis=1,inplace=True)
complete.drop("Latitude",axis=1,inplace=True)
complete.drop("Longitude",axis=1,inplace=True)
complete.drop("Cured/Discharged/Migrated",axis=1,inplace=True)
complete.drop("Death",axis=1,inplace=True)
state=complete["Name of State / UT"].tolist()
selected_state=list(set(state))
for i in selected_state:
    str1=i
    i=complete[complete["Name of State / UT"]==i]
    #print(i)
    str1=str1+"_csv"
    i.to_csv(str1)
for i in selected_state:
    str3=i+"_csv"
    str2=pd.read_csv(str3)
    str2.drop("Unnamed: 0",axis=1,inplace=True)
    str2.drop("Name of State / UT",axis=1,inplace=True)
    str2.Date=pd.to_datetime(str2.Date)
    str2=str2.set_index('Date')
    train=str2
    scaler=MinMaxScaler()
    scaler.fit(train)
    train=scaler.transform(train)
    
    n_input=5
    n_features=1
    if(len(train)<5):
        continue
    generator=TimeseriesGenerator(train,train,length=n_input,batch_size=len(train))
    model=Sequential()
    model.add(LSTM(200,activation='relu',input_shape=(n_input,n_features)))
    model.add(Dropout(0.15))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mse')
    model.fit_generator(generator,epochs=180)
    print(i)
    string=i+".h5"
    model.save(string)