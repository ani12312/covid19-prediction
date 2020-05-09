from sklearn.preprocessing import MinMaxScaler
import sys
import numpy as np
import pandas as pd
a=input("Enter the state:")
i=a+".h5"
n_input=5
n_features=1
try:
    from keras.models import load_model
    model=load_model(i)
    str3=a+"_csv"
    str2=pd.read_csv(str3)
    str2.drop("Unnamed: 0",axis=1,inplace=True)
    str2.drop("Name of State / UT",axis=1,inplace=True)
    str2.Date=pd.to_datetime(str2.Date)
    str2=str2.set_index('Date')
    train=str2
    scaler=MinMaxScaler()
    scaler.fit(train)
    train=scaler.transform(train)

    pred_list=[]
    batch=train[-n_input:].reshape(1,n_input,n_features)
    for i in range(n_input):
        pred_list.append((model.predict(batch)[0]))
        batch=np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
    from pandas.tseries.offsets import DateOffset
    add_dates=[str2.index[-1]+ DateOffset(days=x) for x in range(0,6) ]
    future_dates=pd.DataFrame(index=add_dates[1:],columns=str2.columns)
    str2_predict=pd.DataFrame(scaler.inverse_transform(pred_list),index=future_dates[-n_input:].index,columns=['Prediction'])
    str2_proj=pd.concat([str2,str2_predict],axis=1)
    str2_proj=str2_proj.fillna(0)
    str2_proj=str2_proj.astype({"Prediction":int})

    print(str2_proj.Prediction.tail(5))
except OSError as err:
    print("No Data Found")
