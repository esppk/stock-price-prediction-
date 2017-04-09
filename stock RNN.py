# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 20:26:31 2017

@author: Emrick
"""

import numpy as np
import pandas as pd

#importing the dataset
google = pd.read_csv("C:\\Users\\Emrick\\Documents\\google.csv")
#%%
google = google.iloc[:,6:7].values
#%%
training = google[:1500,]
testing = google[1500:,]
#%%
training = training[1:,]/training[:-1]-1
#%%
testing = testing[1:,]/testing[:-1]-1
#%%
train_x = training[:-1,]
train_y = training[1:,]
#%%
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#%%

model = Sequential()

model.add(LSTM(units = 4, activation= "sigmoid",
               input_shape = (None, 1)))
model.add(Dense(units = 1))
model.compile(optimizer = "adam", loss = "mean_squared_error")

model.fit(train_x.reshape(1498,1,1), train_y.reshape(1498,1), batch_size = 32, epochs =60)

#%%
pred = model.predict(testing.reshape(261,1,1))
#%%
pred = google[1500:-1]*(pred+1)
true_price = google[1501:]


#%%

import matplotlib.pyplot as plt

plt.plot(true_price, color = "red", label = "real price")
plt.plot(pred, color = "blue", label = "predicted price")
plt.title("google stock prediction")
plt.xlabel("Time")
plt.ylabel("price")
plt.legend()
plt.show()















