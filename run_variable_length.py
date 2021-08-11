import numpy as np
import pandas as pd
from keras.preprocessing import sequence
import csv
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Bidirectional, LSTM, Masking, GlobalMaxPooling1D, Input
from keras.layers.convolutional import Convolution1D, Convolution2D
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import *
from keras.callbacks import EarlyStopping
from keras.losses import KLDivergence

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from math import sqrt

import datetime

from tensorflow.python.keras import activations
now = datetime.datetime.now

def process(temp):
    min_len = 3000
    for i in range(11):
        j = 0
        while temp[j][i] != '' and j<len(temp)-1:
            j+=1
        min_len = min(min_len, j)
#     print(min_len)
    return temp[:min_len]


def jumble_up(val):
    return np.random.permutation(val)

def train_generator(lim,x_train,y_train):
    n=0
    while n<lim:
        i = n%53
        xt = []
        yt = []
        xt.append(x_train[i])
        yt.append(y_train[i])
        xt = np.array(xt).astype('float32')
        yt = np.array(yt).astype('float32')
        yield xt,yt
        n+=1
        
def val_generator(lim,x_test,y_test):
    n=0
    while n<lim:
        i = n%24
        xt = []
        yt = []
        xt.append(x_test[i])
        yt.append(y_test[i])
        xt = np.array(xt).astype('float32')
        yt = np.array(yt).astype('float32')
        yield xt,yt
        n+=1


types = {'B_ID':8, 'E_ID':16, 'NE_ID':27, 'S_ID':10, 'P_ID':16}
x_val = []
x_id = []
for ids, vals in types.items():
    for i in range(1,vals+1):
        x_id.append(ids + str(i))
        try: 
            with open("/workspace/data/Movement-Quality-Assessment/po-cf-ex-1-features/"+ids+str(i)+".csv", 'r') as f:
                temp = list(csv.reader(f, delimiter = ","))
            temp = process(temp)
            temp = np.asarray(temp)
            temp = temp.astype(np.float64)
        except:
            print("Problem in:", ids, i)
            continue
        x_val.append(temp)

# x_val = np.asarray(sequence.pad_sequences(x_val, padding='post',maxlen=2500)).astype(np.float64)
print("-- x_val read as a list of 2D numpy arrays whose shapes are as follows")
for i in range(77):
    print(x_val[i].shape)

# po_val = []
# cf_val = []
# for ids in x_id:
#     try:
#         df = pd.read_excel("./KiMoRe/"+ids+"/Es1/Label/ClinicalAssessment_"+ids+".xlsx")
#     except:
#         print("problem in: ", ids)
#         continue
#     df = np.array(df).reshape((16,))
#     po_val.append(df[6])
#     cf_val.append(df[11])
# po_val = np.asarray(po_val).astype(np.float64)
# np.savetxt("po_val.csv", po_val, delimiter=",")
# cf_val = np.asarray(cf_val).astype(np.float64)
# np.savetxt("cf_val.csv", cf_val, delimiter=",")
# ts_val = po_val + cf_val
# np.savetxt("ts_val.csv", ts_val, delimiter=",")
# print("shape of CF: ", cf_val.shape)
# print("shape of PO: ", po_val.shape)
# print("shape of TS: ", ts_val.shape)
# # print(po_val)
# for i in range(len(ts_val)):
#     if np.isnan(ts_val[i]):
#         print(i)
# print("ts_val shape:",ts_val.shape)

df = pd.read_csv("./dgx_data/ts_val.csv",header=None)
ts_val = np.array(df).astype(np.float32)

print("-- randomize --")

indices = np.arange(len(ts_val))
indices = jumble_up(indices)
temp_x = x_val
x_val = []
temp_ts = ts_val
ts_val = []
for i in range(length):
    x_val.append(temp_x[indices[i]])
    ts_val.append(temp_ts[indices[i]])

print("-- normalize --")


max_x_val = -1
max_ts_val = -1
for i in range(len(ts_val)):
    max_x_val = max(max_x_val, np.max(np.abs(x_val[i])))
    max_ts_val = max(max_ts_val, np.max(np.abs(ts_val[i])))
x_val = x_val/max_x_val
ts_val = ts_val/max_ts_val
x_train, x_test, y_train, y_test = train_test_split(x_val,ts_val, test_size=0.3)

# timesteps = 2500 
nr = 77   
n_dim = 11  

model = Sequential()
# model.add(Masking(mask_value=0, input_shape=(2500, n_dim)))
model.add(Input(shape = (None,n_dim)))
model.add(Convolution1D(128,3,activation='relu'))
model.add(Convolution1D(64,3,activation='relu'))
model.add(LSTM(40, return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(20,return_sequences = True))
model.add(Dropout(0.2))
model.add(Convolution1D(16,3,activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

# model.compile(loss='binary_crossentropy', optimizer=Adam())
model.compile(loss='binary_crossentropy',metrics=['accuracy'], optimizer=Adam())
early_stopping = EarlyStopping(monitor='val_loss', patience = 50)

t = now()
history = model.fit(train_generator(7000,x_train,y_train),steps_per_epoch=10, epochs=500, verbose=1,
validation_data=val_generator(7000,x_test,y_test),validation_steps=3,callbacks = [early_stopping])
print('Training time: %s' % (now() - t))

# Plot the results
plt.figure(1)
plt.subplot(221)
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.subplot(222)
plt.plot(history.history['val_loss'])
plt.title('Validation Loss')
plt.subplot(223)
plt.plot(history.history['accuracy'])
plt.title('Accuracy')
plt.tight_layout()
plt.savefig('/workspace/data/Movement-Quality-Assessment/graph2.png')

# Plot the prediction of the CNN model for the training and validation sets
pred_test = []
for i in range(24):
    xt = []
    xt.append(x_test[i])
    xt = np.array(xt).astype('float32')        
    prediction = model.predict(xt)
    pred_test.append(prediction)
pred_test = np.array(pred_test).astype('float32') 
pred_test = pred_test.reshape(24)
pred_train = []
for i in range(53):
    xt = []
    xt.append(x_train[i])
    xt = np.array(xt).astype('float32')
    pred_train.append(model.predict(xt))
pred_train = np.array(pred_train).astype('float32') 
print(pred_train.shape)
pred_train = pred_train.reshape(53)
  

plt.figure(figsize = (8,8))
plt.subplot(2,1,1)
plt.plot(pred_train,'s', color='red', label='Prediction', linestyle='None', alpha = 0.5, markersize=6)
plt.plot(y_train,'o', color='green',label='Quality Score', alpha = 0.4, markersize=6)
plt.ylim([-0.1,1.1])
plt.title('Training Set',fontsize=18)
plt.xlabel('Sequence Number',fontsize=16)
plt.ylabel('Quality Scale',fontsize=16)
plt.legend(loc=3, prop={'size':14}) # loc:position
plt.subplot(2,1,2)
plt.plot(pred_test,'s', color='red', label='Prediction', linestyle='None', alpha = 0.5, markersize=6)
plt.plot(y_test,'o', color='green',label='Quality Score', alpha = 0.4, markersize=6)
plt.title('Testing Set',fontsize=18)
plt.ylim([-0.1,1.1])
plt.xlabel('Sequence Number',fontsize=16)
plt.ylabel('Quality Score',fontsize=16)
plt.legend(loc=3, prop={'size':14}) # loc:position
plt.tight_layout()
plt.savefig('/workspace/data/Movement-Quality-Assessment/graph1.png', dpi=300)
plt.show()


# Calculate the cumulative deviation and rms deviation for the validation set
test_dev = abs(np.squeeze(pred_test)-y_test)
# Cumulative deviation
mean_abs_dev = np.mean(test_dev)
# RMS deviation
rms_dev = sqrt(mean_squared_error(pred_test, y_test))
print('Mean absolute deviation:', mean_abs_dev)
print('RMS deviation:', rms_dev)