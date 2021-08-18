import numpy as np
import pandas as pd
import tensorflow as tensorflow
import make_data
from tensorflow.keras.preprocessing import sequence
import csv
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Masking, Convolution1D, Dropout, Activation, LeakyReLU, Flatten
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from math import sqrt, tanh

import datetime

from tensorflow.python.keras.backend import binary_crossentropy
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


types = {'B_ID':8, 'E_ID':16, 'NE_ID':27, 'S_ID':10, 'P_ID':16}
x_val = []
x_id = []
for ids, vals in types.items():
    for i in range(1,vals+1):
        x_id.append(ids + str(i))
        try: 
            with open("./po-cf-ex-1-features/"+ids+str(i)+".csv", 'r') as f:
                temp = list(csv.reader(f, delimiter = ","))
            temp = np.asarray(temp)
            temp=temp[:,:2]
            temp = temp.astype(np.float64)
        except:
            print("Problem in:", ids, i)
            continue
        x_val.append(temp)

x_val = np.asarray(sequence.pad_sequences(x_val, padding='pre',maxlen=1500)).astype(np.float64)

print("--- reading x_val and performing pre-padding ---")

print("x_val shape:", x_val.shape)

length = x_val.shape[0]


df = pd.read_csv("./dgx_data/po_val.csv",header=None)
ts_val = np.array(df).astype(np.float32)

print("--- reading ts_val ---")

print("ts_val shape:",ts_val.shape)
print(ts_val)
print("--- randomizing ---")
indices = make_data.randomize()
print(indices)
temp_x = x_val
x_val = []
temp_ts = ts_val
ts_val = []
for i in range(length):
    x_val.append(temp_x[indices[i]])
    ts_val.append(temp_ts[indices[i]])

x_val=np.asfarray(x_val)
ts_val=np.asfarray(ts_val)
print("--- normalizing ---")

max_x_val = -1
max_ts_val = -1
min_x_val = -1
min_ts_val = -1
sum_ts_val = 0
sum_x_val = 0

for i in range(len(ts_val)):
    max_x_val = max(max_x_val, np.max(x_val[i]))
    max_ts_val = max(max_ts_val, np.max(ts_val[i]))
    min_x_val = min(min_x_val, np.min(x_val[i]))
    min_ts_val = min(min_ts_val, np.min(ts_val[i]))
    sum_x_val+=x_val[i]
    sum_ts_val+=ts_val[i]

sum_x_val/=77
sum_ts_val/=77
x_val = ((x_val-min_x_val)/(max_x_val-min_x_val))
ts_val = (ts_val-min_ts_val)/(max_ts_val-min_ts_val)
# x_val = ((x_val)/(max_x_val))
# ts_val = (ts_val)/(max_ts_val)
x_val=np.asfarray(x_val)
ts_val=np.asfarray(ts_val)
ts_val = np.tile(ts_val, (5,1))
x_val = np.tile(x_val, (5,1,1))
print("------------------")
print(x_val.shape)
print("------------------")

print(ts_val.shape)

print("--- performing train-test split ---")

x_train, x_test, y_train, y_test = train_test_split(x_val,ts_val, test_size=0.3)

print("x_train", x_train.shape)
print("y_train", y_train.shape)
print("x_test", x_test.shape)
print("y_test", y_test.shape)

timesteps = 750 
nr = 77   
n_dim = 2  

model = Sequential()
model.add(Masking(mask_value=0, input_shape=(1500, n_dim)))
model.add(Convolution1D(256,5,padding ='same', strides = 2))
model.add(LeakyReLU())
model.add(Dropout(0.3))
model.add(Convolution1D(128,5,padding ='same', strides = 2))
model.add(LeakyReLU())
model.add(Dropout(0.2))
model.add(Convolution1D(64,3,padding ='same', strides = 2))
model.add(LeakyReLU())
model.add(Dropout(0.2))
model.add(Convolution1D(32,2,padding ='same', strides = 2))
model.add(LeakyReLU())
model.add(Dropout(0.2))
# model.add(Convolution1D(64,2,padding ='same', strides = 2))
# model.add(LeakyReLU())
# model.add(Dropout(0.2))
model.add(Flatten())
# model.add((LSTM(10, return_sequences = True)))
# model.add(Dropout(0.5))

# model.add(Dense(40, activation = 'tanh'))
# model.add(Dropout(0.5))

# model.add(Bidirectional(LSTM(16, return_sequences = True)))
# model.add(Dropout(0.3))

# model.add((LSTM(5)))
# model.add(Dropout(0.5))

# model.add(Dense(10, activation = 'tanh'))
# model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))
dropout_rate = 0.2
# model = Sequential()
# model.add(Masking(mask_value=0, input_shape = (timesteps,n_dim)))
# model.add(Convolution1D(100, 5, padding ='same', strides = 2))
# model.add(LeakyReLU())
# model.add(Dropout(dropout_rate))

# model.add(Convolution1D(30, 3, padding ='same', strides = 2))
# model.add(LeakyReLU())
# model.add(Dropout(dropout_rate))

# model.add(Convolution1D(10, 3, padding ='same'))
# model.add(LeakyReLU())
# model.add(Dropout(dropout_rate))

# model.add(Flatten())

# model.add(Dense(200))
# model.add(LeakyReLU())
# model.add(Dropout(dropout_rate))

# model.add(Dense(100))
# model.add(LeakyReLU())
# model.add(Dropout(dropout_rate))

# model.add(Dense(1))
# model.add(Activation('sigmoid'))

print(model.summary())

model.compile(loss='mse', optimizer=tensorflow.keras.optimizers.Adam())
# Early stopping if the validaton Loss does not decrease for 100 epochs
early_stopping = EarlyStopping(monitor='val_loss', patience = 100)

t = now()
history = model.fit(x_train,y_train,batch_size=5, epochs=100, verbose=1,
                    validation_data=(x_test,y_test),callbacks = [early_stopping])
print('Training time: %s' % (now() - t))

# Plot the results
plt.figure(1)
plt.subplot(221)
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.subplot(222)
plt.plot(history.history['val_loss'])
plt.title('Validation Loss')
plt.tight_layout()
plt.savefig("./graph2.png", dpi=300)
plt.show()

# Plot the prediction of the CNN model for the training and validation sets
pred_test = model.predict(x_test)
pred_test = pred_test.reshape(x_test.shape[0])
pred_train = model.predict(x_train)
pred_train = pred_train.reshape(x_train.shape[0])

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
plt.savefig('./graph1.png', dpi=300)
plt.show()

# Calculate the cumulative deviation and rms deviation for the validation set
test_dev = abs(np.squeeze(pred_test)-y_test)
# Cumulative deviation
mean_abs_dev = np.mean(test_dev)
# RMS deviation
rms_dev = sqrt(mean_squared_error(pred_test, y_test))
print('Mean absolute deviation:', mean_abs_dev)
print('RMS deviation:', rms_dev)

ans = np.arange(2)
ans[0] = mean_abs_dev
ans[1] = rms_dev
np.savetxt("./ans.csv",ans, delimiter=",")
