import numpy as np
import pandas as pd
import tensorflow as tensorflow
from tensorflow.keras.preprocessing import sequence
import csv
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Masking, Convolution1D, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from math import sqrt

import datetime
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
            with open("/workspace/data/Movement-Quality-Assessment/po-cf-ex-1-features/"+ids+str(i)+".csv", 'r') as f:
                temp = list(csv.reader(f, delimiter = ","))
            temp = process(temp)
            temp = np.asarray(temp)
            temp = temp.astype(np.float64)
        except:
            print("Problem in:", ids, i)
            continue
        x_val.append(temp)

x_val = np.asarray(sequence.pad_sequences(x_val, padding='post',maxlen=2500)).astype(np.float64)

print("--- reading x_val and performing post-padding ---")

print("x_val shape:", x_val.shape)

length = x_val.shape[0]


df = pd.read_csv("/workspace/data/Movement-Quality-Assessment/dgx_data/ts_val.csv",header=None)
ts_val = np.array(df).astype(np.float32)

print("--- reading ts_val ---")

print("ts_val shape:",ts_val.shape)

print("--- randomizing ---")
indices = np.arange(len(ts_val))
indices = jumble_up(indices)
temp_x = x_val
x_val = []
temp_ts = ts_val
ts_val = []
for i in range(length):
    x_val.append(temp_x[indices[i]])
    ts_val.append(temp_ts[indices[i]])

print("--- normalizing ---")

max_x_val = -1
max_ts_val = -1
for i in range(len(ts_val)):
    max_x_val = max(max_x_val, np.max(np.abs(x_val[i])))
    max_ts_val = max(max_ts_val, np.max(np.abs(ts_val[i])))
x_val = x_val/max_x_val
ts_val = ts_val/max_ts_val

print("--- performing train-test split ---")

x_train, x_test, y_train, y_test = train_test_split(x_val,ts_val, test_size=0.3)

print("x_train", x_train.shape)
print("y_train", y_train.shape)
print("x_test", x_test.shape)
print("y_test", y_test.shape)

timesteps = 2500 
nr = 77   
n_dim = 11  

model = Sequential()
model.add(Masking(mask_value=0, input_shape=(2500, n_dim)))
model.add(LSTM(50, recurrent_dropout = 0.5, return_sequences = True), input_shape = (None,n_dim))
model.add(Dropout(0.3))

model.add(LSTM(25, recurrent_dropout = 0.5,return_sequences = True))
model.add(Dropout(0.2))

model.add(Convolution1D(16, 3), activations='sigmoid')

# model.add(Convolution1D(1, 2500))

model.add(LSTM(10, recurrent_dropout = 0.5))
model.add(Dropout(0.25))

model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer=tensorflow.keras.optimizers.Adam())
early_stopping = EarlyStopping(monitor='val_loss', patience = 100)

t = now()
history = model.fit(x_train,y_train, epochs=200, verbose=1,
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
plt.savefig("/workspace/data/Movement-Quality-Assessment/graph2.png", dpi=300)

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
plt.savefig('/workspace/data/Movement-Quality-Assessment/graph1.png', dpi=300)


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
np.savetxt("/workspace/data/Movement-Quality-Assessment/ans.csv",ans, delimiter=",")
