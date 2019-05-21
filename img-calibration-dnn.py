

from __future__ import print_function

import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD, Adam
from keras.utils import plot_model

import numpy as np

import cv2

# load colorbar
data_train = cv2.imread('./figures_data/figure_colorbar.png')
print (data_train.shape)
# use the last column
# note, openCV uses BGR chanels.
data_train = data_train[:,-1,:]

# save output max
v_max = 2200
v_min = 200

output = np.linspace(v_max,v_min,data_train.shape[0])

# normalization
data1 = np.zeros((data_train.shape[0],data_train.shape[1]))
#   B chanel
data1[:,0] = data_train[:,0]/1.0/np.max(data_train[:,0])
#   G chanel
data1[:,1] = data_train[:,1]/1.0/np.max(data_train[:,1])
#   R chanel
data1[:,2] = data_train[:,2]/1.0/np.max(data_train[:,2])

train_input = data1

print (train_input.shape)

train_label = output

train_label_max = np.max(train_label)

train_label /= train_label_max

print (train_label.shape)

# traning parameters
batch_size = 20
output_num = 1
epochs = 300

#-----------------------------------------
# Network setup
#
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(3,)))
#model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dropout(0.2))
model.add(Dense(output_num))
model.summary()

#-----------------------------------------
# build the network
#
model.compile(loss='mean_absolute_error',
#              optimizer=RMSprop(),
              optimizer=Adam(),
#              optimizer=SGD(lr=0.01,clipnorm=1.),
              metrics=['mean_absolute_error'])
#-----------------------------------------
# training
#
history = model.fit(train_input,train_label,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)


#model.save_weights('weights_afterFit.h5')
#   plot_model(model,to_file='model.png')

#-----------predict data--------------------

data_retrive = cv2.imread('./figures_data/figure_data.png')

column_size = data_retrive.shape[0]
row_size = data_retrive.shape[1]

# flat the image to 1d array, data column-major
# the number, 3 stands for the BGR chanel
data_retrive = np.reshape(data_retrive,(column_size*row_size,3))
data_retrive = np.float32(data_retrive)

# normalizing input
data_retrive = data_retrive / np.max(data_retrive)

# prediction
predict_data = model.predict(data_retrive)
data_out = np.reshape(predict_data, (column_size, row_size))

# np.save("predict_data",predict_data_all)

import matplotlib.pyplot as plt
import matplotlib.cm as cm


plt.figure(0)
plt.subplot(111)
plt.imshow(data_out*train_label_max,cmap=cm.afmhot)
plt.colorbar()
plt.savefig("retrived_compare")

plt.show()

