#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import cifar10
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt

from dtcwt_scattering.scattering_network import *

import pickle

from models.convs import model
from time import time


# In[2]:


len_test_set=1000
len_train_set=2000

(x_train, label_train), (x_test, label_test) = cifar10.load_data()

x_train = x_train[:len_train_set]
x_test = x_test[:len_test_set]

y_train=np_utils.to_categorical(label_train[:len_train_set],10)
y_test=np_utils.to_categorical(label_test[:len_test_set],10)


# In[3]:


from multiprocessing import Pool
from dtcwt_scattering.scattering_network import filled

pool = Pool(processes=8)

x_train_dtcwt = np.zeros(shape=(x_train.shape[0], 127, 4, 4, 3))
t1 = time()

for i in range(x_train.shape[3]):
    x_train_dtcwt[:,:,:, :,i] = pool.map(filled, x_train[:,:,:,i])
    print('Time for channel {} : {} seconds'.format(i+1, time()-t1))


# In[4]:


x_train_dtcwt = np.array(x_train_dtcwt)
x_train_dtcwt = x_train_dtcwt.reshape(len_train_set, 127, -1, 3)

with open('data2000.pickle', 'wb') as pfile:
    pickle.dump(x_train_dtcwt, pfile)


# In[5]:


x_train_dtcwt.shape


# In[ ]:


from models.convs import wmodel
t1 = time()
history = wmodel.fit(x_train_dtcwt, y_train, epochs=100, verbose=1, batch_size=250, validation_split=0.1)
print('Training time : {} seconds'.format(time()-t1))

history = history.history

epochs = range(1, 101)

plt.plot(epochs, history['accuracy'], 'bo', label='Training accuracy')
plt.plot(epochs, history['val_accuracy'], 'b', label='Validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, history['loss'], 'ro', label='Training loss')
plt.plot(epochs, history['val_loss'], 'r', label='Validation loss')
plt.legend()
plt.show()
