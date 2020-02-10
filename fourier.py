from keras.datasets import cifar10
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt

from dtcwt_scattering.scattering_network import *

import pickle

from models.convs import model
from time import time
import cv2

# In[2]:


len_test_set=1000
len_train_set=2000

(x_train, label_train), (x_test, label_test) = cifar10.load_data()

x_train = x_train[:len_train_set]
x_test = x_test[:len_test_set]

y_train=np_utils.to_categorical(label_train[:len_train_set],10)
y_test=np_utils.to_categorical(label_test[:len_test_set],10)


def to_fourier(img):
    f_shift = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    #f_shift = np.fft.fftshift(f)
    f_complex = f_shift[:,:,0] + 1j*f_shift[:,:,1]
    f_abs = np.abs(f_complex) + 1 # lie between 1 and 1e6
    f_bounded = 20 * np.log(f_abs)
    f_img = 255 * f_bounded / np.max(f_bounded)
    f_img = f_img.astype(np.uint8)
    return f_img


from multiprocessing import Pool

pool = Pool(processes=8)

x_train_dtcwt = np.zeros(shape=(x_train.shape[0], 32, 32, 3))
t1 = time()

for i in range(x_train.shape[3]):
    print(i)
    x_train_dtcwt[:,:,:,i] = pool.map(to_fourier, x_train[:,:,:,i])
    print('Time for channel {} : {} seconds'.format(i+1, time()-t1))

t1 = time()
history = model.fit(x_train_dtcwt, y_train, epochs=100, verbose=1, batch_size=250, validation_split=0.1)
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
