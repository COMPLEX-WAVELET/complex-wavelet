from dtcwt_scattering import DtcwtScattering2D
import numpy as np
from copy import copy
from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop
from time import time
from statistics import mean


class DtcwtConvClassifier:
    def __init__(self, m=2):
        self.transform2D = DtcwtScattering2D()
        self.m = m
        ## DEFINITION MODEL
        #self.model = SVC(kernel="linear", probability=True)
        self.model = Sequential()
        self.model.add(layers.Conv1D(20, 5, input_shape=(127,16)))
        self.model.add(layers.Conv1D(20, 3))
        self.model.add(layers.Conv1D(20, 3))
        self.model.add(layers.GlobalAveragePooling1D())
        self.model.add(layers.Dense(10, activation='softmax'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def __resize_image(self, x):
        current_size = len(x)
        new_size = 2 ** (int(np.log2(current_size)) + 1)
        return np.pad(
            x,
            pad_width=int((new_size - current_size) / 2),
            mode="constant",
            constant_values=0,
        )

    def __to_scat_vector(self, x):
        x_c = copy(x)
        x_c = self.__resize_image(x_c)
        scatCoef = self.transform2D.transform(np.asarray(x_c), self.m)
        scatVector = []
        for c in scatCoef:
            scatVector = scatVector + [c.flatten()]
        return scatVector

    def fit(self, X, y):
        scatX = []
        times = []
        for i in range(len(X)):
            if i%25 == 0:
                print('{}/{}'.format(i, len(X)))
            t = time()
            scatX.append(self.__to_scat_vector(X[i]))
            times.append(time()-t)
        print('Mean computing time : ', mean(times))
        print('Total computing time : ', sum(times))
        scatX = np.array(scatX)
        print('SCATX SHAPE :', scatX.shape)
        history = self.model.fit(scatX, y, batch_size=100, verbose=1, epochs=100, validation_split=0.2)
        return history

    def predict(self, X):
        scatX = []
        for i in range(len(X)):
            scatX.append(self.__to_scat_vector(X[i]))
        return self.model.predict(scatX)

    def evaluate(self, X, y):
        scatX = []
        for i in range(len(X)):
            scatX.append(self.__to_scat_vector(X[i]))
        return self.model.evaluate(scatX, y)
