from dtcwt_scattering import DtcwtScattering2D
from tensorflow.keras import Sequential
import tensorflow.keras.layers as layers
import numpy as np
from copy import copy
from keras.utils import to_categorical

class DtcwtClassifier:
    def __init__(self, m = 2):
        self.transform2D = DtcwtScattering2D()
        self.m = m
        self.model = Sequential()
        self.epochs = 100
        self.batch_size = 10

    def __compile_model(self, s_input, s_output):
        while len(self.model.layers) != 0:
            print("Number of layers in model:", len(self.model.layers))
            self.model.pop()

        self.model.add(layers.Dense(512, activation='relu', input_shape=(s_input,)))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dropout(0.25))
        self.model.add(layers.Dense(s_output, activation='softmax'))
        self.model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    def __resize_image(self, x):
        current_size = len(x)
        new_size = 2 ** (int(np.log2(current_size)) + 1)
        return np.pad(x, pad_width=int((new_size - current_size)/2), mode='constant', constant_values=0)

    def __to_scat_vector(self, x):
        x_c = copy(x)
        x_c = self.__resize_image(x_c)
        scatCoef = self.transform2D.transform(np.asarray(x_c), self.m)
        scatVector = []
        for c in scatCoef:
            scatVector = scatVector + [c.flatten()]
        return scatVector[0]
    
    def fit(self, X, y):
        scatX = []
        for i in range(len(X)):
            scatX.append(self.__to_scat_vector(X[i]))
        self.__compile_model(len(scatX[0]), max(y) + 1)
        self.model.fit(np.asarray(scatX), to_categorical(y), epochs=self.epochs, batch_size = self.batch_size, verbose=1)
    
    def predict(self, X):
        scatX = []
        for i in range(len(X)):
            scatX.append(self.__to_scat_vector(X[i]))
        return self.model.predict(scatX)
    
    def evaluate(self, X, y):
        scatX = []
        for i in range(len(X)):
            scatX.append(self.__to_scat_vector(X[i]))
        return self.model.evaluate(np.asarray(scatX), to_categorical(y))