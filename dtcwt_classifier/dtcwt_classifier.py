from dtcwt_scattering import DtcwtScattering2D
from tensorflow.keras import Sequential
import tensorflow.keras.layers as layers
import numpy as np
from copy import copy

class DtcwtClassifier:
    def __init__(self, m = 2):
        self.transform2D = DtcwtScattering2D()
        self.m = m
        self.model = Sequential()

    def __compile_model(self, s_input, s_output):
        if len(self.model.layers) != 0:
            print("Number of layers in model:", len(self.model.layers))
            self.model.pop()
        self.model.add(layers.Dense(s_output, activation='linear', input_shape=(s_input,)))
        self.model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    def __to_scat_vector(self, x):
        x_c = copy(x)
        x_c = np.pad(x_c, pad_width=int((64-len(x))/2), mode='constant', constant_values=0)
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
        self.model.fit(np.asarray(scatX), y)
    
    def predict(self, X):
        scatX = []
        for i in range(len(X)):
            scatX.append(self.__to_scat_vector(X[i]))

        return self.model.predict(scatX)
    
    def evaluate(self, X, y):
        scatX = []
        for i in range(len(X)):
            scatX.append(self.__to_scat_vector(X[i]))
        return self.model.evaluate(np.asarray(scatX), y)