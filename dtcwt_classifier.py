from dtcwt_scattering import DtcwtScattering2D
import numpy as np
from copy import copy
from sklearn.svm import SVC


class DtcwtClassifier:
    def __init__(self, m=2):
        self.transform2D = DtcwtScattering2D()
        self.m = m
        self.model = SVC(kernel="linear", probability=True)

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
        return scatVector[0]

    def fit(self, X, y):
        scatX = []
        for i in range(len(X)):
            scatX.append(self.__to_scat_vector(X[i]))
        self.model.fit(scatX, y)

    def predict(self, X):
        scatX = []
        for i in range(len(X)):
            scatX.append(self.__to_scat_vector(X[i]))
        return self.model.predict(scatX)

    def evaluate(self, X, y):
        scatX = []
        for i in range(len(X)):
            scatX.append(self.__to_scat_vector(X[i]))
        return self.model.score(scatX, y)
