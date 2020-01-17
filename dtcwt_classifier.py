import numpy as np
from copy import copy
from dtcwt_scattering import dtcwt_scattering_2d
from models import svm, conv_1d


class DtcwtClassifier:
    MODEL_SELECTOR = {"svm": svm.SVMModel(), "conv_1d": conv_1d.ConvModel_1D()}

    DTCWT_TRANSFORM_SELECTOR = {
        "custom": dtcwt_scattering_2d.DtcwtScattering2D(),
    }

    def __init__(self, m=2, model="svm", dtcwt_transform="custom"):
        self.m = m
        self.transform2D = DtcwtClassifier.DTCWT_TRANSFORM_SELECTOR.get(
            dtcwt_transform, None
        )
        self.model = DtcwtClassifier.MODEL_SELECTOR.get(model, None)

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
            scatVector.append(c.flatten())
        return np.asarray(scatVector).flatten()

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
