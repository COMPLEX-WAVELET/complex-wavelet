from sklearn.svm import SVC


class SVMModel:
    def __init__(self):
        self.model = SVC(kernel="linear", probability=True)

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
