from sklearn import datasets
from step import Step

class Fit(Step):
    def run(self, estimator, X, y, train=None):
        if train is not None:
            X = X[train]
            y = y[train]

        estimator.fit(X, y)
        self.result = estimator

    def dump(self):
        filename = os.path.join(self.get_dump_dirname(), 'estimator.pkl')
        joblib.dump(self.result, filename)

    def load(self):
        filename = os.path.join(self.get_dump_dirname(), 'estimator.pkl')
        self.result = joblib.load(filename)

class ClassificationData(Step):
    def run(self):
        X,y = datasets.make_classification(**self.__kwargs__)
        return X,y
