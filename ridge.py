import numpy as np
import random
import utils
import math
import regression_mixin
import mnist_data
import k_fold
from sklearn.linear_model import Ridge


class myRidge(regression_mixin.RegressionMixin):

    def __init__(self):
        self.r = Ridge()

    def re_init(self):
        self.r = Ridge(alpha=0.45)

    def predict(self, X):
        return self.r.predict([X])[0]

    def train(self, features, labels):
        self.r.fit(features, labels)

if __name__ == '__main__':
    fs, ls = mnist_data.get_3_5()
    r = myRidge()
    k_fold.k_fold(fs, ls, 10, r)
