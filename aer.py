import numpy as np
import random
import utils
import math
import regression_mixin


class AER(regression_mixin.RegressionMixin):

    def __init__(self, k, w_dim, B, learning_rate=None):
        self.init(k, w_dim, B, learning_rate)

    def re_init(self, learning_rate=None):
        self.init(self.k, self.w_dim, self.B, learning_rate)

    def init(self, k, w_dim, B, learning_rate=None):
        self.k = k
        self.w_dim = w_dim
        self.w = [0.0 for i in range(self.w_dim)]
        self.ws = [self.w]
        self._learning_rate = learning_rate
        self.w_inited = False
        self.B = B
        self.indexs = [i for i in range(self.w_dim)]

    @property
    def learning_rate(self):
        if self._learning_rate:
            return self._learning_rate

    def sample_with_weights(self):
        if not self.w_inited:
            return random.choice([i for i in range(self.w_dim)])
        weights = map(abs, self.w)
        sum_weights = sum(weights)
        percents = map(lambda x: x/sum_weights, weights)
        return utils.sample_with_percents(percents)

    def train(self, features, labels):
        assert len(features[0]) == self.w_dim
        assert len(features) == len(labels)
        for label_index, feature in enumerate(features):
            label = train_labels[label_index]
            C = [random.choice(self.indexs) for i in range(self.k/2)]
            v = [0.0 for i in range(self.w_dim)]
            for index in C:
                v[index] += 2.0 / float(self.k) * self.w_dim * feature[index]
            y = 0.0
            for i in range(self.k/2):
                w_index = self.sample_with_weights()
                w_norm = utils.norm1(self.w)
                sgn_w = 1 if self.w[w_index] > 0 else -1
                y += 2.0 / self.k * sgn_w * feature[w_index] 
            w = map(lambda x: (1.0-1.0/(index+1))*x,
                    self.w)
            delta = y - label
            v = map(lambda x: 2.0/(index+1)*self.learning_rate*delta*x,
                    v)
            w = map(lambda x, y: x-y, w, v)
            # find argmin
            self.ws.append(w)
            self.w = self.ws[-1]
