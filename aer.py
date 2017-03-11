import numpy as np
import random
import utils
import math
import regression_mixin
import mnist_data
import k_fold


class AER(regression_mixin.RegressionMixin):

    def __init__(self, k, w_dim, B, learning_rate=None):
        self.init(k, w_dim, B, learning_rate)

    def re_init(self):
        super(AER, self).re_init()
        self.init(self.k, self.w_dim, self.B, self.learning_rate)

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

    def find_min(self, W):
        s = 0
        p = 0
        z = self.B
        U = [i for i in range(self.w_dim)]
        while U:
            k = random.choice(U)
            G = [i for i in U if W[i] >= W[k]]
            L = [i for i in U if W[i] < W[k]]
            del_p = utils.norm1(G)
            del_s = sum([W[i] for i in G])
            if s + del_s - (del_p + p) * W[k] < z:
                s = s + del_s
                p = p + del_p
                U = L
            else:
                U.remove(k)
        o = (s - z) / p
        return map(lambda w: max(w-o, 0), W)


    def train(self, features, labels):
        assert len(features[0]) == self.w_dim
        assert len(features) == len(labels)
        for label_index, feature in enumerate(features):
            print label_index
            label = labels[label_index]
            C = [random.choice(self.indexs) for i in range(self.k/2)]
            v = [0.0 for i in range(self.w_dim)]
            for index in C:
                v[index] += 2.0 / float(self.k) * self.w_dim * feature[index]
            y = 0.0
            for i in range(self.k/2):
                w_index = self.sample_with_weights()
                w_norm = utils.norm1(self.w)
                if self.w[index] > 0:
                    sgn_w = 1
                elif self.w[index] == 0:
                    sgn_w = 0
                else:
                    sgn_w = -1
                y += 2.0 / self.k * sgn_w *  w_norm * feature[w_index] 
            w = map(lambda x: (1.0-1.0/(index+1))*x,
                    self.w)
            delta = y - label
            v = map(lambda x: 2.0/((index+1)*self.learning_rate)*delta*x,
                    v)
            w = map(lambda x, y: x-y, w, v)
            w = self.find_min(w)
            self.ws.append(w)
            self.w = self.ws[-1]
            self.w_inited = True

if __name__ == '__main__':
    fs, ls = mnist_data.get_3_5()
    r = AER(4, len(fs[0]), 10, 0.3)
    k_fold.k_fold(fs, ls, 10, r)
