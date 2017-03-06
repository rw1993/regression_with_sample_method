import numpy as np
import random
import utils
import math
import regression_mixin
import mnist_data


class AELR(regression_mixin.RegressionMixin):

                
    def __init__(self, k, w_dim, B, learning_rate=None):
        self.init(k, w_dim, B, learning_rate)
        
    def init(self, k, w_dim, B, learning_rate=None):
        self.k = k
        self.w_dim = w_dim
        self.ws = []
        self.lr = learning_rate
        self.indexs = [i for i in range(self.w_dim)]
        self.B = B
        self.z_p = [1.0 for i in range(self.w_dim)]
        self.z_n = [1.0 for i in range(self.w_dim)]

    @property
    def avg_w(self):
        avg = [.0 for i in range(self.w_dim)]
        for w in self.ws:
            avg = map(lambda x, y: x+y,
                           avg, w)
        avg = [w/len(self.ws) for w in avg]
        return avg

    def train(self, fs, ls):
        for index, f in enumerate(fs):
            w = map(lambda x, y: x-y, self.z_p, self.z_n)
            w = map(lambda x: x*self.B, w)
            divider = utils.norm1(w)
            if divider != 0:
                w = map(lambda x: x/divider, w)
            self.ws.append(w)
            X = [.0 for i in range(self.w_dim)]
            indexs = [i for i in range(self.w_dim)]
            for r in range(self.k):
                x_index = random.choice(indexs)
                X[x_index] += self.w_dim * f[x_index]
            X = [x/self.k for x in X]
            norm1_w = utils.norm1(w)
            if divider == 0:
                w_index = random.choice(indexs)
            else:
                percents = [abs(w_i)/norm1_w for w_i in w]
                w_index = utils.sample_with_percents(percents)
            sign = 1.0 if w[w_index] > 0 else -1.0
            l = ls[index]
            direction = norm1_w * sign * X[w_index] -l
            g = map(lambda x: x*direction, X)
            for i in range(self.w_dim):
                g[i] = max(min(g[i], 1/self.lr), -self.lr)
                self.z_p[i] = self.z_p[i] * math.e ** (-self.lr*g[i])
                self.z_n[i] = self.z_n[i] * math.e ** (self.lr*g[i])


if __name__ == '__main__':
    fs, ls = mnist_data.get_3_5()
    r = AELR(50, len(fs[0]), 50, 0.3)
    r.train(fs, ls)
    print r.avg_w
