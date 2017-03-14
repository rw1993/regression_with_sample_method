import numpy as np
import random
import utils
import math
import regression_mixin
import mnist_data


class AERR(regression_mixin.RegressionMixin):

    def __init__(self, k, w_dim, B, learning_rate=None):
        self.init(k, w_dim, B, learning_rate)
        

    def init(self, k, w_dim, B, learning_rate=None):
        self.k = k
        self.w_dim = w_dim
        self.w = [0.1*random.random() for i in range(self.w_dim)]
        self.ws = [self.w]
        self.lr = learning_rate
        self.indexs = [i for i in range(self.w_dim)]
        self.B = B

    def train(self, fs, ls):
        for index, f in enumerate(fs):
            print index
            y = ls[index]
            x = [0.0 for i in range(self.w_dim)]
            indexs = [random.choice(self.indexs) for i in range(self.k)]
            for index in indexs:
                x[index] += f[index] * self.w_dim
            x = map(lambda i: i/self.k, x)
            sum_weight = utils.norm2(self.w) ** 2
            percents = [w*w/sum_weight for w in self.w]
            windex = utils.sample_with_percents(percents)
            delta = sum_weight / self.w[windex] * f[windex] -y
            g = [delta * i for i in x]
            v = [w-self.lr*i for i,w in zip(g, self.w)]
            v_norm = utils.norm2(v)
            if v_norm > self.B:
                self.w = [i/v_norm for i in v]
            self.ws.append(self.w)
            self.w = self.ws[-1]


if __name__ == '__main__':
    fs, ls = mnist_data.get_3_5()
    r = AERR(56, len(fs[0]), 20, 0.3)
    r.train(fs, ls)
    print r.avg_w
    
