import numpy as np
import random
import utils
import math
import regression_mixin
import mnist_data


class DDAERR(regression_mixin.RegressionMixin):

    def __init__(self, k, w_dim, B, learning_rate=None,
                 beta=0.3):
        self.beta = beta
        self.init(k, w_dim, B, learning_rate)
        

    def init(self, k, w_dim, B, learning_rate=None):
        self.k = k
        self.w_dim = w_dim
        w = [0.1*random.random() for i in range(self.w_dim)]
        self.lr = learning_rate
        self.indexs = [i for i in range(self.w_dim)]
        self.B = B
        self.ws = [w]

    def estimate_q(self, fs, ls):
        counts = [.0 for i in range(self.w_dim)]
        ssums = [.0 for i in range(self.w_dim)]
        indexs = [i for i in range(self.w_dim)]
        for index, f in enumerate(fs):
            print 'estimate'
            for i in range(self.k+1):
                x_index = random.choice(indexs)
                counts[x_index] += 1
                ssums[x_index] += f[x_index]**2
        a = map(lambda x, y: x/y,
                ssums, counts)
        e = self.w_dim * math.log(2, 2*self.w_dim/self.beta)
        e = e / (self.k+1) / len(fs)
        qs = map(lambda x: (x+13.0/6*e)**0.5, a)
        sqs = sum(qs)
        self.q = [q/sqs for q in qs]



    def train(self, fs, ls):
        e_fs = fs[:len(fs)/2]
        e_ls = ls[:len(fs)/2]
        t_fs = fs[len(fs)/2:]
        t_ls = ls[len(fs)/2:]
        self.estimate_q(e_fs, e_ls)
        
        for index, f in enumerate(t_fs):
            print 'train'
            y = t_ls[index]
            W = self.ws[-1]
            X = [.0 for i in range(self.w_dim)]
            for i in range(self.k):
                x_index = utils.sample_with_percents(self.q)
                X[x_index] += 1/self.q[x_index] * f[x_index]
            X = [x/self.w_dim for x in X]
            WW = [w*w for w in W]
            sWW = sum(WW)
            percents = [ww/sWW for ww in WW]
            w_index = utils.sample_with_percents(percents)
            w = W[w_index]
            o = sWW / w * f[w_index] - y
            gs = [x*o for x in X]
            vs = map(lambda x, g: x-self.lr*g,
                    W, gs)
            vv = [v*v for v in vs]
            svv = sum(vv)*0.5
            new_W = map(lambda x: x*self.B / max(self.B, svv),
                        vs)
            self.ws.append(new_W)

        


if __name__ == '__main__':
    fs, ls = mnist_data.get_3_5()
    r = DDAERR(500, len(fs[0]), 500, 0.3)
    r.train(fs, ls)
    print r.avg_w
    
