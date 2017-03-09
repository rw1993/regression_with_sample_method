import numpy as np
import random
import utils
import math
import regression_mixin
import mnist_data


class DDAELR(regression_mixin.RegressionMixin):

    def __init__(self, k, w_dim, B, learning_rate=None,
                 beta=0.03):
        self.beta = beta
        self.init(k, w_dim, B, learning_rate)
        

    def init(self, k, w_dim, B, learning_rate=None):
        self.k = k
        self.w_dim = w_dim
        w = [0.1*random.random() for i in range(self.w_dim)]
        self.lr = learning_rate
        self.indexs = [i for i in range(self.w_dim)]
        self.B = B
        self.ws = []
        self.z_p = [1.0 for i in range(self.w_dim)]
        self.z_n = [1.0 for i in range(self.w_dim)]


    def estimate_q(self, fs, ls):
        counts = [.0 for i in range(self.w_dim)]
        ssums = [.0 for i in range(self.w_dim)]
        indexs = [i for i in range(self.w_dim)]
        for index, f in enumerate(fs):
            for i in range(self.k+1):
                x_index = random.choice(indexs)
                counts[x_index] += 1
                ssums[x_index] += f[x_index]**2
        a = map(lambda x, y: x/y,
                ssums, counts)
        e = self.w_dim * math.log(2, 2*self.w_dim/self.beta)
        e = e / (self.k+1) / len(fs)
        e = min(e, 1)
        qs = map(lambda x: x+13.0/6*e, a)
        sqs = sum(qs)
        self.q = [q/sqs for q in qs]



    def train(self, fs, ls):
        e_fs = fs[:len(fs)/2]
        e_ls = ls[:len(fs)/2]
        t_fs = fs[len(fs)/2:]
        t_ls = ls[len(fs)/2:]
        self.estimate_q(e_fs, e_ls)
        
        for index, f in enumerate(t_fs):
            y = t_ls[index]
            W = map(lambda x,y: x-y, self.z_p, self.z_n)
            W = map(lambda x: x*self.B, W)
            d = utils.norm1(self.z_p)+utils.norm1(self.z_n)
            W = map(lambda x: x/d, W)
            self.ws.append(W)
            X = [.0 for i in range(self.w_dim)]
            for i in range(self.k):
                x_index = utils.sample_with_percents(self.q)
                X[x_index] += 1/self.q[x_index] * f[x_index]
            X = [x/self.w_dim for x in X]
            WW = [abs(w) for w in W]
            sWW = sum(WW)
            if sWW != 0:
                percents = [ww/sWW for ww in WW]
                try:
                    w_index = utils.sample_with_percents(percents)
                except:
                    print W
                    raise
            else:
                w_index = random.choice([i for i in range(self.w_dim)])
            sign = 1.0 if W[w_index] > 0 else -1.0
            d = utils.norm1(W) * sign * f[w_index] - y
            gs = [x*d for x in X]
            for index, g in enumerate(gs):
                final_g = max(min(g, 1.0/self.lr), -1.0/self.lr)
                self.z_p[index] = self.z_p[index] * math.exp(-self.lr*final_g)
                self.z_n[index] = self.z_n[index] * math.exp(self.lr*final_g)
            
            

if __name__ == '__main__':
    fs, ls = mnist_data.get_3_5()
    r = DDAELR(50, len(fs[0]), 500, 0.3)
    r.train(fs, ls)
    print r.avg_w
