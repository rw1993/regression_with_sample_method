import numpy
import mnist_data
import k_fold
import projection
import math
#import itertoolsmodule

def one_or_one(func):

    def _f(*args, **kw):
        r = func(*args, **kw)
        if r > 1.0:
            return 1.0
        if r < -1.0:
            return -1.0
        else:
            return r
    return _f


class AELR(object):

    def __init__(self, k, B, lr):
        self.k = k
        self.B = B
        self.lr = lr
   
    def predict(self, X):
        return self.avg_w.dot(X)

    def predict_label(self, X):
        if self.predict(X) > 0:
            return 1
        else:
            return -1

            
    def train(self, fs, ls):
        B = self.B
        d = len(fs[0])
        ws = []
        k = self.k
        m = len(fs)
        self.indexs = [i for i in range(d)]
        z_p = numpy.ones(d)
        z_n = numpy.ones(d)
        for index, x in enumerate(fs):
           y = ls[index]
           w = (z_p - z_n) * B / (numpy.linalg.norm(z_p, 1) + \
                   numpy.linalg.norm(z_n, 1))
           ws.append(w)
           x_t = numpy.zeros(d)
           for r in range(k):
               x_index = numpy.random.choice(self.indexs)
               x_t[x_index] += d * x[x_index]
           x_t = x_t / k
           w_norm = numpy.linalg.norm(w, 1)
           if w_norm == 0:
               phi = -y
           else:
               ps = abs(w) / w_norm
               w_index = numpy.random.choice(self.indexs, p=ps)
               phi = w_norm * numpy.sign(w[w_index]) * x[w_index] - y
           g = phi * x_t
           for i in range(d):
               g[i] = max(min(g[i], 1.0/self.lr), -1.0/self.lr)
               z_p[i] *= numpy.exp(-self.lr*g[i])
               z_n[i] *= numpy.exp(self.lr*g[i])
        self.avg_w = sum(ws) / m


if __name__ == '__main__':
    fs, ls = mnist_data.get_3_5()
    Bs = [2**i for i in range(-15, 16)] 
    lrs = [2**i for i in range(-15, 16)] 
    best = (0, 0, 0)
    for B in Bs:
        for lr in lrs:
            r = AELR(4, B, lr)
            ac = k_fold.k_fold(fs, ls, 10, r)
            best = (ac, B, lr) if best < (ac, B, lr) else best
            print best
