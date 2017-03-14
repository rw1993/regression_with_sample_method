import numpy
import mnist_data
import k_fold
import projection
import math
import itertoolsmodule

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


class GAELR(object):

    def __init__(self, k, B, lr=None, p_parameter=100):
        self.k = k
        self.B = B
        self.lr = lr
        self.p_parameter = p_parameter
   
    def predict(self, X):
        return self.avg_w.dot(X)

    def predict_label(self, X):
        if self.predict(X) > 0:
            return 1
        else:
            return -1
    def get_ps(self, x, y):
        k = self.k
        for r in range(k+1):
            index = numpy.random.choice(self.indexs)
            self.counts[index] += 1
            self.square_sums[index] += x[index] ** 2


            
    def train(self, fs, ls):
        B = self.B
        d = len(fs[0])
        ws = []
        k = self.k
        m = len(fs)
        self.lr = 1.0 / 4 / B **2 * (2.0*k*numpy.log2(2*d))**0.5
        self.indexs = [i for i in range(d)]
        self.counts = numpy.zeros(d)
        self.square_sums = numpy.zeros(d)
        z_p = numpy.ones(d)
        z_n = numpy.ones(d)
        for index, x in enumerate(fs):
           def train_this(x, y, ps=None):
               y = ls[index]
               w = (z_p - z_n) * B / (numpy.linalg.norm(z_p, 1) + \
                       numpy.linalg.norm(z_n, 1))
               ws.append(w)
               x_t = numpy.zeros(d)
               for r in range(k):
                   x_index = numpy.random.choice(self.indexs)
                   x_t[x_index] += x[x_index]
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
           train_this(x, ls[index])
           self.get_ps(x, ls[index])
        self.avg_w = sum(ws) / m
        ws = []
        A = self.counts / self.square_sums
        det = (d * numpy.log2(2.0*d/self.p_parameter)) / (k+1) / m
        det = min(det, 1)
        s = sum(A+13.0/6*det)
        ps = (A+13.0/6*det) / s
        self.ps = ps
        for index, x in enumerate(fs):
            train_this(x, ls[index], self.ps)
        self.avg_w = sum(ws) / m


if __name__ == '__main__':
    fs, ls = mnist_data.get_3_5()
    r = GAELR(5, 0.2, 0)
    k_fold.k_fold(fs, ls, 10, r)
