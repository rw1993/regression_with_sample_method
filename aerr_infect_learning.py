import numpy
import k_fold
import projection
import math
from utils import one_or_one
import arma


class AERR(object):

    def __init__(self, k, B=1, lr=0.0625, mr=0.0):
        self.k = k
        self.B = B
        self.lr = lr
        self.mr = mr
  
    #@one_or_one
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
        w = 0.000001 * numpy.random.random(d)
        assert numpy.linalg.norm(w, 2) < B
        ws = [w]
        k = self.k
        m = len(fs)
        self.lr = (float(k)/(2.0*d*m)) ** 0.5
        self.indexs = [i for i in range(d)]
        for index, x in enumerate(fs):
            y = ls[index]
            if y == "*":
                continue
            indexs = [index for index, x_ in enumerate(x) if x_ != '*']
            w = ws[-1]
            x_t = numpy.zeros(d)
            mr = float(len(indexs)) / d
            k1 = int(k * mr)
            #print k1
            for i in range(k1):
                x_index = numpy.random.choice(indexs)
                if x_t[x_index] != '*':
                    x_t[x_index] += d * x[x_index]
            x_t = x_t / k1
            w_norm = numpy.linalg.norm(w, 2) ** 2
            percents = w * w / w_norm
            w_index = numpy.random.choice(self.indexs, p=percents)
            phi = w_norm * x[w_index] / w[w_index] / (1-self.mr) - y
            g = phi * x_t
            v = w - self.lr * mr * g
            new_w = v * B / max(B, numpy.linalg.norm(v, 2))
            ws.append(new_w)
        self.avg_w = sum(ws) / m


def prepare_ar(lenth, missing_rate):
    #a = arma.ARMA([0.3, -0.4, 0.4, -0.5, 0.6], [], 0.3)
    a = arma.ARMA([0.11, -0.5], [], 0.5, noise_type="uni")
    p = a.p
    time_series = [a.generater.next() for i in range(lenth)]
    train_lenth = int(len(time_series)*0.7)
    train_time_series = time_series[:train_lenth]
    test_time_series = time_series[train_lenth:]
    for index, t in enumerate(train_time_series):
        if numpy.random.random() < missing_rate:
            train_time_series[index] = '*'
    train_fs = []
    train_ls = []
    test_fs = []
    test_ls = []
    for index, t in enumerate(train_time_series):
        try:
            l = time_series[index+p]
            f = time_series[index: index+p]
            train_fs.append(f)
            train_ls.append(l)
        except:
            break
    for index, t in enumerate(test_time_series):
        try:
            l = time_series[index+p]
            f = time_series[index: index+p]
            test_fs.append(f)
            test_ls.append(l)
        except:
            break
    return train_fs, train_ls, test_fs, test_ls, train_time_series


if __name__ == '__main__':
    Bs = [2**i for i in range(-3, 6)]
    lrs = [2**i for i in range(-7, 7)]
    # aelr
    # B=32, lr=2**-7 missing = 0.0
    # B=16, lr=2**-7 missing = 0.1 0.0919
    # B=4, lr=2**-4 missing = 0.2 0.0925
    # aerr 
    # B=1 lr=0.0625
    for mr in [0.0, 0.1, 0.2, 0.3]:
        best = (99999999, 0, 0)
        for B in Bs:
            for lr in lrs:
                #try:
                #print B, lr
                mses = []
                for i in range(20):
                    #r = AERR(5, B, lr, mr)
                    r = AERR(2, 1, 0.0625, mr)
                    train_fs, train_ls, test_fs, test_ls, _ = prepare_ar(10000,
                                                                         mr)
                    r.train(train_fs, train_ls)
                    errors = [r.predict(x)-y for x, y in zip(test_fs, test_ls)]
                    mse = sum(e*e for e in errors) / len(errors)
                    mses.append(mse)
                avg_mse = sum(mses) / len(mses)
                best = (avg_mse, lr, B) if best[0] > avg_mse else best
                #except Exception, e:
                #    print e
                #    print B, lr, "failed"
        print mr, best
    '''
    for mr in [0.0, 0.1, 0.2, 0.3]:
        mses = []
        for i in range(20):
            #r = AERR(5, 1, 0.0625, mr)
            r = AERR(2, 1, 0.0625, mr)
            train_fs, train_ls, test_fs, test_ls, _ = prepare_ar(10000,
                                                              mr)
            r.train(train_fs, train_ls)
            errors = [r.predict(x)-y for x, y in zip(test_fs, test_ls)]
            mse = sum(e*e for e in errors) / len(errors)
            mses.append(mse)
        avg_mse = sum(mses) / len(mses)
        print avg_mse

    mses = []
    for i in range(20):
        print i
        #r = AERR(2, 1, 0.0625, mr)
        mr = 0.4 * numpy.random.random()
        train_fs, train_ls, test_fs, test_ls, train_time_series = prepare_ar(10000,
                                                                             mr)
       
        mr = sum(map(lambda x: 1.0 if x == '*' else 0.0,
                     train_time_series[:1000])) / 1000
        #r = AERR(5, 1, 0.0625, mr)
        r = AERR(2, 1, 0.0625, mr)
        r.train(train_fs, train_ls)
        errors = [r.predict(x)-y for x, y in zip(test_fs, test_ls)]
        mse = sum(e*e for e in errors) / len(errors)
        mses.append(mse)
    avg_mse = sum(mses) / len(mses)
    print avg_mse
    '''
