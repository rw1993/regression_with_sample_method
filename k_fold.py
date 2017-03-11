from __future__ import division
import random
from sklearn.linear_model import Ridge
import mnist_data


def split(fs, ls, k):
    fs_ls = zip(fs, ls)
    random.shuffle(fs_ls)
    fs = [i[0] for i in fs_ls]
    ls = [i[1] for i in fs_ls]
    results = {i: ([],[]) for i in range(k)}
    g = random.randint(0, k-1)
    for index, f in enumerate(fs):
        group = g % k
        results[group][0].append(f)
        results[group][1].append(ls[index])
        g += 1
    return results


def k_fold(fs, ls, k, regressor):
    results = split(fs, ls, k)
    mses = []
    for i in range(k):
        #regressor.re_init()
        train_labels = []
        train_fs = []
        test_labels = results[i][1]
        test_fs = results[i][0]
        for j in range(k):
            if j != i:
                train_labels += results[j][1]
                train_fs += results[j][0]
        regressor.train(train_fs, train_labels)
        #'''
        ses = map(lambda f, l: (regressor.predict(f)-l)**2,
                  test_fs, test_labels)
        mse = sum(ses) / len(ses)
        '''
        ac = len(filter(None, map(lambda x, y: x == y,
                            [regressor.predict_label(f) for f in test_fs],
                            test_labels)))
        
        print ac / len(test_fs)
        '''
        print mse
        mses.append(mse)
        #'''
    return mses

if __name__ == "__main__":
    fs, ls = mnist_data.get_3_5()
    r = Ridge()


