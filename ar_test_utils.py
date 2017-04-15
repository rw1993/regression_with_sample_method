import arma
import numpy
import aerr_with_missing
import aelr_with_missing




def random_ar_model(noise, p):
    ars = [-1 + 2*numpy.random.random() for i in range(p)]
    a = arma.ARMA(ars, [], noise)
    return a

def prepare_ar(lenth, missing_rate, ar_model):
    p = ar_model.p
    time_series = [ar_model.generater.next() for i in range(lenth)]
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
    return train_fs, train_ls, test_fs, test_ls


def random_parameters_test(Regessor, noise=0.3, mr=0.0):
    try:
        ps = [i for i in range(1, 20)]
        p = numpy.random.choice(ps)
        print p
        model = random_ar_model(noise, p)
        train_fs, train_ls, test_fs, test_ls = prepare_ar(10000, mr, model)
        print train_fs
        r = Regessor(k=p, mr=mr)
        r.train(train_fs, train_ls)
        errors = [r.predict(x)-y for x, y in zip(test_fs, test_ls)]
        mse = sum(e*e for e in errors) / len(errors)
        return mse
    except:
        return None


def avg_mse(Regessor, noise=0.3, mr=0.0):
    mses = [random_parameters_test(Regessor, noise, mr) for i in range(20)]
    return sum(filter(lambda x: x, mses)) / len(filter(lambda x: x, mses))


if __name__ == "__main__":
    print avg_mse(aerr_with_missing.AERR)
    #print avg_mse(aelr_with_missing.AELR)
