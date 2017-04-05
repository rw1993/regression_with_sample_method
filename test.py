from arma import ARMA
import my_yw
import e_expand_predicter
import expand_predicter
import yule_walker_impute_predicter
import kalman_impute_predicter
import my_kalman1984
import my_kalman_simple
import my_kalman
import ogd_impute_predicter
import sample_not_imput_predicter
from matplotlib import pyplot
import random
import pickle
import sys

def ran_test(missing_percent, ArPredicter, name, time=20):

    result = {}
    result["name"] = name
    result["errors"] = []
    result["mses"] = []

    def ran_a_test():
        a = ARMA([0.3, -0.4, 0.4, -0.5, 0.6], [], 0.3)
        time_series = [a.generater.next() for i in range(2000)]
        p = ArPredicter(len(a.alphas), max_x = max(time_series),)
        #                mr=missing_percent)
        errors = []
        for index, x in enumerate(time_series):
            if index < p.min_ob:
                p.predict_and_fit(x)
            elif random.random() > missing_percent:
                rec_x = p.predict_and_fit(x)
                errors.append(rec_x - x)
            else:
                p.predict_and_fit('*')
        mse = sum(map(lambda x: x**2, errors)) / len(errors)
        result['errors'].append(errors)
        result['mses'].append(mse)
        print mse
    for i in range(time):
        print i
        ran_a_test()

    pickle.dump(result, open("result_"+name+str(missing_percent), "wb"))
    return sum(result['mses']) / time


if __name__ == '__main__':
    missing_percents = [0.0, 0.1, 0.2, 0.3, 0.4]
    for missing_percent in missing_percents:
        #print ran_test(missing_percent, ogd_impute_predicter.ArPredicter, "ogd")
        #print ran_test(missing_percent, my_kalman1984.ArPredicter, "kalman")
        #print ran_test(missing_percent,  expand_predicter.ArPredicter,
        #               "expand_predicter", time=20)
        #print ran_test(missing_percent, yule_walker_impute_predicter.ArPredicter, "yw")
        #print ran_test(missing_percent,
        #               sample_not_imput_predicter.ArPredicter, "sample")
        print ran_test(missing_percent, my_yw.ArPredicter, "yw")
