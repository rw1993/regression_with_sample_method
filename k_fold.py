import random


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


def k_fold(fs, ls, k, regressor, measuments):
    splits = split(fs, ls, k)
    pre_results = {}
    for i in range(k):
        regressor.init()
        train_labels = []
        train_fs = []
        test_labels = results[i][1]
        test_fs = results[i][0]
        for j in range(k):
            if j != i:
                train_labels += results[j][1]
                train_fs += results[j][0]

