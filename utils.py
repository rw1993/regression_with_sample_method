import random


def norm2(weights):
    ws = [w*w for w in weights]
    return sum(ws) ** 0.5

def norm1(weights):
    return sum(map(abs, weights))

def sample_with_percents(percents):
    r = 0
    index = 0
    flag = random.random()
    while True:
        r = r + percents[index]
        if r > flag:
            return index
        index += 1

