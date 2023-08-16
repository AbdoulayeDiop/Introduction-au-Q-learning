import matplotlib.pyplot as plt
import numpy as np


def discrete_vec_to_state(vec, cardinals):
    return sum([int(v*np.prod([cardinals[j] for j in range(i+1, len(cardinals))]))
        for i, v in enumerate(vec)])


def show_rewards(rewards, w=100):
    assert (w <= len(rewards))
    x = [k*w for k in range(len(rewards)//w)]
    y = [np.mean(rewards[k*w:(k+1)*w]) for k in range(len(rewards)//w)]
    plt.scatter(x, y)
    plt.show()