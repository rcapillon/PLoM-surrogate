import numpy as np
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm

from PLoM_surrogate import dmaps

if __name__ == '__main__':
    x = np.linspace(0., 2., 100)
    y = np.power(x, 2.)

    fig, ax = plt.subplots()
    ax.plot(x, y, '-k', label='f(x)')
    ax.set_title('Graph of f(x) = x^2')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.grid()
    plt.legend()
    plt.show()
    # plt.savefig('./test.png')
