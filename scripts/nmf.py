
import numpy as np
import logging

logger = logging.getLogger('NMF')
logging.basicConfig(
    format='[%(asctime)s] %(name)s:%(levelname)s: %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def get_new_c(X, W, C, l):
    return C * np.divide(np.dot(W.T, X), l * C + np.dot(np.dot(W.T, W), C))


def get_new_w(X, W, C, l):
    return W * np.divide(np.dot(X, C.T), l * W + np.dot(W, np.dot(C, C.T)))


def error(X, W, C, l):
    return np.linalg.norm(X - np.dot(W, C)) / np.linalg.norm(X)


def reg_nmf(X, k=2, l=0, e=1e-12):
    n, m = X.shape
    low, high = X.min(), X.max()

    W = np.random.uniform(low=low, high=high, size=(n, k))
    C = np.random.uniform(low=low, high=high, size=(k, m))

    e_current, e_previous, it = error(X, W, C, l), float('inf'), 0
    while e_previous - e_current >= e:
        logger.debug("Error: %f, Iterations: %06d" % (e_current, it))
        if it % 2:
            C = get_new_c(X, W, C, l)
        else:
            W = get_new_w(X, W, C, l)

        e_previous, e_current, it = e_current, error(X, W, C, l), it + 1

    return W, C, it


if __name__ == '__main__':
    w = np.array([[1, 2], [3, 4]])
    c = np.array([[2, 4], [6, 8]])
    x = np.dot(w, c)

    w, c, it = reg_nmf(x, k=100)

    print('Iterations:', it)
    print('Arrays:', x, np.dot(w, c), sep='\n\n')
