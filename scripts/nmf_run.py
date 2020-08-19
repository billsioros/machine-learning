
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nmf import reg_nmf

logger = logging.getLogger('NMF')
logging.basicConfig(
    format='[%(asctime)s] %(name)s:%(levelname)s: %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def benchmark(ks, es):
    n, m, low, high = 500, 1000, 0, 1

    X = np.random.uniform(low=low, high=high, size=(n, m))

    columns, data = ['K', 'Epsilon', 'Iterations', 'Time'], []

    for k in ks:
        for e in es:
            logger.info("k: %d, e: %.1e" % (k, e))
            W, C, iterations = reg_nmf(X, k=k, e=e)

            data.append(dict(zip(columns, [k, e, iterations])))

    return pd.DataFrame(data)


def measurements(ks, es, path, load=False):
    if load and path.is_file():
        return pd.DataFrame.from_csv(path)

    df = benchmark(ks, es)

    if not path.parent.is_dir():
        path.parent.mkdir(parents=True)

    df.to_csv(path)

    return df


if __name__ == "__main__":
    ks, es = [1, 10, 100], [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

    path = Path.cwd() / 'results' / 'second.csv'

    measurements(ks, es, path)
