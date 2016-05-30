__author__ = 'vineet'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run():
    data = pd.Series(data=np.random.rand(100) * 100, index=[i for i in range(1,101)])
    bins = np.linspace(0, 100, 5)

    f = lambda x: pd.Series(np.histogram(x, bins=bins)[0], index=bins[:-1])
    df1 = data.apply(f)
    print df1

    plt.pcolor(df1.T)
    plt.show()


if __name__ == '__main__':
    run()
