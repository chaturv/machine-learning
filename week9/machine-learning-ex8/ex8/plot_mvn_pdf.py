__author__ = 'vineet'

from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def load_data():
    prc = []
    vol = []
    f = open('/home/vineet/data/GOOG_20140101-20160529.csv')

    for line in f:
        # print line
        if line.startswith('#'):
            continue
        # parse
        tokens = line.split(',')
        prc.append(float(tokens[6]))
        vol.append(float(tokens[5]))

    return np.array(prc), np.array(vol)


def scale(X):
    mu = np.mean(X)
    std = np.std(X)
    return (X - mu) / std


def calc_covariance(X, Y):
    # m = length(prc);
    # prc_vol = horzcat(prc, vol);
    # deviations = prc_vol - (ones(m, m) * prc_vol) ./ m;
    # Sigma2 = (transpose(deviations) * deviations) ./ m;

    # create a final 2D vector of prices and volume
    XY = np.vstack((X, Y))
    XY = np.transpose(XY)

    Sigma2 = np.cov(XY, rowvar=0)

    # ones = np.ones((m, m), dtype=np.int16)
    # deviations = XY - np.divide(np.dot(ones, XY), m)
    # Sigma2 = np.divide(np.dot(np.transpose(deviations), deviations), m)
    #
    return Sigma2


STEP = 51

price, volume = load_data()

# price = np.sort(price)
# volume = np.sort(volume)

print 'min(price) = {v}'.format(v=min(price))
print 'max(price) = {v}'.format(v=max(price))
print 'min(volume) = {v}'.format(v=min(volume))
print 'max(volume) = {v}'.format(v=max(volume))

# scale
price = scale(price)
volume = scale(volume)

print 'min(price) = {v}'.format(v=min(price))
print 'max(price) = {v}'.format(v=max(price))
print 'min(volume) = {v}'.format(v=min(volume))
print 'max(volume) = {v}'.format(v=max(volume))


mean = [np.mean(price), np.mean(volume)]
print 'mu : '
print mean

var = [np.var(price), np.var(volume)]
print 'var : '
print var

Sigma2 = calc_covariance(price, volume)
print 'Sigma2 : '
print Sigma2


# ceate axis
# prc_x = np.linspace(min(price), max(price), 50)
# vol_y = np.linspace(min(volume), max(volume), 50)

# create axis
x_step = (max(price) - min(price)) / STEP
y_step = (max(volume) - min(volume)) / STEP

x, y = np.mgrid[min(price):max(price):x_step, min(volume):max(volume):y_step]
print 'x.shape = {s}'.format(s=x.shape)
print 'y.shape = {s}'.format(s=y.shape)


# plt.plot(y[1,:])

pos = np.empty(x.shape + (2,))
print 'pos.shape = {s}'.format(s=pos.shape)

# add labels
pos[:, :, 0] = np.linspace(min(price), max(price), STEP)
pos[:, :, 1] = np.linspace(min(volume), max(volume), STEP)

rv = multivariate_normal(mean, Sigma2)
pdf_pv = rv.pdf(pos)

fig = plt.figure()

# add a subplot
ax = fig.add_subplot(1, 1, 1, projection='3d')
surf = ax.plot_surface(x, y, pdf_pv, cmap=cm.coolwarm)

#fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()
