import scipy as sp
import matplotlib.pyplot as plt
import numpy as np


def etha0(a, W):
    return 1 - np.exp(-2 * a ** 2 / W ** 2)


def lambd(a, W):
    x = 4 * (a / W) ** 2
    z = np.where(1 - np.exp(-x) * sp.special.iv(0, x) < 1e-12, 1e-12, 1 - np.exp(-x) * sp.special.iv(0, x))
    value = 2 * x * (np.exp(-x) * sp.special.iv(1, x)) / z * (np.log(2 * etha0(a, W) / z)) ** (-1)
    return value


def R(a, W):
    x = 4 * (a / W) ** 2
    z = np.where(1 - np.exp(-x) * sp.special.iv(0, x) < 1e-12, 1e-12, 1 - np.exp(-x) * sp.special.iv(0, x))
    return a * (np.log(2 * etha0(a, W) / z)) ** (-1 / lambd(a, W))


def integr(etha, sigm2, a, W):

    pdt = sp.stats.lognorm(s=np.sqrt(0.032), scale=np.exp(-7.4))
    l = np.where(2 / lambd(a, W) < 1e-12, 1e-12, 2 / lambd(a, W))
    _log = np.log(etha0(a, W) / etha)
    y = np.where(_log < 0, 0, _log ** (l - 1))
    z = np.where(_log < 0, 0, _log ** l)

    value = ((R(a, W) ** 2 / (sigm2 * lambd(a, W) * etha)) * y * np.exp(-R(a, W) ** 2 / (2 * sigm2) * z) *
             pdt.pdf(W ** 2))
    value[np.isnan(value)] = 0
    return value


sigm2 = 1.2e-5
a = 0.03

Wleft = 0
Wright = 0.9
ro = lambda etha: sp.integrate.quadrature(lambda W: integr(etha, sigm2, a, W), Wleft, Wright)[0]

exc = lambda etha2: sp.integrate.quadrature(lambda etha1: ro(etha1), etha2, 1)[0]


print(ro(0.75))

t = np.linspace(0.00001, 1)
plt.plot(t, np.vectorize(ro)(t))
plt.show()

print(exc(0.01))

t = np.linspace(0.00001, 1)
plt.plot(t, np.vectorize(exc)(t))
plt.show()
