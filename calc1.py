import scipy as sp
import matplotlib.pyplot as plt
import numpy as np


def etha0(a, W):
    return 1 - np.exp(-2 * (a / W) ** 2)


def lambd(a, W):
    x = 4 * (a / W) ** 2
    z = np.where(1 - np.exp(-x) * sp.special.iv(0, x) < 1e-15, 1e-15, 1 - np.exp(-x) * sp.special.iv(0, x))
    value = 2 * x * (np.exp(-x) * sp.special.iv(1, x) / z) / (np.log(2 * etha0(a, W) / z))
    return value


def R(a, W):
    x = 4 * (a / W) ** 2
    z = np.where(1 - np.exp(-x) * sp.special.iv(0, x) < 1e-15, 1e-15, 1 - np.exp(-x) * sp.special.iv(0, x))
    return a * (np.log(2 * etha0(a, W) / z)) ** (-1 / lambd(a, W))


def integr(W2, etha, sigm2_bw, a):
    W = W2 ** 0.5
    l = np.where(2 / lambd(a, W) < 1e-15, 1e-15, 2 / lambd(a, W))
    _log = np.log(etha0(a, W) / etha)
    y = np.where(_log <= 0, 0, _log ** (l - 1))
    z = np.where(_log <= 0, 0, _log ** l)

    value = ((R(a, W) ** 2 / (sigm2_bw * lambd(a, W) * etha)) * y * np.exp(-R(a, W) ** 2 / (2 * sigm2_bw) * z) *
             W2_distr.pdf(W ** 2))
    value = np.where(value == np.NaN, 0, value)
    return value


turb = 'strong'

if turb == 'weak':
    mu_W2 = -7.4
    sigm2_W2 = 0.032
    sigm2_bw = 1.2e-5
    a = 0.03
elif turb == 'strong':
    mu_W2 = 0.1898684813819034
    sigm2_W2 = 0.034366742745270046
    sigm2_bw = 0.08279368414532647
    a = 1.2

W2_distr = sp.stats.lognorm(s=np.sqrt(sigm2_W2), scale=np.exp(mu_W2))
range_prec = 0.001
Wleft, Wright = W2_distr.ppf(range_prec), W2_distr.ppf(1 - range_prec)

ro = lambda etha: sp.integrate.quad(integr, Wleft, Wright, args=(etha, sigm2_bw, a), limit=75)[0]
print(ro(0.75))

sum=0
N=200
st=1/N
for i in range(1,N):
   sum=sum+ro(i*st)
print(sum/N)


t = np.linspace(0.00001, 1, num=200)
plt.plot(t, np.vectorize(ro)(t))
plt.show()

# exc = lambda etha2: sp.integrate.dblquad(integr, etha2, 1, Wleft, Wright, args=(sigm2_bw, a, mu_W2, sigm2_W2))[0]

# print(exc(0.001))

# t = np.linspace(0.00001, 1)
# plt.plot(t, np.vectorize(exc)(t))
# plt.show()
