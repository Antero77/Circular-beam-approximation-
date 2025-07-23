import numpy as np
import scipy as sp
from pyatmosphere import gpu
from pyatmosphere import simulations
from pyatmosphere.theory.pdt import EllipticBeamAnalyticalPDT

gpu.config['use_gpu'] = True

import matplotlib.pyplot as plt
import seaborn as sns
import datetime


from scipy.interpolate import make_interp_spline
### QuickChannel example

then = datetime.datetime.now()
from pyatmosphere import QuickChannel, measures
from pyatmosphere.theory.pdt.pdt import bw_pdt, beam_wandering_pdt
from pyatmosphere.simulations import PDTResult
from pyatmosphere.theory.pdt.elliptic_beam import EllipticBeamAnalyticalPDT

eta_avr=[]
eta2_avr=[]
axis=[]
m=30
for z in range(m):
    l=500+(10000-500)*z/m
    Cn2 = 5 * 10 ** (-14)
    # beam_w0 = (l * 8.08 * 10 ** (-7) / np.pi) ** 0.5
    beam_w0 = (l * 8.08 * 10 ** (-7) / np.pi) ** 0.5
    quick_channel = QuickChannel(
        Cn2=Cn2,
        length=l,
        count_ps=6,
        beam_w0=beam_w0,
        beam_wvl=8.08e-07,
        aperture_radius=0.015,
        F0=np.inf
    )

    """
    quick_channel.plot(pupil=False)
    ##print(measures.I(quick_channel, pupil=False))
    plt.show()
    quick_channel.plot()
    plt.show()
    """
    # -------------------

    dataW2 = []
    sumW2 = 0
    sumW4 = 0
    sumx2_0 = 0

    n = 10 ** 3
    for i in range(n):
        output = quick_channel.run(pupil=False)

        W2 = 4 * (measures.mean_x2(quick_channel, output=output) -
                  (measures.mean_x(quick_channel, output=output)) ** 2)
        sumW2 = sumW2 + W2
        sumW4 = sumW4 + W2 ** 2
        sumx2_0 = sumx2_0 + (measures.mean_x(quick_channel, output=output)) ** 2
        dataW2.append(W2)

    # ------------
    meanW2 = sumW2 / n
    meanW4 = sumW4 / n
    meanx2_0 = sumx2_0 / n

    mu = np.log((meanW2 ** 2) / (meanW4 ** 0.5))
    sigma2 = np.log(meanW4 / (meanW2 ** 2))


    # --------------------------------------
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


    print(mu, sigma2, meanx2_0, meanW2)

    sigm2_bw = meanx2_0
    a = quick_channel.pupil.radius

    W2_distr = sp.stats.lognorm(s=np.sqrt(sigma2), scale=np.exp(mu))
    range_prec = 0.001
    Wleft, Wright = W2_distr.ppf(range_prec), W2_distr.ppf(1 - range_prec)

    ro = lambda etha: sp.integrate.quad(integr, Wleft, Wright, args=(etha, sigm2_bw, a), limit=75)[0]

    # -----------------------

    sum = 0
    sum2=0
    N = 800
    st = 1 / N
    for k in range(1, N):
        sum = sum + (k*st)*ro(k * st)
        sum2= sum2+(k*st)**2*ro(k*st)
    eta_avr.append(sum/N)
    eta2_avr.append(sum2/N)

    axis.append(l)

print(eta_avr, eta2_avr)
eta_avr.pop()
eta2_avr.pop()
axis.pop()
Q2= [((2 - 1) * b - 2 * a ** 2) / a for a, b in zip(eta_avr, eta2_avr)]
Q5= [((5 - 1) * b - 5 * a ** 2) / a for a, b in zip(eta_avr, eta2_avr)]
Q10= [((10 - 1) * b - 10 * a ** 2) / a for a, b in zip(eta_avr, eta2_avr)]
Q15= [((15 - 1) * b - 15 * a ** 2) / a for a, b in zip(eta_avr, eta2_avr)]


X_Y_Spline2 = make_interp_spline(axis, Q2)
X_Y_Spline5 = make_interp_spline(axis, Q5)
X_Y_Spline10 = make_interp_spline(axis, Q10)
X_Y_Spline15 = make_interp_spline(axis, Q15)


newaxis = np.linspace(axis[0], axis[m-2], 200)
newQ2 = X_Y_Spline2(newaxis)
newQ5 = X_Y_Spline5(newaxis)
newQ10 = X_Y_Spline10(newaxis)
newQ15 = X_Y_Spline15(newaxis)




fig, ax = plt.subplots()
ax.plot(newaxis, newQ2, 'blue', newaxis, newQ5, 'green', newaxis, newQ10, 'orange', newaxis, newQ15, 'red')
ax.set(xlabel='довжина каналу l, m', ylabel='Q-параметр')
ax.legend(['n=2','n=5', 'n=10','n=15'])
plt.axhline(y = 0, color = 'gray', linestyle = '--')

plt.savefig("Q_symbol.pdf", format='pdf')

plt.show()







# ------------------
now = datetime.datetime.now()
delta = now - then
print(delta.seconds / 60)

plt.show()
