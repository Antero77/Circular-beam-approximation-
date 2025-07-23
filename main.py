import numpy as np
from pyatmosphere import gpu

gpu.config['use_gpu'] = True

import matplotlib.pyplot as plt

plt.rcParams['axes.axisbelow'] = True
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = "STIX"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 15})

save_kwargs = {
        "format": "pdf",
        "dpi": 300,
        "bbox_inches": "tight",
        "pad_inches": 0.005
        }


import seaborn as sns
import datetime
import scipy as sp





### QuickChannel example

then = datetime.datetime.now()
from pyatmosphere import QuickChannel, measures

l = 1000
g=-15
Cn2 =  10**g
beam_w0 = (l * 8.08 * 10 ** (-7) / np.pi) ** 0.5

quick_channel = QuickChannel(
    Cn2=Cn2,
    length=l,
    count_ps=6,
    beam_w0=beam_w0,
    beam_wvl=8.08e-07,
    aperture_radius=0.011,
    grid_resolution=512,
    F0=l,
    l0 = 1e-6,
    L0=5e3,
    f_min=1/5e3/15,
    f_max=1/1e-6 *2

)

#quick_channel.plot(pupil=False)
##print(measures.I(quick_channel, pupil=False))
#plt.show()
# -------------------

dataW2 = []
sumW2 = 0
sumW4 = 0
n = 10 ** 5
for i in range(n):
    output = quick_channel.run(pupil=False)

    W2 = 4 * (measures.mean_x2(quick_channel, output=output) -
              (measures.mean_x(quick_channel, output=output)) ** 2)
    #--------------------------
    W2=W2*10**4 # into cm^2
    #--------------------------
    sumW2 = sumW2 + W2
    sumW4 = sumW4 + W2 ** 2
    dataW2.append(W2)

# ------------
meanW2 = sumW2 / n
meanW4 = sumW4 / n

mu = np.log((meanW2 ** 2) / (meanW4 ** 0.5))
sigma2 = np.log(meanW4 / (meanW2 ** 2))

print('mean for lognormal', mu)
print('variance for lognormal', sigma2)


# -------------------


"""

ax = sns.histplot(dataW2, bins=100, kde=False, element="step", stat='density')


x = np.linspace(np.min(dataW2), np.max(dataW2))
pdt = sp.stats.lognorm(s=np.sqrt(sigma2), scale=np.exp(mu))
plt.plot(x, pdt.pdf(x), color='red')

ax.set(xlabel=r'Squared beam-spot radius S, $(m^2)$', ylabel='Probability density')
"""
# ------------------

fig, ax = plt.subplots(1, 1)
x = np.linspace(np.min(dataW2), np.max(dataW2),200)
kde_sim = sp.stats.gaussian_kde(dataW2)
plt.plot(x, kde_sim.pdf(x), color='blue', linewidth='2',linestyle='--')



pdt = sp.stats.lognorm(s=np.sqrt(sigma2), scale=np.exp(mu))
plt.plot(x, pdt.pdf(x), color='red',linewidth='2')

ax.set(xlabel=r'Squared beam-spot radius S $(cm^2)$', ylabel='Probability density P(S)')

# ------------------




ax.grid()


plt.savefig("s_ditrib.pdf", **save_kwargs)

plt.show()
now = datetime.datetime.now()
delta = now - then
print(delta.seconds / 60)