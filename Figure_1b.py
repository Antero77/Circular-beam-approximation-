import numpy as np
from pyatmosphere import gpu
import matplotlib.pyplot as plt
import scipy as sp
from pyatmosphere import QuickChannel, measures


gpu.config['use_gpu'] = True



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






y_ax=[]
x_ax=[]
number_dot=10
for i in range(number_dot):

    l = 500+(5500-500)*i/number_dot
    g = -15
    Cn2 = 10 ** g
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
        l0=1e-6,
        L0=5e3,
        f_min=1 / 5e3 / 15,
        f_max=1 / 1e-6 * 2

    )


    # -------------------

    dataW2 = []
    sumW2 = 0
    sumW4 = 0
    n = 10 ** 5
    for i in range(n):
        output = quick_channel.run(pupil=False)

        W2 = 4 * (measures.mean_x2(quick_channel, output=output) -
                  (measures.mean_x(quick_channel, output=output)) ** 2)
        x2_0=measures.mean_x(quick_channel, output=output)**2

        sumW2 = sumW2 + W2
        sumW4 = sumW4 + W2 ** 2
        dataW2.append(W2)

    # ------------
    meanW2 = sumW2 / n
    meanW4 = sumW4 / n

    mu = np.log((meanW2 ** 2) / (meanW4 ** 0.5))
    sigma2 = np.log(meanW4 / (meanW2 ** 2))

    #print('mean for lognormal', mu)
    #print('variance for lognormal', sigma2)

    # -------------------

    x = np.linspace(np.min(dataW2), np.max(dataW2))
    pdt = sp.stats.lognorm(s=np.sqrt(sigma2), scale=np.exp(mu))
    dataW2_analy = pdt.rvs(size=n)

    y_ax.append(sp.stats.ks_2samp(dataW2, dataW2_analy)[0])
    x_ax.append(quick_channel.get_rythov2())




#-----------------------------

X_Y_Spline = sp.interpolate.make_interp_spline(x_ax, y_ax)

X_ = np.linspace(np.min(x_ax), np.max(x_ax), 200)
Y_ = X_Y_Spline(X_)



fig, ax = plt.subplots(1, 1)
plt.plot(X_,Y_,linewidth='2')
plt.plot(x_ax,y_ax,'go')
ax.set(xlabel=r'Rytov parameter $σ^2_R$', ylabel=r'Kolmogorov-Smirnov statistic')
ax.set_yscale('log')
#plt.ylim([5*10**(-4),10**(0)])
ax.grid()


plt.savefig("s_Kolmog_Smirn.pdf", **save_kwargs)

plt.show()


