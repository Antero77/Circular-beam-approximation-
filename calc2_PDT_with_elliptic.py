import numpy as np
import scipy as sp
from pyatmosphere import gpu

gpu.config['use_gpu'] = True

import matplotlib.pyplot as plt

from pyatmosphere.theory.pdt.elliptic_beam import EllipticBeamAnalyticalPDT



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

import circular_beam




### QuickChannel example

then = datetime.datetime.now()
from pyatmosphere import QuickChannel, measures


l = 2000
g=-15
Cn2 =  10**g
beam_w0 = (l * 8.08 * 10 ** (-7) / np.pi) ** 0.5
#beam_w0 = 0.02

quick_channel = QuickChannel(
    Cn2=Cn2,
    length=l,
    count_ps=6,
    beam_w0=beam_w0,
    beam_wvl=8.08e-07,
    aperture_radius=0.012,
    grid_resolution=512,
    F0=l,
    l0=1e-6,
    L0=5e3,
    f_min=1 / 5e3 / 15,
    f_max=1 / 1e-6 * 2
)

# -------------------

sumx2_0 = 0
sumW2=0
sumW4=0

etha=[]

sum_etha=0
sum_etha2=0

ellipt_mean_x=[]
ellipt_mean_x2=[]
ellipt_mean_y2=[]



n = 10 ** 5
for i in range(n):
    output = quick_channel.run(pupil=False)

    W2 = 4 * (measures.mean_x2(quick_channel, output=output) -
              (measures.mean_x(quick_channel, output=output)) ** 2)
    sumW2 = sumW2 + W2
    sumW4 = sumW4 + W2 ** 2

    sumx2_0 = sumx2_0 + (measures.mean_x(quick_channel, output=output)) ** 2

    # temp = temp + (measures.mean_x2(quick_channel, output=output)) ** 2
    temp_etha=measures.eta(quick_channel, output=quick_channel.run())
    etha.append(temp_etha)
    sum_etha=sum_etha+temp_etha
    sum_etha2 = sum_etha2 + temp_etha**2

    ellipt_mean_x.append(measures.mean_x(quick_channel, output=output))
    ellipt_mean_x2.append(measures.mean_x2(quick_channel, output=output))
    ellipt_mean_y2.append(measures.mean_y2(quick_channel, output=output))


sim_W2 = sumW2 / n
sim_W4 = sumW4 / n
sim_x2_0 = sumx2_0 / n
sim_etha=sum_etha/n
sim_etha2=sum_etha2/n
#------------------------------
omega = quick_channel.source.k * quick_channel.source.w0 ** 2 / 2 / quick_channel.path.length
popravka_ro=1.457/1.5

analy_x2_0 = (0.32 * quick_channel.source.w0 ** 2 * popravka_ro*quick_channel.get_rythov2() * omega ** (7 / 6) -
                    0.06*quick_channel.source.w0 ** 2 * popravka_ro**2*quick_channel.get_rythov2()**2 * omega ** (-1 / 3))



analy_W2 = 4*(quick_channel.source.w0 ** 2 * omega ** (-2)/4 +
                1.07*quick_channel.source.w0 ** 2 * popravka_ro*quick_channel.get_rythov2() * omega ** (-7 / 6)- analy_x2_0 )



analy_W4 = 16* (quick_channel.source.w0 ** 4 * omega ** (-4) /16 +
               0.58 * quick_channel.source.w0 ** 4 * popravka_ro*quick_channel.get_rythov2() * omega ** (-19 / 6) +
               1.37 * quick_channel.source.w0 ** 4 * popravka_ro**2*quick_channel.get_rythov2() ** 2 * omega ** (-7 / 3)-
               0.5 * analy_W2 * analy_x2_0 - 3 * analy_x2_0 ** 2)


print("x2_0 analy=",analy_x2_0, "x2_0 sim=", sim_x2_0)
print("W2 analy=",analy_W2, "W2 sim=", sim_W2)
print("W4 analy=",analy_W4, "W4 sim=", sim_W4)
# --------------------------------------

"""
initial analytical
analy_etha=1-np.exp(-2*quick_channel.pupil.radius**2/(analy_W2+4*analy_x2_0))
analytical with local approx
"""
th=0.136*popravka_ro*quick_channel.get_rythov2()*omega**(-5/6)
analy_etha=np.exp(-th)*(1-np.exp(-quick_channel.pupil.radius**2*omega**2/quick_channel.source.w0**2/(0.5+5*th)))



alph=omega**(-2)+3.26*omega**(-7/6)*popravka_ro*quick_channel.get_rythov2()

norm=1
first_mn=1-np.exp(-quick_channel.pupil.radius**2*(alph*omega**2+1)/alph/quick_channel.source.w0**2)
second_mn=1-np.exp(-4*omega**2*quick_channel.pupil.radius**2/quick_channel.source.w0**2/(alph*omega**2+1))

analy_etha2=norm*first_mn*second_mn

print("etha analy=",analy_etha, "etha sim=", sim_etha)
print("etha2 analy=",analy_etha2, "etha2 sim=", sim_etha2)

# --------------------------------------


def circularPDT(a, meanW2, meanW4, meanx2_0):

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

    mu = np.log((meanW2 ** 2) / (meanW4 ** 0.5))
    sigma2 = np.log(meanW4 / (meanW2 ** 2))

    W2_distr = sp.stats.lognorm(s=np.sqrt(sigma2), scale=np.exp(mu))
    range_prec = 0.001
    Wleft, Wright = W2_distr.ppf(range_prec), W2_distr.ppf(1 - range_prec)


    return lambda etha: sp.integrate.quad(integr, Wleft, Wright, args=(etha, meanx2_0, a), limit=75)[0]


#--------------------------------------------------------



#--------------------------------------------------------------
ax = sns.histplot(etha, bins=200, kde=False, element="step", stat='density', color='C0')

number_for_dots_pdt=200

#--------------------------------------------------------------

t = np.linspace(0.00001, 1, num=number_for_dots_pdt)
#t = np.linspace(0.9, 1, num=number_for_dots_pdt)


acb_model_analy = circular_beam.AnchoredCircularBeamModel.from_beam_params(
    S_BW=np.sqrt(analy_x2_0),
    eta_mean=analy_etha,
    eta2_mean=analy_etha2,
    aperture_radius=quick_channel.pupil.radius,
    initial_guess_W2_mean=analy_W2,
    initial_guess_W4_mean=analy_W4
)

pdt_anc_analy=acb_model_analy.get_pdt(t)
plt.plot(t, pdt_anc_analy, color='violet',linewidth='2', linestyle='dashed')
#-----------------------------------------------------------------------------

N_ITERS = 10**4
eb_model = EllipticBeamAnalyticalPDT(W0=quick_channel.source.w0, a=quick_channel.pupil.radius, size=N_ITERS)
eb_model.set_params_from_data(np.array(ellipt_mean_x), np.array(ellipt_mean_x2), np.array(ellipt_mean_y2))

errors=0
transmittance = eb_model.pdt()
transmittance_cleared=[]
for x in transmittance:
    if not (np.isnan(x) or np.isinf(x)):
        transmittance_cleared.append(x)
        errors=errors+1


print(errors)

kde_sim = sp.stats.gaussian_kde(transmittance_cleared)
pdt_etha = kde_sim.pdf(t)

plt.plot(t, pdt_etha, color='brown',linewidth='2', linestyle='solid')

#------------------------------------------------------------------------------------
"""
eb_model2 = EllipticBeamAnalyticalPDT(W0=quick_channel.source.w0, a=quick_channel.pupil.radius, size=N_ITERS)

elliptic_bw=0.33*quick_channel.source.w0**2*quick_channel.get_rythov2()*omega**(7/6)
elliptic_theta_mean=np.log( (1+2.96*quick_channel.get_rythov2()*omega**(5/6))**2 / omega**2/((1+2.96*quick_channel.get_rythov2()*omega**(5/6))**2+1.2**quick_channel.get_rythov2()*omega**(5/6))**0.5 )

elliptic_disp=np.log(1+1.2*quick_channel.get_rythov2()*omega**(5/6)/(1+2.96*quick_channel.get_rythov2()*omega**(5/6))**2)
elliptic_corr=np.log(1-0.8*quick_channel.get_rythov2()*omega**(5/6)/(1+2.96*quick_channel.get_rythov2()*omega**(5/6))**2)


eb_model2.set_params(elliptic_bw, elliptic_theta_mean, [elliptic_disp, elliptic_corr])
transmittance2 = eb_model2.pdt()

transmittance_cleared2=[]
for x in transmittance2:
    if not (np.isnan(x) or np.isinf(x)):
        transmittance_cleared2.append(x)



kde_sim2 = sp.stats.gaussian_kde(transmittance_cleared2)
pdt_etha2 = kde_sim2.pdf(t)

plt.plot(t, pdt_etha2, color='purple',linewidth='2', linestyle='solid')
"""
#-----------------------------------------------------------------------------


ax.grid()
ax.set(xlabel=r'Transmittance $\eta$', ylabel='Probability density')
#ax.set_xlim(left=0.9, right=1)
ax.set_xlim(left=0, right=1)
plt.savefig("etha_distrib.pdf", **save_kwargs)

#plt.show()

# ------------------
now = datetime.datetime.now()
delta = now - then
print(delta.seconds / 60, "min")


