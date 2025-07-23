import numpy as np
import scipy as sp
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

import circular_beam

from pyatmosphere.theory.pdt import beam_wandering_pdt


### QuickChannel example

then = datetime.datetime.now()
from pyatmosphere import QuickChannel, measures


l = 2000
g=-15
Cn2 =  10**g
beam_w0 = (l * 8.08 * 10 ** (-7) / np.pi) ** 0.5


quick_channel = QuickChannel(
    Cn2=Cn2,
    length=l,
    count_ps=6,
    beam_w0=beam_w0,
    beam_wvl=8.08e-07,
    aperture_radius=0.012,
    grid_resolution=512,
    F0=l,
    l0 = 1e-6,
    L0=5e3,
    f_min=1/5e3/15,
    f_max=1/1e-6 *2

)

# -------------------

sumx2_0 = 0
sumW2=0
sumW4=0

etha=[]

sum_etha=0
sum_etha2=0


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


sim_W2 = sumW2 / n
sim_W4 = sumW4 / n
sim_x2_0 = sumx2_0 / n
sim_etha=sum_etha/n
sim_etha2=sum_etha2/n
#------------------------------
omega = quick_channel.source.k * quick_channel.source.w0 ** 2 / 2 / quick_channel.path.length
popravka_ro=1.457/1.5

analy_x2_0 = (0.32 * quick_channel.source.w0 ** 2 * popravka_ro*quick_channel.get_rythov2() * omega ** (-7 / 6) -
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
#-----------------------------------------------------------
determ_losses=10**(-(3+0.1*(quick_channel.path.length/1000))/10)
#--------------------------------------------------------------
etha=[x * determ_losses for x in etha]
ax = sns.histplot(etha, bins=200, kde=False, element="step", stat='density', color='C0')

number_for_dots_pdt=600

#--------------------------------------------------------------

t = np.linspace(0.00001, 1, num=number_for_dots_pdt)
#t = np.linspace(0.00001, 0,8, num=number_for_dots_pdt)
t_another=[x * determ_losses for x in t]


#----------------------------------------------------------------



acb_model_sim_directloss = circular_beam.AnchoredCircularBeamModel.from_beam_params(
    S_BW=np.sqrt(sim_x2_0),
    eta_mean=sim_etha,
    eta2_mean=sim_etha2,
    aperture_radius=quick_channel.pupil.radius,
    initial_guess_W2_mean=sim_W2,
    initial_guess_W4_mean=sim_W4
)

pdt_anc_sim_directloss=acb_model_sim_directloss.get_pdt(t)

pdt_anc_sim_directloss=[x / determ_losses for x in pdt_anc_sim_directloss]
plt.plot(t_another, pdt_anc_sim_directloss, color='orange',linewidth='2', linestyle='dashed')


acb_model_analy_directloss = circular_beam.AnchoredCircularBeamModel.from_beam_params(
    S_BW=np.sqrt(analy_x2_0),
    eta_mean=analy_etha,
    eta2_mean=analy_etha2,
    aperture_radius=quick_channel.pupil.radius,
    initial_guess_W2_mean=analy_W2,
    initial_guess_W4_mean=analy_W4
)

pdt_anc_analy_directloss=acb_model_analy_directloss.get_pdt(t)

pdt_anc_analy_directloss=[x / determ_losses for x in pdt_anc_analy_directloss]
plt.plot(t_another, pdt_anc_analy_directloss, color='violet',linewidth='2', linestyle='dashed')

#------------------------------------------------------------------------------------------------------
acb_model_sim_hiddenloss = circular_beam.AnchoredCircularBeamModel.from_beam_params(
    S_BW=np.sqrt(sim_x2_0),
    eta_mean=sim_etha*determ_losses,
    eta2_mean=sim_etha2*determ_losses**2,
    aperture_radius=quick_channel.pupil.radius,
    initial_guess_W2_mean=sim_W2,
    initial_guess_W4_mean=sim_W4
)

pdt_anc_sim_hiddenloss=acb_model_sim_hiddenloss.get_pdt(t)

plt.plot(t, pdt_anc_sim_hiddenloss, color='#C84C05',linewidth='2', linestyle='dashed')


acb_model_analy_hiddenloss = circular_beam.AnchoredCircularBeamModel.from_beam_params(
    S_BW=np.sqrt(analy_x2_0),
    eta_mean=analy_etha*determ_losses,
    eta2_mean=analy_etha2*determ_losses**2,
    aperture_radius=quick_channel.pupil.radius,
    initial_guess_W2_mean=analy_W2,
    initial_guess_W4_mean=analy_W4
)

pdt_anc_analy_hiddenloss=acb_model_analy_hiddenloss.get_pdt(t)

plt.plot(t, pdt_anc_analy_hiddenloss, color='#69247C',linewidth='2', linestyle='dashed')


#------------------------------------------------------------------------------------------------------


ax.grid()
ax.set(xlabel=r'Transmittance $\eta$', ylabel='Probability density')
ax.set_xlim(left=0, right=0.35)
#ax.set_xlim(left=0, right=1)
plt.savefig("etha_distrib.pdf", **save_kwargs)

#plt.show()


"""
fig, ax = plt.subplots(1, 1)


res = sp.stats.ecdf(etha)
res.cdf.plot(ax, color='C0',linewidth='2')


cumul_sim_anc_direct=[]
cumul_analy_anc_direct=[]

cumul_sim_anc_hidden=[]
cumul_analy_anc_hidden=[]

spec_t=t[1::2]
spec_pdt_circ_sim=pdt_circ_sim[1::2]
spec_pdt_circ_analy=pdt_circ_analy[1::2]
spec_pdt_anc_sim=pdt_anc_sim[1::2]
spec_pdt_anc_analy=pdt_anc_analy[1::2]


sum_sim_anc_direct=0
sum_analy_anc_direct=0
sum_sim_anc_hidden=0
sum_analy_anc_hidden=0
count=0
for i in spec_t:
    count=count+1
    sum_sim_anc_direct= sum_sim_anc_direct+spec_pdt_circ_sim[count-1]
    sum_analy_anc_direct=sum_analy_anc_direct+spec_pdt_circ_analy[count-1]
    sum_sim_anc_hidden=sum_sim_anc_hidden+spec_pdt_anc_sim[count-1]
    sum_analy_anc_hidden=sum_analy_anc_hidden+spec_pdt_anc_analy[count-1]

    cumul_sim_anc_direct.append(sum_sim_anc_direct * (i  - 0.00001) / count)
    cumul_analy_anc_direct.append(sum_analy_anc_direct * (i  - 0.00001) / count)
    cumul_sim_anc_hidden.append(sum_sim_anc_hidden * (i  - 0.00001) / count)
    cumul_analy_anc_hidden.append(sum_analy_anc_hidden * (i  - 0.00001) / count)






ax.step(spec_t, cumul_sim_anc_direct, 'r', linewidth='2')
ax.step(spec_t, cumul_analy_anc_direct, 'green', linewidth='2')
ax.step(spec_t, cumul_sim_anc_hidden, 'orange', linestyle='dashed', linewidth='2')
ax.step(spec_t, cumul_analy_anc_hidden, 'violet', linestyle='dashed', linewidth='2')




ax.grid()
ax.set(xlabel=r'Transmittance $\eta$', ylabel='Cumulative distribution')
#ax.set_xlim(left=0.8, right=1)
ax.set_xlim(left=0, right=1)
plt.savefig("etha_cumul_distrib.pdf", **save_kwargs)

"""

"""
beam_result = simulations.BeamResult(quick_channel)
pdt_result = simulations.PDTResult(quick_channel)
sim = simulations.Simulation([beam_result, pdt_result])
sim.run(plot_step=100)
"""

# ------------------
now = datetime.datetime.now()
delta = now - then
print(delta.seconds / 60, "min")


