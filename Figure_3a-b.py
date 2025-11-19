import numpy as np
import scipy as sp
from pyatmosphere import gpu
import seaborn as sns
import circular_beam
import matplotlib.pyplot as plt
from pyatmosphere import QuickChannel, measures
import analytics

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







l = 1000
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
    aperture_radius=0.025,
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

    
    temp_etha=measures.eta(quick_channel, output=quick_channel.run())
    etha.append(temp_etha)
    sum_etha=sum_etha+temp_etha
    sum_etha2 = sum_etha2 + temp_etha**2


sim_W2 = sumW2 / n
sim_W4 = sumW4 / n
sim_x2_0 = sumx2_0 / n
sim_etha=sum_etha/n
sim_etha2=sum_etha2/n
# ------------------------------

analy_x2_0 = analytics.get_x2_0(quick_channel.source.k,quick_channel.source.w0,quick_channel.path.length ,quick_channel.get_rythov2())

analy_W2 = analytics.get_W2(quick_channel.source.k,quick_channel.source.w0,quick_channel.path.length ,quick_channel.get_rythov2())

analy_W4 = analytics.get_W4(quick_channel.source.k,quick_channel.source.w0,quick_channel.path.length ,quick_channel.get_rythov2())

# --------------------------------------

analy_etha = analytics.get_etha(quick_channel.source.k,quick_channel.source.w0,quick_channel.path.length ,quick_channel.get_rythov2(), quick_channel.pupil.radius)

analy_etha2 = analytics.get_etha2(quick_channel.source.k,quick_channel.source.w0,quick_channel.path.length ,quick_channel.get_rythov2(), quick_channel.pupil.radius)

#-----------------------------------------------------------
ax = sns.histplot(etha, bins=200, kde=False, element="step", stat='density', color='C0')

number_for_dots_pdt=200

#--------------------------------------------------------------

t = np.linspace(0.00001, 1, num=number_for_dots_pdt)

circ_sim=circular_beam.CircularBeamModel.from_beam_params(
    S_BW=np.sqrt(sim_x2_0),
    W2_mean=sim_W2,
    W4_mean=sim_W4,
    aperture_radius=quick_channel.pupil.radius
)
pdt_circ_sim=circ_sim.get_pdt(t)
plt.plot(t, pdt_circ_sim, color='r',linewidth='2')




circ_analy=circular_beam.CircularBeamModel.from_beam_params(
    S_BW=np.sqrt(analy_x2_0),
    W2_mean=analy_W2,
    W4_mean=analy_W4,
    aperture_radius=quick_channel.pupil.radius
)
pdt_circ_analy=circ_analy.get_pdt(t)
plt.plot(t, pdt_circ_analy, color='green',linewidth='2')

#----------------------------------------------------------------



acb_model_sim = circular_beam.AnchoredCircularBeamModel.from_beam_params(
    S_BW=np.sqrt(sim_x2_0),
    eta_mean=sim_etha,
    eta2_mean=sim_etha2,
    aperture_radius=quick_channel.pupil.radius,
    initial_guess_W2_mean=sim_W2,
    initial_guess_W4_mean=sim_W4
)

pdt_anc_sim=acb_model_sim.get_pdt(t)
plt.plot(t, pdt_anc_sim, color='orange',linewidth='2', linestyle='dashed')



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




ax.grid()
ax.set(xlabel=r'Transmittance $\eta$', ylabel='Probability density')
#ax.set_xlim(left=0.9, right=1)
ax.set_xlim(left=0, right=1)
plt.savefig("etha_distrib.pdf", **save_kwargs)






