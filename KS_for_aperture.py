import numpy as np
import scipy as sp
from pyatmosphere import gpu
from scipy.stats import rv_continuous
from scipy.ndimage import gaussian_filter1d

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



KS_circSim=[]
KS_circanaly=[]
KS_ancSim=[]
KS_ancanaly=[]
frac=[]
longterm=0.02797 #for good 0.02797, for bad  0.01934
number_dot_KS=4

for i in range(number_dot_KS):
    aperture=0.003+(1.7*longterm-0.003)*i/number_dot_KS
    l = 2000
    g = -15
    Cn2 = 10 ** g
    beam_w0 = (l * 8.08 * 10 ** (-7) / np.pi) ** 0.5
    #beam_w0 = 0.02

    quick_channel = QuickChannel(
        Cn2=Cn2,
        length=l,
        count_ps=6,
        beam_w0=beam_w0,
        beam_wvl=8.08e-07,
        aperture_radius=aperture,
        grid_resolution=512,
        F0=l,
        l0=1e-6,
        L0=5e3,
        f_min=1 / 5e3 / 15,
        f_max=1 / 1e-6 * 2

    )

    # -------------------

    sumx2_0 = 0
    sumW2 = 0
    sumW4 = 0

    etha = []

    sum_etha = 0
    sum_etha2 = 0

    n = 10 ** 2
    for i in range(n):
        output = quick_channel.run(pupil=False)

        W2 = 4 * (measures.mean_x2(quick_channel, output=output) -
                  (measures.mean_x(quick_channel, output=output)) ** 2)
        sumW2 = sumW2 + W2
        sumW4 = sumW4 + W2 ** 2

        sumx2_0 = sumx2_0 + (measures.mean_x(quick_channel, output=output)) ** 2

        # temp = temp + (measures.mean_x2(quick_channel, output=output)) ** 2
        temp_etha = measures.eta(quick_channel, output=quick_channel.run())
        etha.append(temp_etha)
        sum_etha = sum_etha + temp_etha
        sum_etha2 = sum_etha2 + temp_etha ** 2

    sim_W2 = sumW2 / n
    sim_W4 = sumW4 / n
    sim_x2_0 = sumx2_0 / n
    sim_etha = sum_etha / n
    sim_etha2 = sum_etha2 / n
    # ------------------------------
    omega = quick_channel.source.k * quick_channel.source.w0 ** 2 / 2 / quick_channel.path.length
    popravka_ro = 1.457 / 1.5

    analy_x2_0 = (0.32 * quick_channel.source.w0 ** 2 * popravka_ro * quick_channel.get_rythov2() * omega ** (-7 / 6) -
                  0.06 * quick_channel.source.w0 ** 2 * popravka_ro ** 2 * quick_channel.get_rythov2() ** 2 * omega ** (
                              -1 / 3))

    analy_W2 = 4 * (quick_channel.source.w0 ** 2 * omega ** (-2) / 4 +
                    1.07 * quick_channel.source.w0 ** 2 * popravka_ro * quick_channel.get_rythov2() * omega ** (
                                -7 / 6) - analy_x2_0)

    analy_W4 = 16 * (quick_channel.source.w0 ** 4 * omega ** (-4) / 16 +
                     0.58 * quick_channel.source.w0 ** 4 * popravka_ro * quick_channel.get_rythov2() * omega ** (
                                 -19 / 6) +
                     1.37 * quick_channel.source.w0 ** 4 * popravka_ro ** 2 * quick_channel.get_rythov2() ** 2 * omega ** (
                                 -7 / 3) -
                     0.5 * analy_W2 * analy_x2_0 - 3 * analy_x2_0 ** 2)

    print("x2_0 analy=", analy_x2_0, "x2_0 sim=", sim_x2_0)
    print("W2 analy=", analy_W2, "W2 sim=", sim_W2)
    print("W4 analy=", analy_W4, "W4 sim=", sim_W4)
    # --------------------------------------

    """
    initial analytical
    analy_etha=1-np.exp(-2*quick_channel.pupil.radius**2/(analy_W2+4*analy_x2_0))
    analytical with local approx
    """
    th = 0.136 * popravka_ro * quick_channel.get_rythov2() * omega ** (-5 / 6)
    analy_etha = np.exp(-th) * (1 - np.exp(
        -quick_channel.pupil.radius ** 2 * omega ** 2 / quick_channel.source.w0 ** 2 / (0.5 + 5 * th)))

    alph = omega ** (-2) + 3.26 * omega ** (-7 / 6) * popravka_ro * quick_channel.get_rythov2()

    norm = 1
    first_mn = 1 - np.exp(
        -quick_channel.pupil.radius ** 2 * (alph * omega ** 2 + 1) / alph / quick_channel.source.w0 ** 2)
    second_mn = 1 - np.exp(
        -4 * omega ** 2 * quick_channel.pupil.radius ** 2 / quick_channel.source.w0 ** 2 / (alph * omega ** 2 + 1))

    analy_etha2 = norm * first_mn * second_mn

    print("etha analy=", analy_etha, "etha sim=", sim_etha)
    print("etha2 analy=", analy_etha2, "etha2 sim=", sim_etha2)


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


    # --------------------------------------------------------



    res = sp.stats.ecdf(etha)

    number_for_dots_pdt = 200
    # --------------------------------------------------------------

    t = np.linspace(0.00001, 1, num=number_for_dots_pdt)
    # t = np.linspace(0.9, 1, num=number_for_dots_pdt)
    # pdt_circ_sim=np.vectorize(circularPDT(quick_channel.pupil.radius, sim_W2, sim_W4, sim_x2_0))(t)
    circ_sim = circular_beam.CircularBeamModel.from_beam_params(
        S_BW=np.sqrt(sim_x2_0),
        W2_mean=sim_W2,
        W4_mean=sim_W4,
        aperture_radius=quick_channel.pupil.radius
    )
    pdt_circ_sim = circ_sim.get_pdt(t)


    # pdt_circ_analy=np.vectorize(circularPDT(quick_channel.pupil.radius, analy_W2, analy_W4, analy_x2_0))(t)
    circ_analy = circular_beam.CircularBeamModel.from_beam_params(
        S_BW=np.sqrt(analy_x2_0),
        W2_mean=analy_W2,
        W4_mean=analy_W4,
        aperture_radius=quick_channel.pupil.radius
    )
    pdt_circ_analy = circ_analy.get_pdt(t)


    # ----------------------------------------------------------------

    acb_model_sim = circular_beam.AnchoredCircularBeamModel.from_beam_params(
        S_BW=np.sqrt(sim_x2_0),
        eta_mean=sim_etha,
        eta2_mean=sim_etha2,
        aperture_radius=quick_channel.pupil.radius,
        initial_guess_W2_mean=sim_W2,
        initial_guess_W4_mean=sim_W4
    )
    """
    anc_sim_W2=np.exp(acb_model_sim.S_mu+acb_model_sim.S_sigma2/2)
    anc_sim_W4=np.exp(2*acb_model_sim.S_mu+2*acb_model_sim.S_sigma2)

    pdt_anc_sim=np.vectorize(circularPDT(quick_channel.pupil.radius, anc_sim_W2, anc_sim_W4, sim_x2_0))(t)
    """
    pdt_anc_sim = acb_model_sim.get_pdt(t)


    acb_model_analy = circular_beam.AnchoredCircularBeamModel.from_beam_params(
        S_BW=np.sqrt(analy_x2_0),
        eta_mean=analy_etha,
        eta2_mean=analy_etha2,
        aperture_radius=quick_channel.pupil.radius,
        initial_guess_W2_mean=analy_W2,
        initial_guess_W4_mean=analy_W4
    )
    """
    anc_analy_W2=np.exp(acb_model_analy.S_mu+acb_model_analy.S_sigma2/2)
    anc_analy_W4=np.exp(2*acb_model_analy.S_mu+2*acb_model_analy.S_sigma2)

    pdt_anc_analy=np.vectorize(circularPDT(quick_channel.pupil.radius, anc_analy_W2, anc_analy_W4, analy_x2_0))(t)
    """
    pdt_anc_analy = acb_model_analy.get_pdt(t)


    # --------------------------------------------------------------
    max_diff_circsim=0
    max_diff_circanaly = 0
    max_diff_ancsim = 0
    max_diff_ancanaly = 0

    cumul_sim_circ = []
    cumul_analy_circ = []

    cumul_sim_anch = []
    cumul_analy_anch = []

    sum_sim_circ = 0
    sum_analy_circ = 0
    sum_sim_anch = 0
    sum_analy_anch = 0
    count = 0
    for i in t:
        count = count +1
        sum_sim_circ = sum_sim_circ + pdt_circ_sim[count-1]
        sum_analy_circ = sum_analy_circ + pdt_circ_analy[count-1]
        sum_sim_anch = sum_sim_anch + pdt_anc_sim[count-1]
        sum_analy_anch = sum_analy_anch + pdt_anc_analy[count-1]

        cumul_sim_circ.append(sum_sim_circ * (i - 0.00001) / count)
        cumul_analy_circ.append(sum_analy_circ * (i - 0.00001) / count)
        cumul_sim_anch.append(sum_sim_anch * (i - 0.00001) / count)
        cumul_analy_anch.append(sum_analy_anch * (i - 0.00001) / count)


        temp=res.cdf.evaluate(i)
        if abs(cumul_sim_circ[count-1]-temp) > max_diff_circsim:
            max_diff_circsim=abs(cumul_sim_circ[count-1]-temp)

        if abs(cumul_analy_circ[count-1]-temp) > max_diff_circanaly:
            max_diff_circanaly=abs(cumul_analy_circ[count-1]-temp)

        if abs(cumul_sim_anch[count-1] - temp) > max_diff_ancsim:
            max_diff_ancsim = abs(cumul_sim_anch[count-1] - temp)

        if abs(cumul_analy_anch[count-1] - temp) > max_diff_ancanaly:
            max_diff_ancanaly = abs(cumul_analy_anch[count-1] - temp)




    KS_circSim.append(max_diff_circsim)
    KS_circanaly.append(max_diff_circanaly)
    KS_ancSim.append(max_diff_ancsim)
    KS_ancanaly.append(max_diff_ancanaly)
    frac.append(quick_channel.pupil.radius/longterm)



#-------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1)



X_ = np.linspace(np.min(frac), np.max(frac), 200)


X_Y_Spline = sp.interpolate.make_interp_spline(frac, KS_circSim)

Y_ = X_Y_Spline(X_)
Y_=gaussian_filter1d(Y_, 1)

plt.plot(X_,Y_,linewidth='2',color='r')


X_Y1_Spline = sp.interpolate.make_interp_spline(frac, KS_circanaly)

Y1_ = X_Y1_Spline(X_)
Y1_=gaussian_filter1d (Y1_, 1)

plt.plot(X_,Y1_,linewidth='2',color='green')


X_Y2_Spline = sp.interpolate.make_interp_spline(frac, KS_ancSim)

Y2_ = X_Y2_Spline(X_)
Y2_=gaussian_filter1d (Y2_, 3)

plt.plot(X_,Y2_,color='orange',linewidth='2', linestyle='dashed')


X_Y3_Spline = sp.interpolate.make_interp_spline(frac, KS_ancanaly)

Y3_ = X_Y3_Spline(X_)
Y3_=gaussian_filter1d (Y3_, 3)

plt.plot(X_,Y3_,color='violet',linewidth='2', linestyle='dashed')





"""
ax.legend(['directly simulated','circular from sim','circular analytical', 'anchored from sim', 'anchored analytical'])
"""
ax.grid()
ax.set_yscale('log')
ax.set(xlabel=r'Normalized aperture radius $a/W_{LT}$', ylabel='Kolmogorov-Smirnov statistic $D_N$')
plt.savefig("PDT_KS.pdf", **save_kwargs)

#plt.show()





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

