import numpy as np
import scipy as sp
from pyatmosphere import gpu
from scipy.stats import rv_continuous
from scipy.ndimage import gaussian_filter1d
import circular_beam
from pyatmosphere import QuickChannel, measures
import matplotlib.pyplot as plt
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










KS_circSim=[]
KS_circanaly=[]
KS_ancSim=[]
KS_ancanaly=[]
frac=[]
longterm=0.02797 
number_dot_KS=20

for i in range(number_dot_KS):
    aperture=0.003+(1.7*longterm-0.003)*i/number_dot_KS
    l = 1000
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

    n = 10 ** 5
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

    analy_x2_0 = analytics.get_x2_0(quick_channel.source.k,quick_channel.source.w0,quick_channel.path.length ,quick_channel.get_rythov2())

    analy_W2 = analytics.get_W2(quick_channel.source.k,quick_channel.source.w0,quick_channel.path.length ,quick_channel.get_rythov2())

    analy_W4 = analytics.get_W4(quick_channel.source.k,quick_channel.source.w0,quick_channel.path.length ,quick_channel.get_rythov2())

    # --------------------------------------

    analy_etha = analytics.get_etha(quick_channel.source.k,quick_channel.source.w0,quick_channel.path.length ,quick_channel.get_rythov2(), quick_channel.pupil.radius)

    analy_etha2 = analytics.get_etha2(quick_channel.source.k,quick_channel.source.w0,quick_channel.path.length ,quick_channel.get_rythov2(), quick_channel.pupil.radius)

    #-----------------------------------------------------------

    res = sp.stats.ecdf(etha)

    number_for_dots_pdt = 200
    # --------------------------------------------------------------

    t = np.linspace(0.00001, 1, num=number_for_dots_pdt)

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

    pdt_anc_sim = acb_model_sim.get_pdt(t)


    acb_model_analy = circular_beam.AnchoredCircularBeamModel.from_beam_params(
        S_BW=np.sqrt(analy_x2_0),
        eta_mean=analy_etha,
        eta2_mean=analy_etha2,
        aperture_radius=quick_channel.pupil.radius,
        initial_guess_W2_mean=analy_W2,
        initial_guess_W4_mean=analy_W4
    )

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






ax.grid()
ax.set_yscale('log')
ax.set(xlabel=r'Normalized aperture radius $a/W_{LT}$', ylabel='Kolmogorov-Smirnov statistic $D_N$')
plt.savefig("PDT_KS.pdf", **save_kwargs)
