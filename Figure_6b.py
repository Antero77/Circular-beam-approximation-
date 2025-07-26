import numpy as np
import scipy as sp
from pyatmosphere import gpu
import circular_beam
from pyatmosphere import QuickChannel, measures
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

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






KS_ancSim_direct=[]
KS_ancanaly_direct=[]
KS_ancSim_hidden=[]
KS_ancanaly_hidden=[]

frac=[]
longterm=0.02797
number_dot_KS=20



for i in range(number_dot_KS):
    aperture=0.003+(1.7*longterm-0.003)*i/number_dot_KS
    l = 2000
    g = -15
    Cn2 = 10 ** g
    beam_w0 = (l * 8.08 * 10 ** (-7) / np.pi) ** 0.5

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


    # --------------------------------------

    #local approximation can be used
    # th = 0.136 * popravka_ro * quick_channel.get_rythov2() * omega ** (-5 / 6)
    th = 0 
    analy_etha = np.exp(-th) * (1 - np.exp(
        -quick_channel.pupil.radius ** 2 * omega ** 2 / quick_channel.source.w0 ** 2 / (0.5 + 5 * th)))

    alph = omega ** (-2) + 3.26 * omega ** (-7 / 6) * popravka_ro * quick_channel.get_rythov2()

    norm = 1
    first_mn = 1 - np.exp(
        -quick_channel.pupil.radius ** 2 * (alph * omega ** 2 + 1) / alph / quick_channel.source.w0 ** 2)
    second_mn = 1 - np.exp(
        -4 * omega ** 2 * quick_channel.pupil.radius ** 2 / quick_channel.source.w0 ** 2 / (alph * omega ** 2 + 1))

    analy_etha2 = norm * first_mn * second_mn

    # -----------------------------------------------------------
    determ_losses = 10 ** (-(3 + 0.1 * (quick_channel.path.length / 1000)) / 10)
    # --------------------------------------------------------------
    etha = [x * determ_losses for x in etha]
    res = sp.stats.ecdf(etha)

    number_for_dots_pdt = 200

    # --------------------------------------------------------------

    t = np.linspace(0.00001, 1, num=number_for_dots_pdt)
    t_another = [x * determ_losses for x in t]

    # ----------------------------------------------------------------

    acb_model_sim_directloss = circular_beam.AnchoredCircularBeamModel.from_beam_params(
        S_BW=np.sqrt(sim_x2_0),
        eta_mean=sim_etha,
        eta2_mean=sim_etha2,
        aperture_radius=quick_channel.pupil.radius,
        initial_guess_W2_mean=sim_W2,
        initial_guess_W4_mean=sim_W4
    )

    pdt_anc_sim_directloss = acb_model_sim_directloss.get_pdt(t)

    pdt_anc_sim_directloss = [x / determ_losses for x in pdt_anc_sim_directloss]

    acb_model_analy_directloss = circular_beam.AnchoredCircularBeamModel.from_beam_params(
        S_BW=np.sqrt(analy_x2_0),
        eta_mean=analy_etha,
        eta2_mean=analy_etha2,
        aperture_radius=quick_channel.pupil.radius,
        initial_guess_W2_mean=analy_W2,
        initial_guess_W4_mean=analy_W4
    )

    pdt_anc_analy_directloss = acb_model_analy_directloss.get_pdt(t)

    pdt_anc_analy_directloss = [x / determ_losses for x in pdt_anc_analy_directloss]


    # ------------------------------------------------------------------------------------------------------
    acb_model_sim_hiddenloss = circular_beam.AnchoredCircularBeamModel.from_beam_params(
        S_BW=np.sqrt(sim_x2_0),
        eta_mean=sim_etha * determ_losses,
        eta2_mean=sim_etha2 * determ_losses ** 2,
        aperture_radius=quick_channel.pupil.radius,
        initial_guess_W2_mean=sim_W2,
        initial_guess_W4_mean=sim_W4
    )

    pdt_anc_sim_hiddenloss = acb_model_sim_hiddenloss.get_pdt(t)



    acb_model_analy_hiddenloss = circular_beam.AnchoredCircularBeamModel.from_beam_params(
        S_BW=np.sqrt(analy_x2_0),
        eta_mean=analy_etha * determ_losses,
        eta2_mean=analy_etha2 * determ_losses ** 2,
        aperture_radius=quick_channel.pupil.radius,
        initial_guess_W2_mean=analy_W2,
        initial_guess_W4_mean=analy_W4
    )

    pdt_anc_analy_hiddenloss = acb_model_analy_hiddenloss.get_pdt(t)



    # --------------------------------------------------------------
    max_diff_ancsim_direct=0
    max_diff_ancanaly_direct = 0
    max_diff_ancsim_hidden = 0
    max_diff_ancanaly_hidden = 0

    cumul_sim_anc_direct = []
    cumul_analy_anc_direct = []

    cumul_sim_anc_hidden = []
    cumul_analy_anc_hidden = []

    sum_sim_anc_direct = 0
    sum_analy_anc_direct = 0
    sum_sim_anc_hidden = 0
    sum_analy_anc_hidden = 0
    count = 0
    for i in t:
        count = count +1
        sum_sim_anc_direct = sum_sim_anc_direct + pdt_anc_sim_directloss[count - 1]
        sum_analy_anc_direct = sum_analy_anc_direct + pdt_anc_analy_directloss[count - 1]
        sum_sim_anc_hidden = sum_sim_anc_hidden + pdt_anc_sim_hiddenloss[count - 1]
        sum_analy_anc_hidden = sum_analy_anc_hidden + pdt_anc_analy_hiddenloss[count - 1]

        cumul_sim_anc_direct.append(sum_sim_anc_direct * (i - 0.00001) * determ_losses / count)
        cumul_analy_anc_direct.append(sum_analy_anc_direct * (i - 0.00001) *determ_losses / count)
        cumul_sim_anc_hidden.append(sum_sim_anc_hidden * (i - 0.00001) / count)
        cumul_analy_anc_hidden.append(sum_analy_anc_hidden * (i - 0.00001) / count)


        temp=res.cdf.evaluate(i)
        temp_another=res.cdf.evaluate(t_another[count-1])
        if abs(cumul_sim_anc_direct[count - 1] - temp_another) > max_diff_ancsim_direct:
            max_diff_ancsim_direct=abs(cumul_sim_anc_direct[count - 1] - temp_another)

        if abs(cumul_analy_anc_direct[count - 1] - temp_another) > max_diff_ancanaly_direct:
            max_diff_ancanaly_direct=abs(cumul_analy_anc_direct[count - 1] - temp_another)

        if abs(cumul_sim_anc_hidden[count - 1] - temp) > max_diff_ancsim_hidden:
            max_diff_ancsim_hidden = abs(cumul_sim_anc_hidden[count - 1] - temp)

        if abs(cumul_analy_anc_hidden[count - 1] - temp) > max_diff_ancanaly_hidden:
            max_diff_ancanaly_hidden = abs(cumul_analy_anc_hidden[count - 1] - temp)




    KS_ancSim_direct.append(max_diff_ancsim_direct)
    KS_ancanaly_direct.append(max_diff_ancanaly_direct)
    KS_ancSim_hidden.append(max_diff_ancsim_hidden)
    KS_ancanaly_hidden.append(max_diff_ancanaly_hidden)
    frac.append(quick_channel.pupil.radius/longterm)



#-------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1)
X_ = np.linspace(np.min(frac), np.max(frac), 200)


X_Y_Spline = sp.interpolate.make_interp_spline(frac, KS_ancSim_direct)

Y_ = X_Y_Spline(X_)
Y_=gaussian_filter1d(Y_, 1)
plt.plot(X_,Y_,color='orange',linewidth='2', linestyle='dashed')


X_Y1_Spline = sp.interpolate.make_interp_spline(frac, KS_ancanaly_direct)

Y1_ = X_Y1_Spline(X_)
Y1_=gaussian_filter1d (Y1_, 1)
plt.plot(X_,Y1_,color='violet',linewidth='2', linestyle='dashed')


X_Y2_Spline = sp.interpolate.make_interp_spline(frac, KS_ancSim_hidden)

Y2_ = X_Y2_Spline(X_)
Y2_=gaussian_filter1d (Y2_, 3)
plt.plot(X_,Y2_,color='#C84C05',linewidth='2', linestyle='dashed')


X_Y3_Spline = sp.interpolate.make_interp_spline(frac, KS_ancanaly_hidden)

Y3_ = X_Y3_Spline(X_)
Y3_=gaussian_filter1d (Y3_, 3)
plt.plot(X_,Y3_,color='#69247C',linewidth='2', linestyle='dashed')



ax.grid()
ax.set_yscale('log')
ax.set(xlabel=r'Normalized aperture radius $a/W_{LT}$', ylabel='Kolmogorov-Smirnov statistic $D_N$')
plt.savefig("PDT_KS_determ.pdf", **save_kwargs)

