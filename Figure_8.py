import numpy as np
import scipy as sp
from pyatmosphere import gpu
import matplotlib.pyplot as plt
import circular_beam
from scipy.interpolate import make_interp_spline
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




#----------------------------------


def avrEta(pdts, t):
    sum=0
    count=0
    for i in t:
        sum=sum+i*pdts[count]*(t[1]-t[0])
        count=count+1
    return sum

def avrEta_sqr(pdts, t):
    sum=0
    count=0
    for i in t:
        sum=sum+np.sqrt(i)*pdts[count]*(t[1]-t[0])
        count=count+1
    return sum


def get_squeez(alpha_0, squeez_in, etha, etha_sqr):
    return 10*np.log10( etha*(10**(squeez_in/10)-1) + 1 + 4*alpha_0**2*(etha-etha_sqr**2) )







#----------------------------------

squeez_circSim=[]
squeez_circanaly=[]
squeez_ancSim=[]
squeez_ancanaly=[]
squeez_sim=[]


squeez_circSim2=[]
squeez_circanaly2=[]
squeez_ancSim2=[]
squeez_ancanaly2=[]
squeez_sim2=[]


frac=[]
longterm=0.02797
number_dot_app=20



for i in range(number_dot_app):
    aperture= 0.0085 + (1.3*longterm-0.0085) * i / number_dot_app
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
    sum_etha1_2 = 0

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
        sum_etha1_2 = sum_etha1_2 + temp_etha ** 0.5

    sim_W2 = sumW2 / n
    sim_W4 = sumW4 / n
    sim_x2_0 = sumx2_0 / n
    sim_etha = sum_etha / n
    sim_etha2 = sum_etha2 / n
    sim_etha1_2 = sum_etha1_2 / n
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

    print("etha analy=", analy_etha, "etha sim=", sim_etha)
    print("etha2 analy=", analy_etha2, "etha2 sim=", sim_etha2)

    # --------------------------------------------------------------
    determ_losses = 10 ** (-(3 + 0.1 * (quick_channel.path.length / 1000)) / 10)


    number_for_dots_pdt = 200

    # --------------------------------------------------------------

    t = np.linspace(0.00001, 1, num=number_for_dots_pdt)


    etha = [x * determ_losses for x in etha]
    kde_sim = sp.stats.gaussian_kde(etha)
    pdt_etha=kde_sim.pdf(t)


    circ_sim = circular_beam.CircularBeamModel.from_beam_params(
        S_BW=np.sqrt(sim_x2_0),
        W2_mean=sim_W2,
        W4_mean=sim_W4,
        aperture_radius=quick_channel.pupil.radius
    )
    pdt_circ_sim = circ_sim.get_pdt(t)



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



    squeez_in=-3
    alpha0=4

    squeez_sim.append(get_squeez(alpha0, squeez_in, sim_etha*determ_losses, sim_etha1_2*determ_losses**0.5))
    squeez_circSim.append(get_squeez(alpha0, squeez_in, avrEta(pdt_circ_sim, t) * determ_losses, avrEta_sqr(pdt_circ_sim, t) * determ_losses ** 0.5))
    squeez_circanaly.append(get_squeez(alpha0, squeez_in, avrEta(pdt_circ_analy, t) * determ_losses, avrEta_sqr(pdt_circ_analy, t) * determ_losses ** 0.5))
    squeez_ancSim.append(get_squeez(alpha0, squeez_in, avrEta(pdt_anc_sim, t) * determ_losses, avrEta_sqr(pdt_anc_sim, t) * determ_losses ** 0.5))
    squeez_ancanaly.append(get_squeez(alpha0, squeez_in, avrEta(pdt_anc_analy, t) * determ_losses, avrEta_sqr(pdt_anc_analy, t) * determ_losses ** 0.5))
    frac.append(quick_channel.pupil.radius / longterm)



#-----------------------------------------------------------------
fig, ax = plt.subplots(1, 1)
X_ = np.linspace(np.min(frac), np.max(frac), 200)


X_Y0_Spline = sp.interpolate.make_interp_spline(frac, squeez_sim)

Y0_ = X_Y0_Spline(X_)

plt.plot(X_,Y0_,linewidth='2',color='C0')



X_Y_Spline = sp.interpolate.make_interp_spline(frac, squeez_circSim)

Y_ = X_Y_Spline(X_)

plt.plot(X_,Y_,linewidth='2',color='r')


X_Y1_Spline = sp.interpolate.make_interp_spline(frac, squeez_circanaly)

Y1_ = X_Y1_Spline(X_)

plt.plot(X_,Y1_,linewidth='2',color='green')


X_Y2_Spline = sp.interpolate.make_interp_spline(frac, squeez_ancSim)

Y2_ = X_Y2_Spline(X_)

plt.plot(X_,Y2_,color='orange',linewidth='2', linestyle='dashed')


X_Y3_Spline = sp.interpolate.make_interp_spline(frac, squeez_ancanaly)

Y3_ = X_Y3_Spline(X_)

plt.plot(X_,Y3_,color='violet',linewidth='2', linestyle='dashed')




ax.grid()
ax.set_ylim(-1.25, 0.0)
ax.set(xlabel=r'Normalized aperture radius $a/W_{\text{LT}}$', ylabel='Squeezing (dB)')
plt.savefig("app_Squeezing4.pdf", **save_kwargs)

