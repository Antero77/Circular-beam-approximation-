import numpy as np
import scipy as sp
import scipy.special
from pyatmosphere import gpu
from pyatmosphere import simulations


gpu.config['use_gpu'] = True

import matplotlib.pyplot as plt

plt.rcParams['axes.axisbelow'] = True
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = "STIX"
plt.rcParams["mathtext.fontset"] = "cm"
#plt.rcParams.update({'font.size': 15})
plt.rcParams.update({'font.size': 23})

save_kwargs = {
        "format": "pdf",
        "dpi": 300,
        "bbox_inches": "tight",
        "pad_inches": 0.005
        }


import datetime

### QuickChannel example

then = datetime.datetime.now()
from pyatmosphere import QuickChannel, measures


data_analy_x20=[]
data_analy_S=[]
data_analy_S2=[]
data_analy_1eta=[]
data_analy_1eta2=[]
data_analy_2eta=[]
data_analy_2eta2=[]
data_analy_1_delta_eta2=[]
data_analy_2_delta_eta2=[]


data_sim_x20=[]
data_sim_S=[]
data_sim_S2=[]
data_sim_1eta=[]
data_sim_1eta2=[]
data_sim_2eta=[]
data_sim_2eta2=[]
data_sim_1_delta_eta2=[]
data_sim_2_delta_eta2=[]


dataX=[]






number_dot=10
for i in range(number_dot):

    l = l = 500+(4500-500)*i/number_dot
    g = -15
    Cn2 = 10 ** g
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
        l0=1e-6,
        L0=5e3,
        f_min=1 / 5e3 / 15,
        f_max=1 / 1e-6 * 2

    )

    quick_channel_another = QuickChannel(
        Cn2=Cn2,
        length=l,
        count_ps=6,
        beam_w0=beam_w0,
        beam_wvl=8.08e-07,
        aperture_radius=0.015,
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

    first_etha = []
    first_etha2 = []
    sum_1etha = 0
    sum_1etha2 = 0


    second_etha = []
    second_etha2 = []
    sum_2etha = 0
    sum_2etha2 = 0


    n = 10 ** 4
    for i in range(n):
        output = quick_channel.run(pupil=False)

        W2 = 4 * (measures.mean_x2(quick_channel, output=output) -
                  (measures.mean_x(quick_channel, output=output)) ** 2)
        sumW2 = sumW2 + W2
        sumW4 = sumW4 + W2 ** 2

        sumx2_0 = sumx2_0 + (measures.mean_x(quick_channel, output=output)) ** 2

        temp_1etha = measures.eta(quick_channel, output=quick_channel.run())
        temp_2etha = measures.eta(quick_channel, output=quick_channel_another.run())


        sum_1etha = sum_1etha + temp_1etha
        sum_1etha2 = sum_1etha2 + temp_1etha ** 2



        sum_2etha = sum_2etha + temp_2etha
        sum_2etha2 = sum_2etha2 + temp_2etha ** 2
        # etha.append(measures.eta(quick_channel, output=quick_channel.run()))

    sim_W2 = sumW2 / n
    sim_W4 = sumW4 / n
    sim_x2_0 = sumx2_0 / n

    sim_1etha = sum_1etha / n
    sim_1etha2 = sum_1etha2 / n

    sim_2etha = sum_2etha / n
    sim_2etha2 = sum_2etha2 / n

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
    #initial analytical
    analy_etha=1-np.exp(-2*quick_channel.pupil.radius**2/(analy_W2+4*analy_x2_0))
    #analytical with local approx
    """
    th = 0.136 * popravka_ro * quick_channel.get_rythov2() * omega ** (-5 / 6)
    analy_1etha = np.exp(-th) * (1 - np.exp(
        -quick_channel.pupil.radius ** 2 * omega ** 2 / quick_channel.source.w0 ** 2 / (0.5 + 5 * th)))


    analy_2etha = np.exp(-th) * (1 - np.exp(
        -quick_channel_another.pupil.radius ** 2 * omega ** 2 / quick_channel.source.w0 ** 2 / (0.5 + 5 * th)))





    alph = omega ** (-2) + 3.26 * omega ** (-7 / 6) * popravka_ro * quick_channel.get_rythov2()

    norm = 1
    first_mn = 1 - np.exp(
        -quick_channel.pupil.radius ** 2 * (alph * omega ** 2 + 1) / alph / quick_channel.source.w0 ** 2)
    second_mn = 1 - np.exp(
        -4 * omega ** 2 * quick_channel.pupil.radius ** 2 / quick_channel.source.w0 ** 2 / (alph * omega ** 2 + 1))

    analy_1etha2 = norm * first_mn * second_mn


    first_mn = 1 - np.exp(
        -quick_channel_another.pupil.radius ** 2 * (alph * omega ** 2 + 1) / alph / quick_channel.source.w0 ** 2)
    second_mn = 1 - np.exp(
        -4 * omega ** 2 * quick_channel_another.pupil.radius ** 2 / quick_channel.source.w0 ** 2 / (alph * omega ** 2 + 1))

    analy_2etha2 = norm * first_mn * second_mn




    print("1etha analy=", analy_1etha, "1etha sim=", sim_1etha)
    print("1etha2 analy=", analy_1etha2, "1etha2 sim=", sim_1etha2)
    print("delta 1etha2 analy=", analy_1etha2 - analy_1etha ** 2, "delta 1etha2 sim=", sim_1etha2 - sim_1etha ** 2)


    print("2etha analy=", analy_2etha, "2etha sim=", sim_2etha)
    print("2etha2 analy=", analy_2etha2, "2etha2 sim=", sim_2etha2)
    print("delta 2etha2 analy=", analy_2etha2 - analy_2etha ** 2, "delta 2etha2 sim=", sim_2etha2 - sim_2etha ** 2)

    data_analy_x20.append(analy_x2_0*10**6)   #into mm**2

    data_analy_S.append(analy_W2*10**4) # into cm**2
    data_analy_S2.append(analy_W4*10**8) # into cm**4

    data_analy_1eta.append(analy_1etha)
    data_analy_1eta2.append(analy_1etha2)

    data_analy_2eta.append(analy_2etha)
    data_analy_2eta2.append(analy_2etha2)

    data_analy_1_delta_eta2.append((analy_1etha2-analy_1etha**2)*10**2) #mult by 10**2 for graph
    data_analy_2_delta_eta2.append((analy_2etha2-analy_2etha**2)*10**2) #mult by 10**2 for graph


    data_sim_x20.append(sim_x2_0*10**6)  # into mm**2

    data_sim_S.append(sim_W2*10**4) # into cm**2
    data_sim_S2.append(sim_W4*10**8) # into cm**4

    data_sim_1eta.append(sim_1etha)
    data_sim_1eta2.append(sim_1etha2)

    data_sim_2eta.append(sim_2etha)
    data_sim_2eta2.append(sim_2etha2)

    data_sim_1_delta_eta2.append((sim_1etha2-sim_1etha**2)*10**2) #mult by 10**2 for graph
    data_sim_2_delta_eta2.append((sim_2etha2-sim_2etha**2)*10**2) #mult by 10**2 for graph



    dataX.append(quick_channel.get_rythov2())


# ------------------
X_Y_Spline = sp.interpolate.make_interp_spline(dataX, data_analy_x20)

X_ = np.linspace(np.min(dataX), np.max(dataX), 200)
Y_ = X_Y_Spline(X_)

X_Z_Spline = sp.interpolate.make_interp_spline(dataX, data_sim_x20)


Z_ = X_Z_Spline(X_)


fig, ax = plt.subplots(1, 1)
plt.plot(X_,Y_,linewidth='3', color='#99DDFF', linestyle='solid')
plt.plot(X_,Z_,linewidth='3', color='#EE8866', linestyle='dashed')
ax.set(xlabel=r'Rytov parameter $σ^2_R$', ylabel=r'Variance $σ^2_{\text{bw}}$ (mm$^2$)')
#ax.set_yscale('log')
#plt.ylim([5*10**(-4),10**(0)])
ax.grid()


plt.savefig("comp_variance.pdf", **save_kwargs)

#plt.show()




# ------------------
X_Y_Spline = sp.interpolate.make_interp_spline(dataX, data_analy_S)

X_ = np.linspace(np.min(dataX), np.max(dataX), 200)
Y_ = X_Y_Spline(X_)

X_Z_Spline = sp.interpolate.make_interp_spline(dataX, data_sim_S)


Z_ = X_Z_Spline(X_)





fig, ax = plt.subplots(1, 1)




ax.plot(X_,Y_,linewidth='3', color='#99DDFF', linestyle='solid')
ax.plot(X_,Z_,linewidth='3', color='#EE8866', linestyle='dashed')






ax.set_xlabel(r'Rytov parameter $σ^2_R$')
ax.set_ylabel(r'First moment $\langle S \rangle$ (cm$^2$)')


#ax.set_yscale('log')
#plt.ylim([5*10**(-4),10**(0)])
ax.grid()


plt.savefig("comp_S.pdf", **save_kwargs)

#plt.show()

#-------------------



X_Y_Spline = sp.interpolate.make_interp_spline(dataX, data_analy_S2)

Y_ = X_Y_Spline(X_)

X_Z_Spline = sp.interpolate.make_interp_spline(dataX, data_sim_S2)

Z_ = X_Z_Spline(X_)







fig, ax = plt.subplots(1, 1)




ax.plot(X_,Y_,linewidth='3', color='#99DDFF', linestyle='solid')
ax.plot(X_,Z_,linewidth='3', color='#EE8866', linestyle='dashed')




ax.set_xlabel(r'Rytov parameter $σ^2_R$')
ax.set_ylabel(r'Second moment $\langle S^2 \rangle$ (cm$^4$)')

#ax.set_yscale('log')
#plt.ylim([5*10**(-4),10**(0)])
ax.grid()


plt.savefig("comp_S2.pdf", **save_kwargs)

#plt.show()






# ------------------


X_Y_Spline = sp.interpolate.make_interp_spline(dataX, data_analy_1eta)

X_ = np.linspace(np.min(dataX), np.max(dataX), 200)
Y_ = X_Y_Spline(X_)

X_Z_Spline = sp.interpolate.make_interp_spline(dataX, data_sim_1eta)


Z_ = X_Z_Spline(X_)


X_Y2_Spline = sp.interpolate.make_interp_spline(dataX, data_analy_2eta)

Y2_ = X_Y2_Spline(X_)

X_Z2_Spline = sp.interpolate.make_interp_spline(dataX, data_sim_2eta)

Z2_ = X_Z2_Spline(X_)







fig, ax = plt.subplots(1, 1)




ax.plot(X_,Y_,linewidth='3', color='#99DDFF', linestyle='solid') # first aper, etha, analy
ax.plot(X_,Z_,linewidth='3', color='#EE8866', linestyle='dashed') # first aper, etha, sim

ax.plot(X_,Y2_,linewidth='3', color='#99DDFF', linestyle='dashdot') # second aper, etha, analy
ax.plot(X_,Z2_,linewidth='3', color='#EE8866', linestyle='dotted') # second aper, etha, sim



ax.set_xlabel(r'Rytov parameter $σ^2_R$')
ax.set_ylabel(r'First moment $\langle \eta \rangle$')


#ax.set_yscale('log')
#plt.ylim([5*10**(-4),10**(0)])
ax.grid()


plt.savefig("comp_eta.pdf", **save_kwargs)

#plt.show()




#--------------------



X_Y3_Spline = sp.interpolate.make_interp_spline(dataX, data_analy_1eta2)


Y3_ = X_Y3_Spline(X_)

X_Z3_Spline = sp.interpolate.make_interp_spline(dataX, data_sim_1eta2)


Z3_ = X_Z3_Spline(X_)


X_Y4_Spline = sp.interpolate.make_interp_spline(dataX, data_analy_2eta2)

Y4_ = X_Y4_Spline(X_)

X_Z4_Spline = sp.interpolate.make_interp_spline(dataX, data_sim_2eta2)

Z4_ = X_Z4_Spline(X_)








fig, ax = plt.subplots(1, 1)




ax.plot(X_,Y3_,linewidth='3', color='#99DDFF', linestyle='solid') # first aper, etha2, analy
ax.plot(X_,Z3_,linewidth='3', color='#EE8866', linestyle='dashed') # first aper, etha2, sim

ax.plot(X_,Y4_,linewidth='3', color='#99DDFF', linestyle='dashdot') # second aper, etha2, analy
ax.plot(X_,Z4_,linewidth='3', color='#EE8866', linestyle='dotted') # second aper, etha2, sim


ax.set_xlabel(r'Rytov parameter $σ^2_R$')
ax.set_ylabel(r'Second moment $\langle \eta^2 \rangle$')

#ax.set_yscale('log')
#plt.ylim([5*10**(-4),10**(0)])
ax.grid()


plt.savefig("comp_eta2.pdf", **save_kwargs)

#plt.show()









#--------------------

X_Y_Spline = sp.interpolate.make_interp_spline(dataX, data_analy_1_delta_eta2)

X_ = np.linspace(np.min(dataX), np.max(dataX), 200)
Y_ = X_Y_Spline(X_)

X_Z_Spline = sp.interpolate.make_interp_spline(dataX, data_sim_1_delta_eta2)


Z_ = X_Z_Spline(X_)


X_Y2_Spline = sp.interpolate.make_interp_spline(dataX, data_analy_2_delta_eta2)

Y2_ = X_Y2_Spline(X_)

X_Z2_Spline = sp.interpolate.make_interp_spline(dataX, data_sim_2_delta_eta2)

Z2_ = X_Z2_Spline(X_)







fig, ax = plt.subplots(1, 1)




ax.plot(X_,Y_,linewidth='3', color='#99DDFF', linestyle='solid')
ax.plot(X_,Z_,linewidth='3', color='#EE8866', linestyle='dashed')


ax.plot(X_,Y2_,linewidth='3', color='#99DDFF', linestyle='dashdot')
ax.plot(X_,Z2_,linewidth='3', color='#EE8866', linestyle='dotted')



ax.set_xlabel(r'Rytov parameter $σ^2_R$')
ax.set_ylabel(r'Variance $\langle \Delta \eta^2 \rangle \times 10^{-2}$')


#ax.set_yscale('log')
#plt.ylim([5*10**(-4),10**(0)])
ax.grid()


plt.savefig("comp_delta_eta.pdf", **save_kwargs)

#plt.show()






# ------------------
now = datetime.datetime.now()
delta = now - then
print(delta.seconds / 60, "min")

plt.show()
