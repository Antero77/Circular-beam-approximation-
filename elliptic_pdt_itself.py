import numpy as np
import scipy as sp
from pyatmosphere import gpu
from pyatmosphere import simulations
from pyatmosphere.theory.pdt import EllipticBeamAnalyticalPDT

gpu.config['use_gpu'] = True

import matplotlib.pyplot as plt
import seaborn as sns
import datetime

### QuickChannel example

then = datetime.datetime.now()
from pyatmosphere import QuickChannel, measures
from pyatmosphere.theory.pdt.pdt import bw_pdt, beam_wandering_pdt
from pyatmosphere.simulations import PDTResult
from pyatmosphere.theory.pdt.elliptic_beam import EllipticBeamAnalyticalPDT


l = 1000
Cn2 = 5*10 ** (-15)
#beam_w0 = (l * 8.08 * 10 ** (-7) / np.pi) ** 0.5
beam_w0=0.02
quick_channel = QuickChannel(
    Cn2=Cn2,
    length=l,
    count_ps=6,
    beam_w0=beam_w0,
    beam_wvl=8.08e-07,
    aperture_radius=0.025,

)

"""
quick_channel.plot(pupil=False)
##print(measures.I(quick_channel, pupil=False))
plt.show()
quick_channel.plot()
plt.show()
"""
# -------------------

etha = []
meanxData=[]
meanx2Data=[]
meany2Data=[]
n = 10 ** 3
for i in range(n):
    output = quick_channel.run(pupil=False)

    meanxData.append(measures.mean_x(quick_channel, output=output))
    meanx2Data.append(measures.mean_x2(quick_channel, output=output))
    meany2Data.append(measures.mean_y2(quick_channel, output=output))

    etha.append(measures.eta(quick_channel, output=quick_channel.run()))





#--------------
a = quick_channel.pupil.radius
elliptic=EllipticBeamAnalyticalPDT(W0=quick_channel.source.w0,a=a,size=2000)
elliptic.set_params_from_data(np.array(meanxData), np.array(meanx2Data), np.array(meany2Data))



ax = sns.histplot(etha, bins=100, kde=False, element="step", stat='density')
ax.set(xlabel='etha', ylabel='pdt')

ax = sns.histplot(elliptic.pdt(), bins=100, kde=False, element="step", stat='density')

#plt.plot(t, elliptic.pdt(), color='orange')

ax.legend(['simulated','elliptic'])
plt.show()



# ------------------
now = datetime.datetime.now()
delta = now - then
print(delta.seconds / 60)

plt.show()
