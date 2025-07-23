import numpy as np
from pyatmosphere import gpu

gpu.config['use_gpu'] = True

import csv
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import scipy as sp

### QuickChannel example

then = datetime.datetime.now()
from pyatmosphere import QuickChannel, measures


corcoef1=[]
corcoef2=[]
scalex=[]
scaley=[]
m=9
for g in range(m):
    for k in range(m):
        l = 100 + 9900 * g / m
        k1 = -17 + 4 * k / m
        C = 10 ** k1
        beam_w0=(l*8.08*10**(-7)/3.14)**0.5
        quick_channel = QuickChannel(
            Cn2=C,
            length=l,
            count_ps=7,
            beam_w0=beam_w0,
            beam_wvl=8.08e-07,
            aperture_radius=0.12
        )

        W2 = []
        x2_0 = []
        integr=[]
        n = 10 ** 3
        for i in range(n):
            output = quick_channel.run(pupil=False)

            W2.append( 4 * (measures.mean_x2(quick_channel, output=output) -
                      (measures.mean_x(quick_channel, output=output)) ** 2) )
            integr.append( measures.mean_x2(quick_channel, output=output) )
            x2_0.append( (measures.mean_x(quick_channel, output=output)) ** 2 )



        corcoef1.append(np.corrcoef(x2_0, integr)[0][1])
        corcoef2.append(np.corrcoef(x2_0, W2)[0][1])
        scalex.append(l)
        scaley.append(np.log10(C))
    quick_channel.plot(pupil=False)
    plt.savefig('length %d.png'%l)


# -------------------


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(scalex,scaley, corcoef1)

ax.set(xlabel='length,m', ylabel='log10(C^2_n)', zlabel='r', title='Correlation coef between x_0^2 and Integral(x^2 I(r))')
plt.show()




fig2 = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(scalex,scaley, corcoef2)

ax.set(xlabel='length,m', ylabel='log10(C^2_n)', zlabel='r', title='Correlation coef between x_0^2 and W^2')
plt.show()


# ------------------
now = datetime.datetime.now()
delta = now - then
print(delta.seconds / 60)

plt.show()
