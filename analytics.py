import numpy as np


popravka_ro = 1.457 / 1.5

def get_omega(k,w0,L):
    return k * w0 ** 2 / 2 / L

def get_x2_0(k,w0,L ,rythov2):
    return (0.32 * w0 ** 2 * popravka_ro * rythov2 * get_omega(k,w0,L) ** (-7 / 6) -
              0.06 * w0 ** 2 * popravka_ro ** 2 * rythov2 ** 2 * get_omega(k,w0,L) ** (-1 / 3))

def get_W2(k,w0,L ,rythov2):
    return 4 * (w0 ** 2 * get_omega(k,w0,L) ** (-2) / 4 + 1.07 * w0 ** 2 * popravka_ro * rythov2 *
                get_omega(k,w0,L) ** (-7 / 6) - get_x2_0(k,w0,L ,rythov2))

def get_W4(k,w0,L ,rythov2):
    return 16 * (w0 ** 4 * get_omega(k,w0,L) ** (-4) / 16 + 0.58 * w0 ** 4 * popravka_ro * rythov2 *
                 get_omega(k,w0,L) ** (-19 / 6) + 1.37 * w0 ** 4 * popravka_ro ** 2 * rythov2 ** 2 *
                 get_omega(k,w0,L) ** (-7 / 3) - 0.5 * get_W2(k,w0,L ,rythov2) * get_x2_0(k,w0,L ,rythov2) -
                 3 * get_x2_0(k,w0,L ,rythov2) ** 2)

def get_etha(k,w0,L,rythov2, a):
    th = 0.136 * popravka_ro * rythov2 * get_omega(k,w0,L) ** (-5 / 6)
    # Although th is close to zero, setting th = 0 guarantees physical consistency, albeit at the expense of some accuracy

    return np.exp(-th) * (1 - np.exp(-a ** 2 * get_omega(k,w0,L) ** 2 / w0 ** 2 / (0.5 + 5 * th)))

def get_etha2(k,w0,L,rythov2, a):
    alph = get_omega(k,w0,L) ** (-2) + 3.26 * get_omega(k,w0,L) ** (-7 / 6) * popravka_ro * rythov2
    first_mn = 1 - np.exp( -a ** 2 * (alph * get_omega(k,w0,L) ** 2 + 1) / alph / w0 ** 2)
    second_mn = 1 - np.exp(-4 * get_omega(k,w0,L) ** 2 * a ** 2 / w0 ** 2 / (alph * get_omega(k,w0,L) ** 2 + 1))
    
    return first_mn * second_mn
