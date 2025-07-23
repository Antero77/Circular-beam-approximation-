import numpy as np
from pyatmosphere import gpu

gpu.config['use_gpu'] = True

from pyatmosphere.theory.atmosphere.beam_wandering import get_r_bw
from pyatmosphere.theory.atmosphere.gamma2 import get_gamma_2
import scipy
import matplotlib.pyplot as plt
import datetime
import scipy as sp
import mpmath


then = datetime.datetime.now()
from pyatmosphere import QuickChannel, measures

# -------------------



m=1
for count in range(m):

    l=1500+(8000-1000)*count/m
    Cn2 = 10 ** (-14.85)
    beam_w0=np.sqrt(2 * 10000 / 2 / np.pi * 808e-9)
    quick_channel = QuickChannel(
        Cn2=Cn2,
        length=l,
        count_ps=6,
        beam_wvl=8.08e-07,
        grid_resolution=512,



    )
    print("rytov2=",quick_channel.get_rythov2())

    # -------------------

    sumW2 = 0
    sumW4 = 0
    sumx2_0 = 0
    for_gamma4=0

    """
    n = 5* 10 ** 3
    for i in range(n):
        output = quick_channel.run(pupil=False)

        W2 = 4 * (measures.mean_x2(quick_channel, output=output) -
                  (measures.mean_x(quick_channel, output=output)) ** 2)
        sumW2 = sumW2 + W2
        sumW4 = sumW4 + W2 ** 2
        sumx2_0 = sumx2_0 + (measures.mean_x(quick_channel, output=output)) ** 2

        for_gamma4=for_gamma4+measures.mean_x2(quick_channel, output=output)**2


    # ------------
    meanW2 = sumW2 / n
    meanW4 = sumW4 / n
    meanx2_0 = sumx2_0 / n
    mean_for_gamma4=for_gamma4/n
    print("x2_0=",meanx2_0,"S=",meanW2, "S2=",meanW4)
    """

    #aux functions

    def S(z, S0, F,k):
        return S0*((1-z/F)**2+(2*z/k/S0)**2)
    def phi_n(kappa,l0, L0,Cn2):
        k0 = (2 * np.pi) / L0
        km = 5.92 / l0
        return 0.033 * Cn2 * np.exp(-(kappa / km)**2) / (kappa**2 + k0**2)**(11/6)

    def D_sp(rho, L, k, l0, L0,Cn2):
        def integr1(kappa, z, rho, L, l0, L0,Cn2):
            return kappa * phi_n(kappa,l0, L0,Cn2) * (1 - scipy.special.jn(0, kappa*rho*z/L))
        return 8*np.pi**2*k**2 * sp.integrate.dblquad(integr1, 0, L,0, np.inf, args=(rho, L, l0, L0,Cn2))[0]


    def D_sp_Karman(rho, L, k, l0, L0, Cn2):
        k0 = (2 * np.pi) / L0
        return 1.09* Cn2* k**2*L*l0**(-1/3)* rho**2*( (1+rho**2/l0**2)**(-1/6) -0.72*(k0*l0)**(1/3))


    #for weak


    #def x2_0_weak(L, model, source):
    #    return 0.5*get_r_bw(L, model, source)**2


    #x2_0 weak, Andrews page 203, from formula 88 modified
    def x2_0_weak(L, S0, F, k, l0, L0,Cn2):
        def integr1(kappa, z, L, S0, F, k, l0, L0,Cn2):
            Lambd = 2 * L / k / S(L, S0, F, k)
            return kappa*phi_n(kappa,l0, L0,Cn2)* np.exp(-kappa**2*S(z, S0, F,k)) *(1-np.exp(-Lambd*kappa**2*(L-z)**2/L/k))
        return 2*np.pi**2*k**2*S(L, S0, F,k)*sp.integrate.dblquad(integr1,  0, L,0, np.inf, args=(L, S0, F, k, l0, L0,Cn2))[0]


    print("analytical weak x2_0=",x2_0_weak(quick_channel.path.length, quick_channel.source.w0**2, quick_channel.source.F0, quick_channel.source.k,
                    quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                    quick_channel.path.phase_screen.model.Cn2))

    print("Semenov x2_0 weak=",0.94*quick_channel.path.phase_screen.model.Cn2*quick_channel.path.length**3*quick_channel.source.w0**(-1/3))



    #W2 weak, Andrews page 189, from formula 40 modified

    def gamma2_weak(r, L, S0, F, k, l0, L0, Cn2, **params):
        """
        For weak turbulence channels.
        `int gamma2(r) dr^2 = 1` normalization.

        Andrews p. 189 eq. 40 with eq. 37, 38
        """

        _S = S(L, S0, F, k)
        _const = 2 / _S / np.pi
        Lambd = 2 * L / k / _S

        def integr1(kappa, ksi):
            _exp = np.exp(-Lambd * L * kappa ** 2 * ksi ** 2 / k)
            _bess = scipy.special.iv(0, 2 * Lambd * r * kappa * ksi)
            _phi_n = phi_n(kappa,l0, L0,Cn2)
            return kappa * _phi_n * (_exp * _bess - 1)

        _integr_const = 4 * np.pi ** 2 * k ** 2 * L
        _integr = _integr_const * scipy.integrate.dblquad(integr1, 0, 1, 0, np.inf, epsabs=1e-17)[0]
        return _const * np.exp(-2 * r ** 2 / _S + _integr)

    """
    def gamma2_weak(r,L, S0, F, k, l0, L0,Cn2):
        def integr1(kappa, ksi,r, L, S0, F, k, l0, L0,Cn2):
            Lambd = 2 * L / k / S(L, S0, F, k)

            #print()

            if 2 * Lambd * r * kappa * ksi>2:
                g=(2*np.pi* 2 * Lambd * r * kappa * ksi)**(-0.5)*np.exp(2 * Lambd * r * kappa * ksi - Lambd * L * kappa ** 2 * ksi ** 2 / k)
            else:
                g=np.exp(-Lambd * L * kappa ** 2 * ksi ** 2 / k) * scipy.special.iv(0, 2 * Lambd * r * kappa * ksi)


            a = -4 * np.pi ** 2 * k ** 2 * L * kappa*phi_n(kappa,l0, L0,Cn2)*(1-g)
            #print(a)
            return a
        b = sp.integrate.dblquad(integr1,  0, 1,0, np.inf, args=(r, L, S0, F, k, l0, L0, Cn2))[0]
        #print(b)
        if b>4:
            b=1
        else:
            b=np.exp(b)
        #print(b)
        return  (2/S(L, S0, F,k)/np.pi) * np.exp(-2*r**2/S(L, S0, F,k)) *b

    """

    """

    x = np.linspace(0, 0.17, 100)
    y = [gamma2_weak(i, quick_channel.path.length, quick_channel.source.w0**2, quick_channel.source.F0, quick_channel.source.k,
                    quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                    quick_channel.path.phase_screen.model.Cn2) for i in x]

    plt.plot(x, y)
    
    
    """


    def W2_weak(L, S0, F, k, l0, L0,Cn2):
        def integr(r,L, S0, F, k, l0, L0,Cn2):
            return np.pi*r**3*gamma2_weak(r,L, S0, F, k, l0, L0,Cn2)
        right=0.17
        return 4*(sp.integrate.quad(integr, 0, right,  args=(L, S0, F, k, l0, L0,Cn2))[0] - x2_0_weak(L, S0, F, k, l0, L0, Cn2))


    """
    
    def integr(r, L, S0, F, k, l0, L0, Cn2):
        return np.pi * r ** 3 * gamma2_weak(r, L, S0, F, k, l0, L0, Cn2)
    x = np.linspace(0, 0.3, 100)
    y = [integr(i, quick_channel.path.length, quick_channel.source.w0**2, quick_channel.source.F0, quick_channel.source.k,
                    quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                    quick_channel.path.phase_screen.model.Cn2) for i in x]

    plt.plot(x, y)

    """
    temp=W2_weak(quick_channel.path.length, quick_channel.source.w0**2, quick_channel.source.F0, quick_channel.source.k,
                    quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                    quick_channel.path.phase_screen.model.Cn2)
    print("analytical weak S =",temp)




    #from formula 45 modified
    def gamma2_weak_simple(r, L, S0, F, k, l0,L0, Cn2 ,rytov2):
        Lambd = 2 * L / k / S(L, S0, F, k)
        #Kolmogorov
        #S_cor= S(L, S0, F, k)*(1+1.33*rytov2*Lambd**(5/6))
        #von Karman
        S_cor = S(L, S0, F, k) * (1 + 4.35* Cn2*k**2*L* (5.92 / l0)**(-5/3)*(scipy.special.hyp2f1(-5/6,1/2,3/2,-Lambd*L*(5.92 / l0)**2/k)-1)-
                                  0.78* Cn2*k**2*L* ((2 * np.pi) / L0)**(-5/3)*(scipy.special.hyp2f2(1/2,1,3/2,1/6,Lambd*L*((2 * np.pi) / L0)**2/k)-1))
        return (2/S_cor/np.pi)*np.exp(-2*r**2/S_cor)





    # W4 weak, Andrews page 279, from formula 46 modified

    def gamma4_weak(r1, r2, a1, a2, L, S0, F, k, l0, L0, Cn2):
        def integrand(kappa, xi, r1, r2,a1,a2 ,L, S0, F, k, l0, L0, Cn2):
            p = np.sqrt(r1**2+r2**2-2*r1*r2*np.cos(a1-a2))
            R = 0.5*np.sqrt(r1**2+r2**2+2*r1*r2*np.cos(a1-a2))

            Theta_bar = 1 - (1-L/F)*S0/S(L, S0, F, k)
            Lambd = 2 * L / k / S(L, S0, F, k)



            #modul=np.sqrt((1 - Theta_bar * xi) ** 2 * p**2 + (2 * Lambd * xi * R) ** 2)
            modul=np.sqrt( (1 - Theta_bar * xi) ** 2 * p**2 - 4 * Lambd**2 * xi**2 * R**2 +
                           2* 1j*Lambd * xi * (1 - Theta_bar * xi)*(r1**2-r2**2) )
            real1 = scipy.special.jn(0, kappa * modul)



            exp2=np.exp(-1j * L * kappa ** 2 * xi * (1 - Theta_bar * xi) / k)

            if abs((1 - Theta_bar * xi - 1j * Lambd * xi) * kappa * p) > 10:
                Bes2 = 1
            else:
                Bes2=scipy.special.jn(0, (1 - Theta_bar * xi - 1j * Lambd * xi) * kappa * p)



            real2 = (exp2*Bes2)
            #print(exp2*Bes2)
            real_part = np.real(real1 - real2)


            return (8*np.pi**2 * k ** 2 * L ) * kappa * phi_n(kappa, l0, L0, Cn2) * np.exp(-Lambd * L * xi ** 2 * kappa ** 2 / k) * real_part

        return  sp.integrate.dblquad(integrand,  0,1, 3,np.inf,args=(r1, r2,a1,a2 ,L, S0, F, k, l0, L0, Cn2),epsrel=1.49e-04)[0]


    def gamma4_weak_final(r1, r2, a1, a2, L, S0, F, k, l0, L0, Cn2):
        return gamma2_weak(r1,L, S0, F, k, l0, L0,Cn2)*gamma2_weak(r2,L, S0, F, k, l0, L0,Cn2)*(1+gamma4_weak(r1, r2, a1, a2, L, S0, F, k, l0, L0, Cn2))




    """
    sum=0
    n=10
    a=0.2
    b=2*np.pi
    for i in range(n):
        for j in range(n):
            for k in range(n):
                sum = sum + gamma4_weak_final(i * a / n, j * a / n, k * (b / n), 0,
                                                                          quick_channel.path.length,
                                                                          quick_channel.source.w0 ** 2,
                                                                          quick_channel.source.F0,
                                                                          quick_channel.source.k,
                                                                          quick_channel.path.phase_screen.model.l0,
                                                                          quick_channel.path.phase_screen.model.L0,
                                                                          quick_channel.path.phase_screen.model.Cn2) * (a / n) ** 2 * (b / n) * (2 * np.pi)

    print(sum)
    """
    """
    
    x = np.linspace(0, 0.2, 100)
    y = [gamma4_weak_final(i, i, 0, 0, quick_channel.path.length, quick_channel.source.w0**2, quick_channel.source.F0, quick_channel.source.k,
                    quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                    quick_channel.path.phase_screen.model.Cn2) for i in x]

    z = [gamma2_weak(i, quick_channel.path.length, quick_channel.source.w0 ** 2, quick_channel.source.F0,
                       quick_channel.source.k,
                       quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                       quick_channel.path.phase_screen.model.Cn2) ** 2 for i in x]

    plt.plot(x, y, 'red', x, z, 'blue')

    
    

    x = np.linspace(0, 0.2, 100)
    y = [gamma4_weak(i, i, 0, 0, quick_channel.path.length, quick_channel.source.w0**2, quick_channel.source.F0, quick_channel.source.k,
                    quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                    quick_channel.path.phase_screen.model.Cn2) for i in x]

    plt.plot(x, y)

    """


    # W4 weak, Andrews page 280, from formula 50 modified

    def gamma4_weak_simple(r1, r2, a1, a2, L, S0, F, k, l0, L0, Cn2, rytov2):
        def funct(r1, r2, a1, a2, L, S0, F, k, rytov2):
            p = np.sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * np.cos(a1 - a2))
            R = 0.5 * np.sqrt(r1 ** 2 + r2 ** 2 + 2 * r1 * r2 * np.cos(a1 - a2))

            Theta_bar = 1 - (1 - L / F) * S0 / S(L, S0, F, k)
            Lambd = 2 * L / k / S(L, S0, F, k)
            d=0.67-0.17*(1+L/F)

            real1=1j**(5/6)*(1-d*(Theta_bar+1j*Lambd))**(5/6)*mpmath.hyp1f1(-5/6, 1, -k*p**2*(1-d*(Theta_bar+1j*Lambd))/4/1j/L/d)

            modul = np.sqrt((1 - Theta_bar * d) ** 2 * p ** 2 - 4 * Lambd ** 2 * d ** 2 * R ** 2 +
                            2 * 1j * Lambd * d * (1 - Theta_bar * d) * (r1 ** 2 - r2 ** 2))
            real2=(Lambd*d)**(5/6)*mpmath.hyp1f1(-5/6, 1, -k*modul**2/4/L/d**2)


            real_part = np.real(real1 - real2)

            return 3.87*rytov2*real_part
        return gamma2_weak(r1, L, S0, F, k, l0, L0, Cn2) * gamma2_weak(r2, L, S0, F, k, l0, L0, Cn2) * (1 + funct(r1, r2, a1, a2, L, S0, F, k, rytov2))


    """
    norm=0
    sum = 0
    n = 20
    a = 0.2
    b = 2 * np.pi
    for i in range(n):
        for j in range(n):
            for k in range(n):
                tr=gamma4_weak_simple(i * a / n, j * a / n, k * (b / n), 0,
                                                                          quick_channel.path.length,
                                                                          quick_channel.source.w0 ** 2,
                                                                          quick_channel.source.F0,
                                                                          quick_channel.source.k,
                                                                          quick_channel.path.phase_screen.model.l0,
                                                                          quick_channel.path.phase_screen.model.L0,
                                                                          quick_channel.path.phase_screen.model.Cn2,
                                                                            quick_channel.get_rythov2()) * (
                                  a / n) ** 2 * (b / n) * (2 * np.pi)
                sum = sum + 0.25*(i * a / n)**3 * (j * a / n)**3 * tr
                norm= norm+(i * a / n) * (j * a / n)*tr
    print(sum, norm)
    sum=sum/norm
    print(sum)

    longterm=W2_weak(quick_channel.path.length, quick_channel.source.w0**2, quick_channel.source.F0, quick_channel.source.k,
                    quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                    quick_channel.path.phase_screen.model.Cn2)
    x2_0=x2_0_weak(quick_channel.path.length, quick_channel.source.w0 ** 2, quick_channel.source.F0,
                  quick_channel.source.k,
                  quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                  quick_channel.path.phase_screen.model.Cn2)
    w2=4*(longterm-x2_0)
    print("S**2=",16*(sum-0.5*w2*x2_0-3*x2_0**2))

    """





    def integral_x1_2_x2_2_gamma4_weak(L, S0, F, k, l0, L0, Cn2,rytov2):
        def integ(r1, r2,  L, S0, F, k, l0, L0, Cn2,rytov2):
            def integ1(r1, r2,  L, S0, F, k, l0, L0, Cn2,rytov2):
                def integ2(a1, a2,r1, r2,  L, S0, F, k, l0, L0, Cn2,rytov2):
                    return gamma4_weak_simple(r1, r2, a1, a2, L, S0, F, k, l0, L0, Cn2,rytov2)
                return sp.integrate.dblquad(integ2, 0, 2*np.pi, 0, 2*np.pi, args=(r1, r2,  L, S0, F, k, l0, L0, Cn2,rytov2))[0]
            return 0.25*r1**3* r2**3* integ1(r1, r2,  L, S0, F, k, l0, L0, Cn2,rytov2)
        return sp.integrate.dblquad(integ, 0, np.inf, 0, np.inf, args=(L, S0, F, k, l0, L0, Cn2,rytov2))


    """
    print(integral_x1_2_x2_2_gamma4_weak(quick_channel.path.length, quick_channel.source.w0**2, quick_channel.source.F0, quick_channel.source.k,
                    quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                    quick_channel.path.phase_screen.model.Cn2, quick_channel.get_rythov2()))
    """
    """
    x = np.linspace(0, 0.20, 50)
    y = [0.25*i**3*(i+0.05)**3*gamma4_weak_simple(i, i+0.1, 0, 0, quick_channel.path.length, quick_channel.source.w0**2, quick_channel.source.F0, quick_channel.source.k,
                    quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                    quick_channel.path.phase_screen.model.Cn2, quick_channel.get_rythov2()) for i in x]

    plt.plot(x, y)
    """
    """

    a=0
    edge_l=0
    while a<1e-7:
        edge_l=edge_l+0.001
        a=0.25*edge_l**6*gamma4_weak_final(edge_l, edge_l, 0, 0, quick_channel.path.length, quick_channel.source.w0**2, quick_channel.source.F0, quick_channel.source.k,
                    quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                    quick_channel.path.phase_screen.model.Cn2)

    a=1000
    edge_r=edge_l
    while a>1e-7:
        edge_r=edge_r+0.001
        a=0.25*edge_r**6*gamma4_weak_final(edge_r, edge_r, 0, 0, quick_channel.path.length, quick_channel.source.w0**2, quick_channel.source.F0, quick_channel.source.k,
                    quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                    quick_channel.path.phase_screen.model.Cn2)


    print(edge_l, edge_r)


    """
    """
    sum = 0
    n = 10
    a=0.02
    c=0.17
    #a = edge_l
    #c = edge_r
    delt=(c-a)/n
    b = 2 * np.pi
    for i in range(n):
        for j in range(n):
            for k in range(n):
                sum = sum + 0.25*(a+(i * delt))**3 * (a+(j * delt))**3 * gamma4_weak_final( l+(i * delt), l+(j * delt), k * (b / n), 0,
                                                                          quick_channel.path.length,
                                                                          quick_channel.source.w0 ** 2,
                                                                          quick_channel.source.F0,
                                                                          quick_channel.source.k,
                                                                          quick_channel.path.phase_screen.model.l0,
                                                                          quick_channel.path.phase_screen.model.L0,
                                                                          quick_channel.path.phase_screen.model.Cn2) * delt ** 2 * (b / n) * (2 * np.pi)

    
    print(sum)
    """


    def gamma4_weak(x1, y1, x2, y2, L, S0, F, k, l0, L0, Cn2, **params):
        r1 = np.array([x1, y1])
        r2 = np.array([x2, y2])
        r = 1 / 2 * (r1 + r2)
        p = r1 - r2
        rho = np.linalg.norm(p)

        _S = S(L, S0, F, k)
        theta_0 = 1 - L / F
        Lambda_0 = 2 * L / k / S0
        Theta_bar = 1 - theta_0 / (theta_0 ** 2 + Lambda_0 ** 2)
        Lambda = Lambda_0 / (theta_0 ** 2 + Lambda_0 ** 2)

        def integrand(kappa, xi):
            _real1_abs_arg = (1 - Theta_bar * xi) * p - 2j * Lambda * xi * r
            _real1_abs = np.sqrt(sum(_real1_abs_arg ** 2))
            _real1 = scipy.special.jn(0, kappa * _real1_abs)

            _real2_exp = -1j * L * kappa ** 2 / k * xi * (1 - Theta_bar * xi)
            _real2_J0 = (1 - Theta_bar * xi - 1j * Lambda * xi) * kappa * rho
            _real2 = np.exp(_real2_exp) * scipy.special.jn(0, _real2_J0)

            _real_part = np.real(_real1 - _real2)
            _exp_arg = -Lambda * L * xi ** 2 * kappa ** 2 / k
            return kappa * phi_n(kappa, l0, L0, Cn2) * np.exp(_exp_arg) * _real_part

        _const = 8 * np.pi ** 2 * k ** 2 * L
        B_I = _const * scipy.integrate.dblquad(integrand, 0, 1, 0, np.inf)[0]

        gamma2_r1 = gamma2_weak(np.linalg.norm(r1), L, S0, F, k, l0, L0, Cn2)
        gamma2_r2 = gamma2_weak(np.linalg.norm(r2), L, S0, F, k, l0, L0, Cn2)
        return gamma2_r1 * gamma2_r2 * (1 + B_I)

    """"""
    _r = np.linspace(0, 0.15, 25)
    _f = np.linspace(0, 2*np.pi, 10)
    _R1, _R2, _F = np.meshgrid(_r, _r, _f)

    _res = []
    for i, _ in np.ndenumerate(_R1):
        r1, r2, f = _R1[i], _R2[i], _F[i]
        x1, y1, x2, y2 = (r1, 0, r2 * np.cos(f), r2 * np.sin(f))
        _res.append( r1**3 * r2**3 *(0.25+0.125*np.cos(2*f))* gamma4_weak(x1, y1, x2, y2, quick_channel.path.length,
                                                                          quick_channel.source.w0 ** 2,
                                                                          quick_channel.source.F0,
                                                                          quick_channel.source.k,
                                                                          quick_channel.path.phase_screen.model.l0,
                                                                          quick_channel.path.phase_screen.model.L0,
                                                                          quick_channel.path.phase_screen.model.Cn2))
    sum=2 * np.pi * np.asarray(_res).sum() * (_r[1] - _r[0]) ** 2 * (_f[1] - _f[0])
    print(sum)

    longterm=W2_weak(quick_channel.path.length, quick_channel.source.w0**2, quick_channel.source.F0, quick_channel.source.k,
                    quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                    quick_channel.path.phase_screen.model.Cn2)
    x2_0=x2_0_weak(quick_channel.path.length, quick_channel.source.w0 ** 2, quick_channel.source.F0,
                  quick_channel.source.k,
                  quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                  quick_channel.path.phase_screen.model.Cn2)

    print("S**2=",16*(sum-0.5*longterm*x2_0-3*x2_0**2))


    #for strong

    #x2_0 strong

    def x2_0_strong(L, S0, F, k, l0, L0,Cn2):
        def integr1(kappa, z,L, S0, F, k, l0, L0,Cn2):
            return ( (L-z)**2 * kappa**3 * phi_n(kappa,l0, L0,Cn2)*
                    np.exp(-kappa**2*S(z, S0, F,k)/4- np.pi*D_sp(kappa*z/k, L, k, l0, L0,Cn2)) )
        return 2*np.pi**2*sp.integrate.dblquad(integr1, 0, L, 0, np.inf,args=(L, S0, F, k, l0, L0,Cn2))[0]


    """
    print("analytical strong x2_0=",x2_0_strong(quick_channel.path.length, quick_channel.source.w0**2, quick_channel.source.F0, quick_channel.source.k,
                    quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                    quick_channel.path.phase_screen.model.Cn2))
    """

    # W2 strong

    def gamma2_strong(r, L, S0, F, k, l0, L0,Cn2):
        def integr(Q, r, L, S0, F, k, l0, L0,Cn2):
            Lambd=2*L/k/S(L, S0, F,k)
            return Q* scipy.special.jn(0, k*r*Q/L) * np.exp(-k*Q**2/4/L/Lambd-0.5*D_sp(Q, L, k, l0, L0,Cn2))
        return (2/np.pi/S0)*(k**2*S0/4/L**2) * sp.integrate.quad(integr, 0, np.inf, args=(r, L, S0, F, k, l0, L0,Cn2))[0]

    """
    g=1.2
    x = np.linspace(0, g, 100)
    y = [gamma2_strong(i, quick_channel.path.length, quick_channel.source.w0**2, quick_channel.source.F0, quick_channel.source.k,
                    quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                    quick_channel.path.phase_screen.model.Cn2) for i in x]
    plt.plot(x, y)
    
    z=0
    for i in range(100):
        z=z+(i*g/100)*y[i]*(g/100) *(2*np.pi)
    print(z)
    """





    def W2_strong(L, S0, F, k, l0, L0, Cn2):
        def integr(r, L, S0, F, k, l0, L0, Cn2):
            return np.pi * r ** 3 * gamma2_strong(r, L, S0, F, k, l0, L0, Cn2)
        right=1.8
        # right limit for integration of W2 strong is NOT equal to right limit for gama2 strong
        # Here right limit must be evaluated from graph of (np.pi * r ** 3 * gamma2_strong(r, L, S0, F, k, l0, L0, Cn2))
        return 4 * ( sp.integrate.quad(integr, 0, right, args=(L, S0, F, k, l0, L0, Cn2))[0] - x2_0_strong(L, S0, F, k, l0,L0, Cn2))

    """

    g=1.5
    n=100
    # right limit for integration of W2 strong is NOT equal to right limit for gama2 strong
    # Here right limit must be evaluated from graph of (np.pi * r ** 3 * gamma2_strong(r, L, S0, F, k, l0, L0, Cn2))
    x = np.linspace(0, g, n)
    y = [np.pi*i**3*gamma2_strong(i, quick_channel.path.length, quick_channel.source.w0**2, quick_channel.source.F0, quick_channel.source.k,
                    quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                    quick_channel.path.phase_screen.model.Cn2) for i in x]
    plt.plot(x, y)
    
    # somehow it is important to calculate it in separate cycles,
    # if you calculate it within one cycle it will give wrong values
    sum=0
    for i in range(n):
        sum=sum+y[i]*(g/n)

    S_strong=4*(sum-x2_0_strong(quick_channel.path.length, quick_channel.source.w0**2, quick_channel.source.F0, quick_channel.source.k,
                    quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                    quick_channel.path.phase_screen.model.Cn2))
    print("analytical strong S =",S_strong)

    """

    



    # W4 strong Andrews page  364 , formula 125 , modified
    def gamma4_strong(r1, r2, a1, a2, rytov2,k,L):
        # Compute rho
        rho = np.sqrt(r1**2+r2**2-2*r1*r2*np.cos(a1-a2))

        # Compute p1, p2, eta1, eta2
        p1 = 1.37 / rytov2 ** (2 / 5)
        p2 = np.log(2)
        eta_x = 8.56 / (1 + 0.19 * rytov2 ** (6 / 5))
        eta_y = 9 * (1 + 0.23 * rytov2 ** (6 / 5))


        term_1 = p1 * mpmath.hyper([7 / 6, 3 / 2, 2], [7 / 2, 3, 1], -k * rho **2* eta_x / (4 * L))
        term_2 = p2 * mpmath.hyp1f2(1 / 2, 1 / 6, 3 / 2, k * rho **2 * eta_y / (4 * L))
        term_3 = -2.22 * p2 * (k * rho**2 * eta_y / (4 * L)) ** (5 / 6) * mpmath.hyp1f2(4 / 3, 11 / 6, 7 / 3,
                                                                              k * rho**2 * eta_y / (4 * L))

        x=mpmath.exp(term_1 + term_2 + term_3) - 1


        return x

    def gamma4_strong_final(r1, r2, a1, a2, rytov2,k, L, S0, F, l0, L0, Cn2):
        return gamma2_strong(r1,L, S0, F, k, l0, L0,Cn2)*gamma2_strong(r2,L, S0, F, k, l0, L0,Cn2)*(1+gamma4_strong(r1, r2, a1, a2, rytov2,k,L))


    """
    x = np.linspace(0,  0.9,100)
    y = [gamma4_strong(i, 0,0,0, quick_channel.get_rythov2(),quick_channel.source.k,quick_channel.path.length) for i in x]

    plt.plot(x, y)


    
    sum=0
    n=10
    a=0.9
    b=2*np.pi
    for i in range(n):
        for j in range(n):
            for k in range(n):
                sum=sum+(i*a/n)*(j*a/n)*gamma4_strong_final(i*a/n, j*a/n ,k*b/n,0, quick_channel.get_rythov2(),quick_channel.source.k,quick_channel.path.length,
                                        quick_channel.source.w0 ** 2, quick_channel.source.F0,
                                         quick_channel.path.phase_screen.model.l0,
                                         quick_channel.path.phase_screen.model.L0,
                                         quick_channel.path.phase_screen.model.Cn2)*(a/n)**2* (b/n) * b
        print(i)

    print(sum)

    """

    def gamma4_strong_chumak(r1, r2, a1, a2, L, k, l0, L0,Cn2):


        rho2 = r1**2+r2**2-2*r1*r2*np.cos(a1-a2)
        R2 = r1**2+r2**2+2*r1*r2*np.cos(a1-a2)

        def integ(kappa, l0, L0, Cn2):
            return kappa**2 *phi_n(kappa,l0, L0,Cn2)
        temp=sp.integrate.quad(integ, 0, np.inf, args=(l0, L0,Cn2))[0]

        d1=(4/3)*L**3*np.pi*temp


        answ=(1/np.pi**2/d1**2)*np.exp(-(r1**2+r2**2)/d1)+(1/np.pi**2/d1**2)*np.exp(-R2/2/d1-rho2*d1*(3/8)*k**2/L**2)
        return answ*(1+(32/3)*L**2/k**2/d1**2)**(-1)


    """
    x = np.linspace(0,  0.6,500)
    y = [gamma4_strong_chumak(i, i,0,0, quick_channel.path.length,quick_channel.source.k,
                    quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                    quick_channel.path.phase_screen.model.Cn2) for i in x]

    plt.plot(x, y)
    """
    """
    sum=0
    n=25
    a=0.6
    b=2*np.pi
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    sum=sum+(i*a/n)*(j*a/n)*gamma4_strong_chumak(i*a/n, j*a/n, k*(b/n), l*(b/n), quick_channel.path.length,                                          quick_channel.source.k,
                                         quick_channel.path.phase_screen.model.l0,
                                         quick_channel.path.phase_screen.model.L0,
                                         quick_channel.path.phase_screen.model.Cn2)*(a/n)**2*(b/n)**2

    print(sum)
    
    
    
    x = np.linspace(0,  1,100)
    y = [gamma4_strong_chumak(i, i,0,0, quick_channel.path.length ,quick_channel.source.k,
                    quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                    quick_channel.path.phase_screen.model.Cn2) for i in x]

    z = [gamma2_strong(i, quick_channel.path.length, quick_channel.source.w0 ** 2, quick_channel.source.F0,
                       quick_channel.source.k,
                       quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
                       quick_channel.path.phase_screen.model.Cn2)**2 for i in x]

    plt.plot(x, y,'red', x,z, 'blue')
    """


    def integral_x1_2_x2_2_gamma4_chumak(L, k, l0, L0,Cn2):
        def integ(r1, r2,  L, k, l0, L0,Cn2):
            def integ1(r1, r2,  L, k, l0, L0,Cn2):
                def integ2(a1, a2,r1, r2,  L, k, l0, L0,Cn2):
                    return gamma4_strong_chumak(r1, r2, a1, a2, L, k, l0, L0,Cn2)
                return sp.integrate.dblquad(integ2, 0, 2*np.pi, 0, 2*np.pi, args=(r1, r2,  L, k, l0, L0,Cn2))[0]
            return 0.25*r1**3* r2**3* integ1(r1, r2,  L, k, l0, L0,Cn2)
        return sp.integrate.dblquad(integ, 0, np.inf, 0, np.inf, args=(L, k, l0, L0,Cn2))




    """

    sum = 0
    n = 60
    a = 0.6
    b = 2 * np.pi
    for i in range(n):
        for j in range(n):
            for k in range(n):
                sum = sum + 0.25*(i * a / n)**3 * (j * a / n)**3 * gamma4_strong_chumak(i * a / n, j * a / n, k * (b / n), 0,
                                                                             quick_channel.path.length,
                                                                             quick_channel.source.k,
                                                                             quick_channel.path.phase_screen.model.l0,
                                                                             quick_channel.path.phase_screen.model.L0,
                                                                             quick_channel.path.phase_screen.model.Cn2) * (
                              a / n) ** 2 * (b / n) * (2 * np.pi)



    print(sum)
    
    
    omeg=quick_channel.source.k*quick_channel.source.w0**2/2/quick_channel.path.length
    gam=1+omeg**(-2)
    int=(gam**2*quick_channel.source.w0**4/16)+(4.34*gam*quick_channel.source.w0**4*quick_channel.get_rythov2()**(6/5)/omeg)


    
    # here next S_strong is not function value, but value of sum, so section with its calculation must be uncommented

    #S=S_strong
    #x2_0=x2_0_strong(quick_channel.path.length, quick_channel.source.w0**2, quick_channel.source.F0, quick_channel.source.k,
    #                quick_channel.path.phase_screen.model.l0, quick_channel.path.phase_screen.model.L0,
    #                quick_channel.path.phase_screen.model.Cn2)
    #S=0.2565706427256079
    #x2_0=0.012210410003401697
    S=0.20327090753112367
    x2_0=0.010016363081098788
    




    print("strong S**2=",16*(int-0.5*S*x2_0-3*x2_0**2))

    """





    def S2_strong_Vasiliev(x,y,x1, y1, x2, y2, L, S0, k,  Cn2, **params):
            r = np.array([x, y])
            r1 = np.array([x1, y1])
            r2 = np.array([x2, y2])

            r1_n = np.linalg.norm(r1)
            r2_n = np.linalg.norm(r2)

            omeg = k * S0 / 2 / L
            gam = 1 + omeg ** (-2)
            #rho0=(1.5*Cn2*k**2*L)**(-3/5)
            rho53=1.5*Cn2*k**2*L


            part0 = (0.75 * gam ** 2 * S0 ** 2 - gam * S0 * r[0] ** 2 + r[0] ** 4) * np.exp(
                -(r1_n ** 2 + r2_n ** 2) / S0 + 2 * 1j * omeg * (np.dot(r1, r2) - np.dot(r, r2)) / S0)

            def integrand1(ksi):
                return (np.linalg.norm(r * ksi + (r1 + r2) * (1 - ksi)) ** (5 / 3) +
                        np.linalg.norm(r * ksi + (r1 - r2) * (1 - ksi)) ** (5 / 3))

            int1 = sp.integrate.quad(integrand1, 0, 1)[0]


            def integrand2(ksi):
                return np.linalg.norm(r * ksi + r1 * (1 - ksi)) ** (5 / 3)

            int2 = sp.integrate.quad(integrand2, 0, 1)[0]

            part1 = np.exp(-rho53 * int1) * (1 + 0.75 * rho53 * r2_n ** (5 / 3) - 2 * rho53 * int2)
            part2 = np.exp(-2 * rho53 * int2) * (1 + 0.75 * rho53 * r2_n ** (5 / 3) - rho53 * int1)

            def integrand3(ksi):
                return (np.linalg.norm(r * ksi + (r1 + r2) * (1 - ksi)) ** (5 / 3) +
                        np.linalg.norm(r * ksi + (r1 - r2) * (1 - ksi)) ** (5 / 3) +
                        2 * np.linalg.norm(r * ksi + r1 * (1 - ksi)) ** (5 / 3))

            int3 = sp.integrate.quad(integrand3, 0, 1)[0]

            part3 = np.exp(-rho53 * int3) * (1 + 0.75 * rho53 * r2_n ** (5 / 3))



            return part0 * (part1 + part2 - part3) * (omeg**2/2/(2*np.pi)**3/S0**3)


    """
    sum = 0
    n = 10
    a = 0.6

    for i1 in range(n):
        for i2 in range(n):
            for j1 in range(n):
                for j2 in range(n):
                    for k1 in range(n):
                        for k2 in range(n):
                            sum=sum+S2_strong_Vasiliev(-a+i1*(2*a/n), -a+i2*(2*a/n), -a+j1*(2*a/n), -a+j2*(2*a/n),
                                                       -a+k1*(2*a/n), -a+k2*(2*a/n), quick_channel.path.length,
                                                       quick_channel.source.w0**2, quick_channel.source.k,quick_channel.path.phase_screen.model.Cn2)
    print(sum)
    #print(S2_final(quick_channel.path.length, quick_channel.source.w0**2, quick_channel.source.k,quick_channel.path.phase_screen.model.Cn2))

    """





    print()





# ------------------
now = datetime.datetime.now()
delta = now - then
print('Time', delta.seconds / 60, 'minute')

plt.show()
