"""
Coefficient Diagram Method (CDM) Toolbox by Python

@author: Rui Hirokawa

[1] Shunji Manabe, Young Chol Kim, 
Coefficient Diagram Method for Control System Design, 
Springer, 2021
"""

import numpy as np
import control as ct
import matplotlib.pyplot as plt


def get_coeff(F):
    """get coefficient and non-zero index"""
    F_ = F.num[0][0]
    idx = np.arange(len(F_)-1, -1, -1)
    idx_z = np.where(F_ == 0)[0]
    if len(idx_z) > 0:
        F_ = np.delete(F_, idx_z)
        idx = np.delete(idx, idx_z)

    return F_, idx


def get_idx(P):
    """Returns coefficients, stability index, time constant"""
    a = P.num[0][0]
    nt = len(a)

    gam = np.zeros(nt-2)
    for i in range(nt-2):
        gam[-1-i] = a[i+1]**2/(a[i]*a[i+2])
    tau = a[-2]/a[-1]

    if nt > 3:
        gams = np.zeros(nt-2)
        gams[0] = 1/gam[1]
        gams[-1] = 1/gam[-2]

        for i in range(1, nt-3):
            gams[i] = 1/gam[i-1]+1/gam[i+1]
    else:
        gams = []

    return a, gam, tau, gams


def c2g(Ap, Ac, Bp, Bc, Ba=None):
    """Returns coefficients, stability index, time constant

    Parameters:
    ----------
    Ap : transfer function
        numerator part of transfer function for plant
    Bp : transfer function
        denominator part of transfer function for plant
    Ac : transfer function
        numerator part of transfer function for controller
    Bc : transfer function
        denominator part of transfer function for controller
    Ap : transfer function
        numerator part of transfer function for pre-filter

    Returns:
    ----------
    a : array
        coefficient of characsteric polynomials
    gam : array
        stability index
    tau : float
        time constant
    gams : array
        limit of stability index

    """
    P = Ap*Ac+Bp*Bc
    a, gam, tau, gams = get_idx(P)

    return a, gam, tau, gams


def cdia(P, opt_p=[], leg_opt_p=[]):
    """Plot coefficient diagram

    Parameters:
    ----------
    P : transfer function
        characsteric polynomial.
    opt_p : array of transfer function (option)
        list of transfer function to be shown in CDM.
    leg_opt_p : array of string (option)
        list of name of transfer function.

    """
    a, gam, tau, gams = get_idx(P)

    nt = len(a)
    idx = np.arange(nt-1, -1, -1)

    grp, ax = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[2, 1]))

    ax[0].semilogy(idx, a, 'b.-', label='P(s)')
    ax[0].set_xlim([-0.2, nt-0.8])
    ax[0].set_xticks(np.arange(0, nt))
    ax[0].invert_xaxis()
    ax[0].grid()
    ax[0].set_ylabel('coefficient')
    ax[0].set_title('$\\tau$={:.3f}[s]'.format(tau))

    ax[1].plot(np.arange(1, nt-1), gam, 'b.-', label='$\\gamma$')
    if len(gams) > 0:
        ax[1].plot(np.arange(1, nt-1), gams, 'r.--', label='$\\gamma^*$')
    ax[1].set_xlim([-0.2, nt-0.8])
    ax[1].set_xticks(np.arange(0, nt))
    ax[1].invert_xaxis()
    ax[1].set_ylim([0, np.ceil(np.max(gam)+1)])
    ax[1].grid()
    ax[1].set_ylabel('stability index')

    for i, opt in enumerate(opt_p):
        p_, idx_ = get_coeff(opt)
        idx = np.where(p_ < 0)[0]
        ax[0].semilogy(idx_, np.abs(p_), 'x--', label=leg_opt_p[i])
        if len(idx) > 0:
            ax[0].plot(idx_[idx], np.abs(p_[idx]), 'v', color='red')

    ax[0].legend()
    ax[1].legend()

    grp.tight_layout()
    plt.show()


def aref(gr, taur):
    """Returns coefficients of reference characteristic polynomial

    Parameters:
    ----------
    gr : array
        reference stability index
    taur : array
        candidates of reference time constant

    Returns:
    ----------
    ar: array
        coefficients of reference characteristic polynomial
    """

    ng = len(gr)
    v = np.cumprod(np.cumprod(gr))
    eta = 1/np.flip(v)
    ar = np.r_[taur ** np.arange(ng+1, 1, -1)*eta, taur, 1]
    return ar


def g2c(Ap, Bp, nc, mc, gr, taur, nd=0):
    """Returns controller based on the order of controller.

    Returns coefficients, stability index, time constant

    Parameters:
    ----------
    Ap : transfer function
        numerator part of transfer function for plant
    Bp : transfer function
        denominator part of transfer function for plant
    nc : int
        order of numerator part of controller
    mc : int
        order of denominator part of controller
    gr : array
        stability index
    tau : float
        time constant
    nd : int (option)
        index of denominator of controoler to normalize

    Returns:
    ----------
    P: transfer function
        characteristic polynomials
    Ac : transfer function
        numerator part of transfer function for controller
    Bc : transfer function
        denominator part of transfer function for controller
    """

    s = ct.tf('s')
    ar = aref(gr, taur)

    if type(Ap) is int:
        Ap = s*0+Ap

    if type(Bp) is int:
        Bp = s*0+Bp

    cf_ap = Ap.num[0][0]
    cf_bp = Bp.num[0][0]

    np_ = len(cf_ap)-1
    mp_ = len(cf_bp)-1

    nk = nc + mc + 2  # number of parameters
    nt = max(np_+nc, mp_+mc)
    M = np.zeros((nk, nt+1))

    # sylvester equation
    for k in range(nc+1):
        M[k, k:k+np_+1] = cf_ap

    for k in range(mc+1):
        i0 = k+nt-(mc+mp_)
        M[k+nc+1, i0:i0+mp_+1] = cf_bp

    i0 = nt+1-nk
    M = M[:, i0:i0+nk]
    ak = ar[ng+2-nk:ng+2]

    d = np.linalg.lstsq(M.T, ak, rcond=None)[0]

    # normalize l_nd in controller
    l0 = d[nc-nd]
    Ac = ct.tf(d[0:nc+1]/l0, [1])
    Bc = ct.tf(d[nc+1:]/l0, [1])

    P = Ap*Ac+Bp*Bc

    return P, Ac, Bc


def g2t(Ap, Bp, nc, mc, gr):
    """Returns reference time constant to realized stability index

    Parameters:
    ----------
    Ap : transfer function
        numerator part of transfer function for plant
    Bp : transfer function
        denominator part of transfer function for plant
    nc : int
        order of numerator part of controller
    mc : int
        order of denominator part of controller
    gr : array
        reference stability index

    Returns:
    ----------
    tau : array
        candidates of reference time constant
    """

    s = ct.tf('s')

    if type(Ap) is int:
        Ap = s*0+Ap

    if type(Bp) is int:
        Bp = s*0+Bp

    cf_ap = Ap.num[0][0]
    cf_bp = Bp.num[0][0]

    np_ = len(cf_ap)-1
    mp_ = len(cf_bp)-1

    mm = nc+mc+2
    nn = max(np_+nc, mp_+mc)

    ng = len(gr)
    v = np.cumprod(np.cumprod(gr))
    eta = 1/np.flip(v)

    aab = np.r_[eta, 1, 1]
    aacc = aab[ng+1-mm:ng+2]

    pp = np.zeros((mm, nn+1))

    for k in range(nc+1):
        pp[k, k:k+np_+1] = cf_ap

    for k in range(mc+1):
        i0 = k + nn - mc
        pp[k+nc+1, i0:i0+mp_+1] = cf_bp

    ppc = pp[:, nn-mm:nn+1]
    aac = np.r_[np.zeros(mm), 1]
    ppd = np.r_[1, np.zeros(mm-2), 1, 0]
    ppmm = np.vstack((ppc, ppd))
    aacm = np.linalg.lstsq(ppmm, aac)[0]
    v = aacc * aacm
    # aa = ut.tf(v, 1)
    tau = np.roots(v)

    return tau
