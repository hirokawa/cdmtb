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
    Bp : transfer function or list of transfer function
        denominator part of transfer function for plant
    nc : int
        order of numerator part of controller Ac(s)
    mc : int or array of int
        order of denominator part of controller Bc(s)
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
    Bc : transfer function or list of transfer function
        denominator part of transfer function for controller
    """

    s = ct.tf('s')

    ar = aref(gr, taur)
    ng = len(gr)

    if type(Ap) is int:
        Ap = s*0+Ap

    if type(Bp) is int:
        Bp = s*0+Bp

    if type(Bp) is list:
        mp_ = []
        mpd = len(Bp)
        for k in range(mpd):
            if type(Bp[k]) is int:
                Bp[k] = s*0+Bp[k]
            cf_bp = Bp[k].num[0][0]
            mp_.append(len(cf_bp)-1)
    else:
        mpd = 1
        cf_bp = Bp.num[0][0]
        mp_ = len(cf_bp)-1

    cf_ap = Ap.num[0][0]
    np_ = len(cf_ap)-1

    if mpd == 1:
        mct = mc
        mcd = 1
    else:
        if len(mc) != mpd:
            print("dimension mismatch for Bp and Bc.")
        mct = sum(mc)
        mcd = len(mc)

    nk = (nc+1) + (mct+mcd)  # number of parameters
    nt = max(np_+nc, np.max(np.array(mp_)+np.array(mc)))

    M = np.zeros((nk, nt+1))

    # sylvester equation
    for k in range(nc+1):
        M[k, k:k+np_+1] = cf_ap

    if mpd == 1:
        for k in range(mc+1):
            i0 = k+nt-(mc+mp_)
            M[k+nc+1, i0:i0+mp_+1] = cf_bp
    else:
        k0 = 0
        for j in range(mpd):
            cf_bp = Bp[j].num[0][0]
            for k in range(0, mc[j]+1):
                i0 = k+nt-(mc[j]+mp_[j])
                M[k+k0+nc+1, i0:i0+mp_[j]+1] = cf_bp
            k0 += mc[j]+1

    i0 = nt+1-nk
    k0 = ng+2-nk

    if k0 > 0 and i0 >= k0:
        M = M[:, i0-1:i0+nk]
        ak = ar[k0-1:k0+nk]
    else:
        M = M[:, i0:i0+nk]
        ak = ar[k0:k0+nk]

    d = np.linalg.lstsq(M.T, ak, rcond=None)[0]
    # d = np.linalg.solve(M.T, ak)

    # normalize l_nd in controller
    l0 = d[nc-nd]
    Ac = ct.tf(d[0:nc+1]/l0, [1])

    if mpd == 1:
        Bc = ct.tf(d[nc+1:]/l0, [1])
        P = Ap*Ac + Bp*Bc
    else:
        P = Ap*Ac
        Bc = []
        i0 = nc+1
        for j in range(mpd):
            Bc.append(ct.tf(d[i0:i0+mc[j]+1]/l0, [1]))
            i0 += mc[j]+1
            P += Bc[j]*Bp[j]

    return P, Ac, Bc


def g2t(Ap, Bp, nc, mc, gr, tau_max=1e3):
    """Returns reference time constant to realized stability index

    Parameters:
    ----------
    Ap : transfer function
        numerator part of transfer function for plant
    Bp : transfer function
        denominator part of transfer function for plant
    nc : int
        order of numerator part of controller Ac(s)
    mc : int or array of int
        order of denominator part of controller Bc(s)
    gr : array
        reference stability index

    Returns:
    ----------
    tau : array
        candidates of reference time constant
    """

    s = ct.tf('s')
    ng = len(gr)

    if type(Ap) is int:
        Ap = s*0+Ap

    if type(Bp) is int:
        Bp = s*0+Bp

    if type(Bp) is list:
        mp_ = []
        mpd = len(Bp)
        for k in range(mpd):
            if type(Bp[k]) is int:
                Bp[k] = s*0+Bp[k]
            cf_bp = Bp[k].num[0][0]
            mp_.append(len(cf_bp)-1)
    else:
        mpd = 1
        cf_bp = Bp.num[0][0]
        mp_ = len(cf_bp)-1

    cf_ap = Ap.num[0][0]
    np_ = len(cf_ap)-1

    if mpd == 1:
        mct = mc
        mcd = 1
    else:
        if len(mc) != mpd:
            print("dimension mismatch for Bp and Bc.")
        mct = sum(mc)
        mcd = len(mc)

    nk = (nc+1) + (mct+mcd)  # number of parameters
    nt = max(np_+nc, np.max(np.array(mp_)+np.array(mc)))

    ncp = nk+1  # number of coefficient based on parameters
    ncg = ng+2  # number of coefficient based on gammas

    if ncg < ncp:
        print(f"Error: ncg ({ncg})< ncp ({ncp})")
        return None

    if nt < nk:
        print(f"Error: nt ({nt})< nk ({nk})")
        return None

    v = np.cumprod(np.cumprod(gr))
    eta = 1/np.flip(v)

    aref_n = np.r_[eta, 1, 1]
    aref_p = aref_n[ncg-ncp:ncg]

    M = np.zeros((nk, ncp))

    for k in range(nc+1):
        M[k, k:k+np_+1] = cf_ap

    if mpd == 1:
        for k in range(mc+1):
            i0 = k+nt-(mc+mp_)
            M[k+nc+1, i0:i0+mp_+1] = cf_bp
    else:
        k0 = 0
        for j in range(mpd):
            cf_bp = Bp[j].num[0][0]
            for k in range(0, mc[j]+1):
                i0 = k+nt-(mc[j]+mp_[j])
                M[k+k0+nc+1, i0:i0+mp_[j]+1] = cf_bp
            k0 += mc[j]+1

    Mc = M[:, nt-nk:nt+1]
    b = np.r_[np.zeros(nk), 1]
    ppd = np.r_[1, np.zeros(nk-2), 1, 0]
    Mce = np.r_[Mc, [ppd]]

    # c = np.linalg.lstsq(Mce, aac, rcond=None)[0]
    c = np.linalg.solve(Mce, b)
    v = aref_p*c
    tau = np.roots(v)
    tau = tau[(tau > 0.0) & (tau < tau_max)]
    return tau
