"""

Example of ACC two-mass-spring problem
 from section 4.3 of CDM Book [1]

[1] Shunji Manabe, Young Chol Kim, Coefficient Diagram Method
for Control System Design, Springer, 2021

"""
import numpy as np
import control as ct
from cdmtb import cdia, g2c, g2t
import matplotlib.pyplot as plt

s = ct.tf('s')


def lim_(u):
    y = np.min([2.0, np.max([-2.0, u])])
    return y


def ulim_(u):
    y = np.min([2.0, u])
    return y


def is_stable(m1=1.0, m2=1.0, k=1.0):
    Ap = m1*m2/k*s**4+(m1+m2)*s**2
    P = Ac*Ap+Bc*Bp
    if not np.all(np.real(np.roots(P.num[0][0])) < 0):
        return False
    return True


def get_index(P, Ap, Bp, Bd, Ac, Bc, Ba, show_plt=False):

    k3, k2, k1, k0 = Bc.num[0][0]
    l3, l2, l1, l0 = Ac.num[0][0]

    KF = np.array([k3/l2, k2])

    Gop = Bc*Bp/(Ac*Ap)

    GM, PM, _, _, _, _ = ct.stability_margins(Gop)
    GM = 20*np.log10(GM)

    tin = np.arange(0, 40, 0.01)

    # w1,w2 to y (x2) for ts
    sys_cl = ct.append(Bp/P, Bd/P)
    # ct.impulse_response(sys_cl).plot()
    yresp = ct.impulse_response(sys_cl, T=tin)

    yr1 = yresp.y[0, 0]
    imax = np.argmax(np.abs(yr1))
    tmax = yresp.t[imax]
    ts1 = yresp.t[(yresp.t > tmax) & (np.abs(yr1) < 0.1)][0]

    yr2 = yresp.y[1, 1]
    imax = np.argmax(np.abs(yr2))
    tmax = yresp.t[imax]
    ts2 = yresp.t[(yresp.t > tmax) & (np.abs(yr2) < 0.1)][0]

    ts = np.max([ts1, ts2])

    # w1, w2 to u for umax
    sys_cl = ct.append(-Bc*Bp/P, -Bc*Bd/P)
    # ct.impulse_response(sys_cl).plot()
    uresp = ct.impulse_response(sys_cl, T=tin)
    umax = np.max(np.abs(uresp.y[1, 1]))

    cdia(P, opt_p=[Ac*Ap, Bc*Bp], leg_opt_p=['$A_cA_p$', '$B_cB_p$'])

    if show_plt:
        fig = plt.figure(figsize=[10, 8])
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(yresp.t, yresp.y[0][0])
        ax1.set_title('impulse $w_1$ to $x_2$')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(yresp.t, yresp.y[1][1])
        ax2.set_title('impulse $w_2$ to $x_2$')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(uresp.t, uresp.y[0][0])
        ax3.set_title('impulse $w_1$ to $u$')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(uresp.t, uresp.y[1][1])
        ax4.set_title('impulse $w_2$ to $u$')

    for k in np.arange(1, 10, 0.01):
        if not is_stable(k=k):
            kmax = k - 0.01
            break

    for k in np.arange(1, 0, -0.01):
        if not is_stable(k=k):
            kmin = k + 0.01
            break

    for r in np.arange(0, 1, 0.01):
        if not is_stable(m1=1.0+r, m2=1.0+r, k=1.0+r) or \
           not is_stable(m1=1.0+r, m2=1.0+r, k=1.0-r) or \
           not is_stable(m1=1.0+r, m2=1.0-r, k=1.0-r) or \
           not is_stable(m1=1.0+r, m2=1.0-r, k=1.0+r) or \
           not is_stable(m1=1.0-r, m2=1.0+r, k=1.0+r) or \
           not is_stable(m1=1.0-r, m2=1.0-r, k=1.0+r) or \
           not is_stable(m1=1.0-r, m2=1.0+r, k=1.0-r) or \
           not is_stable(m1=1.0-r, m2=1.0-r, k=1.0-r):
            pm = r-0.01
            break

    bonus = 2.0
    if PM < 30.0:
        bonus = 0
    if GM < 6.0:
        bonus = 0
    if ts > 15.0:
        bonus = 0
    if umax > 1.0:
        bonus = 0
    if pm < 0.3:
        bonus = 0
    if 20.0*np.log10(kmax/kmin) < 12.0:
        bonus = 0

    FM = [PM, GM, ts, umax, kmin/kmax, pm]
    score = lim_((PM-30)/5)+lim_((GM-6)/2)+lim_((15-ts)/3) + \
        lim_((pm-0.3)/0.05)+ulim_((-20*np.log10(umax)/3)) + \
        lim_((20*np.log10(kmax/kmin)-12)/3)+bonus

    return FM, KF, score


m1 = m2 = 1.0
k = 1

Ap = m1*m2/k*s**4+(m1+m2)*s**2
Bp = 1+s*0
Bd = m1/k*s**2+1

nc = 3  # Ac=l3*s^3+l2*s^2+l1*s+l0
mc = 3  # Bc=k3*s^3+k2*s^2+k1*s+k0

# CDM-1 score= -1.41
gr, taur = [2.5, 2, 2, 2, 2, 2], 6.0

P, Ac, Bc = g2c(Ap, Bp, nc, mc, gr, taur)
Ba = Bc.num[0][0][-1]
FM, KF, score = get_index(P, Ap, Bp, Bd, Ac, Bc, Ba)

print(f"CDM-1 FM={FM}, KF={KF} score={score}")
print(f"Ac={Ac} Bc={Bc}")

# CDM-2 score = 0.35
gr, taur = [2.5, 2, 1.5, 1.5, 1.5, 2], 6.0

P, Ac, Bc = g2c(Ap, Bp, nc, mc, gr, taur)
Ba = Bc.num[0][0][-1]
FM, KF, score = get_index(P, Ap, Bp, Bd, Ac, Bc, Ba)

print(f"CDM-2 FM={FM}, KF={KF} score={score}")
print(f"Ac={Ac} Bc={Bc}")

# CDM-3 score = 5.99
gr, taur = [2, 2.5, 1.5, 1.5, 1.5, 4], 6.0

P, Ac, Bc = g2c(Ap, Bp, nc, mc, gr, taur)
Ba = Bc.num[0][0][-1]
FM, KF, score = get_index(P, Ap, Bp, Bd, Ac, Bc, Ba)

print(f"CDM-3 FM={FM}, KF={KF} score={score}")
print(f"Ac={Ac} Bc={Bc}")

# CDM-4 score 8.54 (not reproduced) => 6.70
# gr, taur = [2.0, 2.5, 1.5, 1.5, 1.5, 4], 6.4

# CDM-4 (modified) score = 7.63
gr, taur = [2.0, 2.5, 1.5, 1.5, 1.5, 6], 6.05

P, Ac, Bc = g2c(Ap, Bp, nc, mc, gr, taur)
Ba = Bc.num[0][0][-1]
FM, KF, score = get_index(P, Ap, Bp, Bd, Ac, Bc, Ba, show_plt=True)

print(f"CDM-4 FM={FM}, KF={KF} score={score}")
print(f"Ac={Ac} Bc={Bc}")

Gop = Bc*Bp/(Ac*Ap)
S = Ac*Ap/P
T = Bc*Bp/P
ct.bode_plot(Gop, dB=True, label='G(s)', display_margins=True)
ct.bode_plot(T, dB=True, label='T(s)')
ct.bode_plot(S, dB=True, label='S(s)')
