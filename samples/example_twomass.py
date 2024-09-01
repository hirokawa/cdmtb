"""

Example of Vibration Suppression Control of Two-Inertia System
 from section 4.2 of CDM Book [1]

[1] Shunji Manabe, Young Chol Kim, Coefficient Diagram Method
for Control System Design, Springer, 2021

"""
import numpy as np
import control as ct
from cdmtb import cdia, g2c, g2t
import matplotlib.pyplot as plt

# from sympy import var
# import sympy.physics.control.lti as lti
# import sympy.physics.control as symcontrol
# import sympy

# var('s omega_s zeta_s kp_s ki_s kd_s')
# var("J_s g_s l_s  mu_s M_s Mgl_s")

# Msys_s = lti.TransferFunction(
#    omega_s**2, s**2 + 2*zeta_s*omega_s*s + omega_s**2, s)  # 規範モデル
# Plant_s = lti.TransferFunction(1, J_s*s**2 + mu_s*s + Mgl_s, s)


def showResult(Ap, Bp, Ac, Bc, Ba):
    P = Bc*Bp + Ac*Ap
    if P.num[0][0][0] < 0:
        Ac, Bc, Ba = -Ac, -Bc, -Ba
        P = Bc*Bp+Ac*Ap

    print(f"Bc={Bc}")
    print(f"Ac={Ac}")

    # plot CDM
    opt_p = [Bc*Bp, Ac*Ap]
    leg_opt_p = ['$B_cB_p$', '$A_cA_p$']
    cdia(P, opt_p, leg_opt_p)

    # plot closed-loop step response
    sys_cl = Ba*Bp/P
    sys_cl = ct.append(Ba*Bp/P, Ba*Ko*wa**2/P)
    t, y = ct.step_response(sys_cl, T=np.linspace(0, 0.15, num=100))

    plt.plot(t, y[0, 0, :], 'r-', label='$\\omega_M$')
    plt.plot(t, y[1, 1, :], 'b--', label='$\\omega_L$')
    plt.legend()
    plt.ylabel('Response')
    plt.xlabel('time')
    plt.show()


s = ct.tf('s')

JM = 4.016e-3
JL = 2.921e-3
Ks = 39.21

wr = np.sqrt(Ks/JL*(1+JL/JM))
wa = np.sqrt(Ks/JL)
Ko = 1/JM

Bp = Ko*(s**2+wa**2)
Ap = s*(s**2+wr**2)

gr = np.array([2.5, 2.0, 2.0])
# taur = 0.0305

# controller parameters
nc = 0  # Ac = l0 (=1)
mc = 2  # Bc = k2s^2+k1s+k0

# controller gain calculation

tau_ = g2t(Ap*s, Bp, nc, mc, gr)

if True:
    taur = tau_[0]
    P, Ac, Bc = g2c(Ap*s, Bp, nc, mc, gr, taur)
    Ac = Ac*s
    Ba = Bc.num[0][0][-1]
else:  # analytic solution
    taur = gr[0]*np.sqrt(gr[1])/wa

    tmp_ = gr[0]*gr[1]*gr[2]-gr[0]-gr[2]
    Ki = JL*wa**2*gr[2]/tmp_
    Kd = (JL*gr[0])/tmp_-1/Ko
    Kp = JL*wa*gr[0]*np.sqrt(gr[2])*gr[2]/tmp_

    Bc = Kd*s**2+Kp*s+Ki
    Ba = Ki
    Ac = s

showResult(Ap, Bp, Ac, Bc, Ba)

if True:
    nc, mc = 1, 1
    gr = [2.5, 2, 2]
    tau_ = g2t(Ap, Bp, nc, mc, gr)

    P, Ac, Bc = g2c(Ap, Bp, nc, mc, gr, tau_[0])
    Ba = Bc.num[0][0][1]
    showResult(Ap, Bp, Ac, Bc, Ba)

    P, Ac, Bc = g2c(Ap, Bp, nc, mc, gr, tau_[1])
    Ba = Bc.num[0][0][1]
    showResult(Ap, Bp, Ac, Bc, Ba)
