"""

Example of Inverted pendulum control from section 4.1 of CDM Book [1]

[1] Shunji Manabe, Young Chol Kim, Coefficient Diagram Method
for Control System Design, Springer, 2021

"""
import numpy as np
import control as ct
from cdmtb import cdia, g2c
import matplotlib.pyplot as plt

s = ct.tf('s')

M = 0.25
ls = 0.1
m = 0.048
g0 = 9.8

Is = m*ls**2/3

a = m*g0*ls/(m*ls**2+Is)
b = M*ls/(m*ls**2+Is)

Bp = b*s
Ap = s**2-a

gr = np.array([2.5, 4, 4])
taur = 1.5

# Bc = k2*s^2+k1*s
# Ac = s^2+l1*s+l0
# Ba = m0

# controller parameters
nc = 2  # Ac = l0 (=1)
mc = 1  # Bc = k1s+k0

# controller gain calculation
P, Ac, Bc = g2c(Ap, Bp*s, nc, mc, gr, taur, nd=2)
Bc = Bc*s
# k1, k0 = Bc.num[0][0]
Ba = Ac.num[0][0][-1]

# plot CDM
opt_p = [Bc*Bp, Ac*Ap]
leg_opt_p = ['$B_cB_p$', '$A_cA_p$']
cdia(P, opt_p, leg_opt_p)

# plot closed-loop step response: vr=1, phi0=0
sys_cl = ct.append(Ba*Bp/P, Ba*Ap/P)
t, y = ct.step_response(sys_cl, T=np.linspace(0, 8, num=100))
plt.plot(t, y[0, 0, :], 'r-', label='$\\phi$')
plt.plot(t, y[1, 1, :], 'b--', label='$v$')
plt.legend()
plt.ylabel('Response')
plt.xlabel('time')
plt.show()

# plot closed-loop step response: vr=0, phi0=0.25
sys_cl = ct.append(Ac*s**2/P*0.25, -Bc*s**2/P*0.25)
t, y = ct.step_response(sys_cl, T=np.linspace(0, 3, num=100))
plt.plot(t, y[0, 0, :], 'r-', label='$\\phi$')
plt.plot(t, y[1, 1, :], 'b--', label='$v$')
plt.legend()
plt.ylabel('Response')
plt.xlabel('time')
plt.show()
