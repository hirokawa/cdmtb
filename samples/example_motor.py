"""

Example of motor control from section 3.5 of CDM Book [1]

[1] Shunji Manabe, Young Chol Kim, Coefficient Diagram Method
for Control System Design, Springer, 2021

"""
import numpy as np
import control as ct
from cdmtb.cdmtb import cdia, g2c

s = ct.tf('s')

# plant definition
tau_v, tau_m = 0.25, 1

Ap = (tau_v*s+1)*(tau_m*s+1)*s
Bp = 1+0*s

# controller parameters
nc = 0  # Ac = l0 (=1)
mc = 1  # Bc = k1s+k0

gr = np.array([2.5, 2])
taur = 1

# controller gain calculation
P, Ac, Bc = g2c(Ap, Bp, nc, mc, gr, taur)
k1, k0 = Bc.num[0][0]
Ba = k0

# plot CDM
opt_p = [k1*Bp*s, k0*Bp, Ap]
leg_opt_p = ['$k_1B_ps$', '$k_0B_p$', '$A_p$']
cdia(P, opt_p, leg_opt_p)

# plot closed-loop step response
sys_cl = Ba*Bp/P
ct.step_response(sys_cl).plot()
