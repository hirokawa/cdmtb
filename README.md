# cdmtb
Control System Design Toolbox in Python for Coefficient Diagram Method (CDM) 

Coefficient Diagram Method (CDM) is control system design method proposed by Retired Prof. Shunji Manabe. The basic tutorial is available from [Manabe's CDM Tutorial](http://www.cityfujisawa.ne.jp/~manabes/CDMRef2011-8-3/BriefTutorialCdm(AC028102)2002b.pdf).
CDM is an unique control system design method using an algegraic control design approach based on the coefficient of characteristic polynomials. The design tradeoff between stability and the response and the robustness analysis can be conducted in the coefficient diageam.

The detail of CDM is described in CDM Book[^1].

[^1]: Shunji Manabe, Young Chol Kim, Coefficient Diagram Method for Control System Design, Springer, 2021

[![PyPI - Version](https://img.shields.io/pypi/v/cdmtb.svg)](https://pypi.org/project/cdmtb)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cdmtb.svg)](https://pypi.org/project/cdmtb)

-----

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install cdmtb
```

## Simple example

A simple example for motor control described in section 3.5 in CDM Book is shown as follows:

The system diagram of motor controller:
<img src="docs/system_motor.png" alt="motor controller" title="motor controller">

The sample of Python code:
```python
import numpy as np
import control as ct
from cdmtb import cdia, g2c

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
```

The CDM is shown as follows:
<img src="docs/t1_cdm.png" alt="CDM plot" title="CDM plot">

It shows the coeffficient of characsteric polynomials, the stability index $\gamma$, the stability index limit $\gamma^*$. The contribution each feedback gain is also shown in the CDM.

The step response is shown as follows:
<img src="docs/t1_step.png" alt="step response plot" title="step response plot">

## License

`cdmtb` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.


