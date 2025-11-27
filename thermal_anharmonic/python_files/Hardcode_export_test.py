import numpy as np
import json
import cvxpy as cp


domains={"X" : False, "X2" : False, "P2" : False, "P3" : False, "X4" : False,
 "X2P2" : False, "P4" : False, "XP2" : True, "XP3" : True}

with open("wolfram_output/output_for_l=4_m=3.json","r") as f:
    data=json.load(f)

    psd_vars={}
    psd_coef_mats={}
    for key,value in data.items():
        psd_coef_mats[key]=np.array(value[0])+1j*np.array(value[1])
        if key!="constant":
            psd_vars[key]=cp.Variable(name=key,complex=domains[key])

M = psd_coef_mats["constant"] + sum(np.array([psd_vars[key] * psd_coef_mats[key] for key in psd_vars.keys()]))

objective = cp.Minimize(psd_vars["X4"] + psd_vars["P2"])
constraints = [M >> 0]
print("passed")
prob = cp.Problem(objective, constraints)
prob.solve()

print("Status:", prob.status)
print("Optimal value:", prob.value)
print("Optimal variables:")
for key, variable in psd_vars.items():
    print(key + " =", variable.value)
    """
    Status: optimal
Optimal value: 9.92730264065434
Optimal variables:
X = 1.3072214471009065e-05
X2 = 0.072274123977404
P2 = 6.26343237383771
XP2 = (0.005244868949792456+0j)
P3 = 4.9072670298670536e-06
X4 = 3.6638702668166285
X2P2 = 303.6296445113358
P4 = 24998.452761812303
XP3 = (-8.492732824381106e-07+0j)

Process finished with exit code 0
    """