import json
import cvxpy as cp
import numpy as np

with open("/home/pripoll/Desktop/test.json","r") as f:
    data=json.load(f)

    psd_vars={}
    psd_coef_mats={}
    for key,value in data.items():
        psd_coef_mats[key]=np.array(value[0])+1j*np.array(value[1])
        if key!="constant":
            psd_vars[key]=cp.Variable(name=key)

    print(psd_coef_mats["constant"])
    for key in psd_vars:
        print(psd_vars[key]*psd_coef_mats[key])