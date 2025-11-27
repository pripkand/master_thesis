import cvxpy as cp
import numpy as np
import json

from header_file import build_matrix

with open("/home/pripoll/Documents/Uni_Classes/Masters_thesis/thermal_anharmonic/wolfram_output/output_for_l=4_m=3_k=3.json","r") as f:
    data=json.load(f)

    parameters=data["parameters"]
    variable_domains=data["domains"]

# Build Variables with names and domains defined by the Json file data
variables={key:cp.Variable(name=key,complex=value) for key,value in variable_domains.items()}

# Take size parameters from data
n = parameters["n"]
quadrature = parameters["quadrature"]
m = len(quadrature)
k = parameters["k"]

# Build Matrices from the Json file data
M = build_matrix(data["M"],variables)
A = build_matrix(data["A"],variables)
B = build_matrix(data["B"],variables)
C = build_matrix(data["C"],variables)

z_matrices=np.array([cp.Variable((n,n),name="Z_"+str(i),hermitian=True) for i in range(k)])
t_matrices=np.array([cp.Variable((n,n),name="T_"+str(i+1),hermitian=True) for i in range(m)])

objective=cp.Minimize(3/2*variables["P2"]) # Due to the Schwinger-Dyson Equations, X4=P2/2
constraints=[M >> 0]
z_psd=[ cp.bmat([[z_matrices[i],z_matrices[i+1]],[z_matrices[i+1],A]])>>0 for i in range(k-1)]
t_psd=[ cp.bmat() for i in range(m)]
#comment
problem=cp.Problem(objective, constraints)
problem.solve()

print("Status:", problem.status)
print("Optimal value:", problem.value)
print("Optimal variables:")
for key, variable in variables.items():
    print(key + " =", variable.value)