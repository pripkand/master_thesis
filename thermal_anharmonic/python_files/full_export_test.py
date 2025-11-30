import cvxpy as cp
import numpy as np
import json

from header_file import build_matrix

with open("//home/pripoll/Documents/Uni_Classes/Masters_thesis/anharmonic_thermal/thermal_anharmonic/wolfram_output/output_for_l=4_m=3_k=3.json","r") as f:
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

# Define the Z and T matrices as variables where needed
"""
z_matrices=[]
for i in range(k+1):
    if i==0:
        z_matrices.append(B) # This has to be done this way because B will have a seemingly different shape than the other Z_i
        # It is also possible that this equality could be imposed as a constraint further down but I dont see why that would be better
    else:
        z_matrices.append(cp.Variable((n,n),name="Z_"+str(i),hermitian=True))"""
z_matrices=np.array([cp.Variable((n,n),name="Z_"+str(i),hermitian=True) for i in range(k+1)])
t_matrices=np.array([cp.Variable((n,n),name="T_"+str(i+1),hermitian=True) for i in range(m)])

objective=cp.Minimize(3/2*variables["P2"]) # Due to the Schwinger-Dyson Equations, X4=P2/2

# Sets up the constraints that impose the KMS condition
z_psd=[ cp.bmat([[z_matrices[i],z_matrices[i+1]],[z_matrices[i+1],A]])>>0 if i!=0 else cp.bmat([[B,z_matrices[1]],[z_matrices[1],A]])>>0 for i in range(k)]
t_psd=[ cp.bmat([[z_matrices[-1]-A-t_matrices[i],-np.sqrt(quadrature[i][0])*t_matrices[i]],[-np.sqrt(quadrature[i][0])*t_matrices[i],A-np.sqrt(quadrature[i][0])*t_matrices[i]]])>>0 for i in range(m)]
t_eq=[sum([quadrature[i][1]*t_matrices[i] for i in range(m)])==-2**(-k)*1/0.1*C]

# Joins all constraints together
constraints=[M >> 0]+z_psd+t_psd+t_eq

problem=cp.Problem(objective, constraints)
problem.solve()

print("Status:", problem.status)
print("Optimal value:", problem.value)
print("Optimal variables:")
for key, variable in variables.items():
    print(key + " = ", variable.value)
for matrix in z_matrices:
    print(matrix.name," = ",matrix.value)
for matrix in t_matrices:
    print(matrix.name, " = ",matrix.value)