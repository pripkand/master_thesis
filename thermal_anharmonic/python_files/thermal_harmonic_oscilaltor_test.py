import cvxpy as cp
import numpy as np
import h5py
import json

from cvxpy import length

from header_file import build_matrix

with open("//home/pripoll/Documents/Uni_Classes/Masters_thesis/anharmonic_thermal/thermal_anharmonic/wolfram_output/harmonic_output_for_l=4_m=3_k=3.json","r") as f:
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

z_matrices=np.array([cp.Variable((n,n),name="Z_"+str(i),hermitian=True) for i in range(k+1)])
t_matrices=np.array([cp.Variable((n,n),name="T_"+str(i+1),hermitian=True) for i in range(m)])



# Sets up the constraints that impose the KMS condition
z_psd=[ cp.bmat([[z_matrices[i],z_matrices[i+1]],[z_matrices[i+1],A]])>>0 if i!=0 else cp.bmat([[B,z_matrices[1]],[z_matrices[1],A]])>>0 for i in range(k)]
t_psd=[ cp.bmat([[z_matrices[-1]-A-t_matrices[i],-np.sqrt(quadrature[i][0])*t_matrices[i]],[-np.sqrt(quadrature[i][0])*t_matrices[i],A-np.sqrt(quadrature[i][0])*t_matrices[i]]])>>0 for i in range(m)]


with h5py.File("python_outputs/harmonic_n="+str(n)+"_k="+str(k),'a') as f:
    beta_range = np.arange(10,60,1)
    data_sets = ["energy","status","variables"]
    e_values = []
    variable_values = []
    problem_status = []

    for data_set in data_sets:
        if data_set in f:
            del f[data_set]

    for beta in beta_range:
        objective = cp.Minimize(2*variables["P2"])  # Due to the Schwinger-Dyson Equations, X2=P2

        t_eq = [sum([quadrature[i][1] * t_matrices[i] for i in range(m)]) == -2 ** (-k) * beta * C]
        constraints = [M >> 0] + z_psd + t_psd + t_eq

        problem = cp.Problem(objective, constraints)
        problem.solve()

        problem_status.append(problem.status)
        e_values.append(problem.value)

        variables_dictionary = {
        "z_values" : [z_matrices[i].value for i in range(len(z_matrices))],
        "t_values" : [t_matrices[i].value for i in range(len(t_matrices))],
        "variables_values": [variable.value for variable in variables]}

        variable_values.append(variables_dictionary)

    f.create_dataset("energy", data=e_values)
    f.create_dataset("status", data=problem_status)
    f.create_dataset("variables", data=variable_values)


