import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import h5py
import json

from header_file import build_matrix,run_sdp
print(cp.installed_solvers())
input_file = "/home/pripoll/Documents/Uni_Classes/Masters_thesis/anharmonic_thermal/thermal_anharmonic/wolfram_output/harmonic_output_for_l=6_m=3_k=3.json"
out_folder = "python_outputs"

with open(input_file, "r") as f:
    data = json.load(f)

    parameters = data["parameters"]
    variable_domains = data["domains"]

# Build Variables with names and domains defined by the Json file data
variables = {key: cp.Variable(name=key, complex=value) for key, value in variable_domains.items()}

# Take size parameters from data
system = parameters["type"]
L = parameters['L']
n = parameters["n"]
quadrature = parameters["quadrature"]
m = len(quadrature)
k = parameters["k"]

# Build Matrices from the Json file data
M = build_matrix(data["M"], variables)
A = build_matrix(data["A"], variables)
B = build_matrix(data["B"], variables)
C = build_matrix(data["C"], variables)
#print("C",C,"B",B,"A",A)
# Create the inverse temperature parameter
beta = cp.Parameter(nonneg=True)

# Define the Z and T matrices as variables where needed.
z_matrices = np.array([cp.Variable((n, n), name="Z_" + str(i), hermitian=True) for i in range(k + 1)])
t_matrices = np.array([cp.Variable((n, n), name="T_" + str(i + 1), hermitian=True) for i in range(m)])

constraints = [M >> 0, z_matrices[0] == B]

# Append Z block constraints
for i in range(k):
    expr = cp.bmat([
        [z_matrices[i],z_matrices[i+1]],
        [z_matrices[i+1],A]
    ])
    constraints.append(expr >> 0)

# Append T block constraints
for j in range(m):
    t_j = quadrature[j][0]
    expr = cp.bmat([
        [z_matrices[-1]-A-t_matrices[j],-cp.Constant(np.sqrt(t_j))*t_matrices[j]],
        [-cp.Constant(np.sqrt(t_j))*t_matrices[j],A-cp.Constant(t_j)*t_matrices[j]]
    ])
    constraints.append(expr>> 0)

weights = [cp.Constant(quadrature[i][1]) for i in range(m)]
t_eq = sum(weights[j]*t_matrices[j] for j in range(m))+cp.Constant(2**(-k))*beta*C

constraints.append(t_eq==0)

objective = cp.Minimize(2 *variables["P2"])
problem = cp.Problem(objective, constraints)

temp_range=np.arange(0.1,0.8,0.1)
status=[]
energy=[]
for t in temp_range:
    beta.value = 1/t
    problem.solve(warm_start=True,verbose=True,solver=cp.SCS)
    status.append(problem.status)
    energy.append(problem.value)
print(status)
plt.plot(temp_range,[en for en in energy])
plt.plot(temp_range,[1/2+np.exp(1/(t))/(np.exp(1/t)-1) for t in temp_range],color="red")

plt.show()

