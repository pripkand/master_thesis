import cvxpy as cp
import numpy as np
import json
import h5py
import time


def build_matrix(coefficients,variables):
    """
    Reconstructs a matrix from it's coefficient matrices.
    variables.
    :param coefficients: Takes a dictionary that is formated as: {"constant":[Real Part,Imaginary Part],variable_name:[R,I]...}
    :param variables: Takes a dictionary that is formated as {variable_name:variable,...}
    :return: A matrix that is a function of cvxpy variables
    """
    matrix_coefficients = {key: np.array(value[0]) + 1j * np.array(value[1]) for key, value in coefficients.items()}
    return  matrix_coefficients["constant"] + cp.sum(
        [matrix_coefficients[key] * variables[key] for key in matrix_coefficients.keys() & variables.keys()])

def run_sdp(temp_range:np.ndarray,input_file:str,output_folder:str)->None:
    """
    Runs the thermal and saves it to a file
    :param temp_range: The range over which the SDP will run.
    :param input_file: The file to fetch data from.
    :param output_folder: The folder to output the h5py file. The output file has the convention <system>_L=<L>_n=<n>_k=<k>
    :return: None
    """
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
    # print("C",C,"B",B,"A",A)
    # Create the inverse temperature parameter
    beta = cp.Parameter(nonneg=True)

    # Define the Z and T matrices as variables where needed.
    z_matrices = np.array([cp.Variable((n, n), name="Z_" + str(i), hermitian=True) for i in range(k + 1)])
    t_matrices = np.array([cp.Variable((n, n), name="T_" + str(i + 1), hermitian=True) for i in range(m)])

    constraints = [M >> 0, z_matrices[0] == B]

    # Append Z block constraints
    for i in range(k):
        expr = cp.bmat([
            [z_matrices[i], z_matrices[i + 1]],
            [z_matrices[i + 1], A]
        ])
        constraints.append(expr >> 0)

    # Append T block constraints
    for j in range(m):
        t_j = quadrature[j][0]
        expr = cp.bmat([
            [z_matrices[-1] - A - t_matrices[j], -cp.Constant(np.sqrt(t_j)) * t_matrices[j]],
            [-cp.Constant(np.sqrt(t_j)) * t_matrices[j], A - cp.Constant(t_j) * t_matrices[j]]
        ])
        constraints.append(expr >> 0)

    weights = [cp.Constant(quadrature[i][1]) for i in range(m)]
    t_eq = sum(weights[j] * t_matrices[j] for j in range(m)) + cp.Constant(2 ** (-k)) * beta * C

    constraints.append(t_eq == 0)
    objective = 2*variables["P2"] if system.strip().lower()=="harmonic" else 3/2*variables["P2"]
    objective_func = cp.Minimize(objective)
    problem = cp.Problem(objective_func, constraints)
    
    status = []
    energy = []
    for t in temp_range:
        beta.value = 1 / t

        status.append(problem.status)
        energy.append(problem.value)

    e_values = []
    variable_values = []
    problem_status = []
    for i,t in enumerate(temp_range):
            # Progress Print
            if i%5==0:
                start_time=time.time()
                print(f"beta={1/t} L={L} m={m} k={k} n={n}")

            beta.value = 1/t
            problem.solve(
                warm_start=True, verbose=True, solver=cp.SCS,eps=1e-5,
                eps_abs=1e-6,
                eps_rel=1e-6,
                acceleration_lookback=10,
                normalize=True,
                scale=1.0
            )# Solve using SCS with some parameters adjusted. Look at https://www.cvxgrp.org/scs/api/settings.html#settings for particulars

            # Progress Print
            if i%5==0:
                end_time=time.time()
                elapsed=end_time-start_time
                print(f"Status:{problem.status} Energy:{problem.value} Elapsed Time:{elapsed:.3f} seconds")

            # Save probelm status, energy values and variable values
            problem_status.append(problem.status)
            e_values.append(problem.value)

            variables_dictionary = {
                "z_values": [z_matrices[i].value for i in range(len(z_matrices))],
                "t_values": [t_matrices[i].value for i in range(len(t_matrices))],
                "variables_values": [var.value for key, var in variables.items()]}
            variable_values.append(variables_dictionary)

    # Export File
    with h5py.File(output_folder+"/"+system+"_L=" + str(L) + "_m=" + str(m) + "_k=" + str(k), 'a') as f:
        data_sets = ["energy", "status", "variables","temperatures"]

        for data_set in data_sets:
            if data_set in f:
                del f[data_set]
        f.create_dataset("energy", data=e_values)
        f.create_dataset("status", data=problem_status)
        f.create_dataset("temperatures",data=temp_range)
        # f.create_dataset("variables", data=variable_values)

    # Print out export file location and name
    print("File Exported as: ",output_folder+"/"+system+"_L=" + str(L) + "_m=" + str(m) + "_k=" + str(k))

    return None