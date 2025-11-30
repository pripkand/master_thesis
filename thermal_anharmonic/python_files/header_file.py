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
    return  matrix_coefficients["constant"] + sum(np.array(
        [matrix_coefficients[key] * variables[key] for key in matrix_coefficients.keys() & variables.keys()]))

def run_sdp(beta_range:np.ndarray,input_file:str,output_folder:str)->None:
    """
    Runs the thermal and saves it to a file
    :param beta_range: The range over which the SDP will run.
    :param input_file: The file to fetch data from.
    :param output_folder: The folder to output the h5py file. The output file has the convention <system>_L=<L>_n=<n>_k=<k>
    :return: None
    """
    with open(input_file,"r") as f:
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

    # Create the inverse temperature parameter
    beta = cp.Parameter(nonneg=True)

    # Define the Z and T matrices as variables where needed.
    z_matrices = np.array([cp.Variable((n, n), name="Z_" + str(i), hermitian=True) for i in range(k + 1)])
    t_matrices = np.array([cp.Variable((n, n), name="T_" + str(i + 1), hermitian=True) for i in range(m)])

    # Sets up the constraints that impose the KMS condition
    z_psd = [cp.bmat([[z_matrices[i], z_matrices[i + 1]], [z_matrices[i + 1], A]]) >> 0 if i != 0 else cp.bmat(
        [[B, z_matrices[1]], [z_matrices[1], A]]) >> 0 for i in range(k)]
    t_psd = [cp.bmat([[z_matrices[-1] - A - t_matrices[i], -np.sqrt(quadrature[i][0]) * t_matrices[i]],
                      [-np.sqrt(quadrature[i][0]) * t_matrices[i], A - np.sqrt(quadrature[i][0]) * t_matrices[i]]]) >> 0
             for i in range(m)]

    t_eq = [sum([quadrature[i][1] * t_matrices[i] for i in range(m)]) == -2 ** (-k) * beta * C]
    constraints = [M >> 0] + z_psd + t_psd + t_eq
    objective = cp.Minimize(2 * variables["P2"]) if system == "Harmonic" else cp.Minimize(
        3 / 2 * variables["P2"])  # Due to the Schwinger-Dyson Equations, X2=P2 or X4=P2/2
    problem = cp.Problem(objective, constraints)

    e_values = []
    variable_values = []
    problem_status = []
    for i,inv_temp in enumerate(beta_range):
            # Progress Print
            if i%5==0:
                start_time=time.time()
                print(f"beta={inv_temp} L={L} m={m} k={k} n={n}")

            beta.value = inv_temp
            problem.solve(solver='SDPA',verbose=True,warm_start=True)

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
        data_sets = ["energy", "status", "variables"]

        for data_set in data_sets:
            if data_set in f:
                del f[data_set]
        f.create_dataset("energy", data=e_values)
        f.create_dataset("status", data=problem_status)
        # f.create_dataset("variables", data=variable_values)

    # Print out export file location and name
    print("File Exported as: ",output_folder+"/"+system+"_L=" + str(L) + "_m=" + str(m) + "_k=" + str(k))