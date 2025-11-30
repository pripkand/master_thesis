import cvxpy as cp
import numpy as np

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
