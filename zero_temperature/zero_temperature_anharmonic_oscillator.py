# ================================
# IMPORTS
# ================================
import sympy as sp
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import json


# ================================
# Recursive relation
# ================================
def req_rel(t, EE, gg, x):
    return (
        EE*(t-3)*x.get(t-4, 0)
        + sp.Rational(1,4)*(t-4)*(t-5)*(t-3)*x.get(t-6, 0)
        - (t-1)*x.get(t, 0)
        - (t-2)*gg*x.get(t-2, 0)
    )


# ================================
# Build symbolic Hankel matrix
# ================================
def build_matrix(k):
    e, g, x2 = sp.symbols('e g x2')

    max_index = 2*k + 2
    x = {}

    # Create moment symbols x0, x2, ..., x_{2k+2}
    for i in range(0, max_index+1, 2):
        x[i] = sp.symbols(f'x{i}')

    # Generate recursion relations
    rels = []
    for j in range(2, k+2):
        rels.append(req_rel(2*j, e, g, x))

    # Solve recursively
    rules = {x[0]: 1}

    for i in range(1, k+1):
        xi = x[2*i+2]
        eq = rels[i-1].subs(rules)
        sol = sp.solve(eq, xi)[0]
        rules[xi] = sp.simplify(sol)

    # Build Hankel matrix
    M = sp.zeros(k+1)

    for i in range(k+1):
        for j in range(k+1):
            if (i + j) % 2 == 1:
                M[i, j] = 0
            elif i == 0 and j == 0:
                M[i, j] = 1
            else:
                M[i, j] = x[i+j]

    # Substitute recursion and x2 parameter
    M = M.subs(rules)
    M = M.subs({x[2]: x2})

    return sp.simplify(M)


# ================================
# Decompose M = M0 + x2 M1
# ================================
def matrix_decomposition(M):
    x2 = sp.symbols('x2')

    M0 = M.subs(x2, 0)
    M1 = sp.zeros(*M.shape)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            M1[i, j] = sp.diff(M[i, j], x2)

    return sp.simplify(M0), sp.simplify(M1)


# ================================
# Solve SDP at fixed energy
# ================================
def solve_sdp_at_e(M0_sym, M1_sym, e_val, g_val, k, tolerance=1e-8):

    test = M0_sym.subs({'e': e_val, 'g': g_val})
    print(test.free_symbols)
    # Substitute parameters numerically
    M0_num = np.array(
        sp.N(M0_sym.subs({'e': e_val, 'g': g_val}), 20)
    ).astype(np.float64)

    M1_num = np.array(
        sp.N(M1_sym.subs({'e': e_val, 'g': g_val}), 20)
    ).astype(np.float64)

    # CVXPY variables
    t = cp.Variable()
    x2 = cp.Variable()

    I = np.eye(k+1)

    constraint = M0_num + x2*M1_num - t*I

    problem = cp.Problem(
        cp.Maximize(t),
        [constraint >> 0]
    )

    problem.solve(solver=cp.SDPA, abstol=tolerance)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        return None

    return t.value, x2.value

def solve_reverse_sdp_at_e(M0_sym, M1_sym, e_val, g_val, k, x2_low, tolerance=1e-8):

    test = M0_sym.subs({'e': e_val, 'g': g_val})
    print(test.free_symbols)
    # Substitute parameters numerically
    M0_num = np.array(
        sp.N(M0_sym.subs({'e': e_val, 'g': g_val}), 20)
    ).astype(np.float64)

    M1_num = np.array(
        sp.N(M1_sym.subs({'e': e_val, 'g': g_val}), 20)
    ).astype(np.float64)

    # CVXPY variables
    t = cp.Variable()
    x2 = cp.Variable()

    I = np.eye(k+1)

    constraint = -M0_num + x2*M1_num + t*I

    problem = cp.Problem(
        cp.Maximize(t),
        [constraint >> 0, x2 >= x2_low]
    )

    problem.solve(solver=cp.SDPA, abstol=tolerance)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        return None

    return t.value, x2.value


# ================================
# Allowed energies filter
# ================================
def allowed_energies(data):
    accepted = []
    for (e, tstar, eigenvalue, x2star) in data:
        flag = 1 if tstar >= 0 else 0
        accepted.append([float(e), float(tstar), float(eigenvalue), float(x2star), int(flag)])
    return accepted


# ================================
# Main driver
# ================================
def do_sdp_at_k(searchspace, step, g, k, tolerance=1e-8):

    # Build symbolic matrix once
    M = build_matrix(k)
    M0_sym, M1_sym = matrix_decomposition(M)

    full_data = []

    e_values = np.arange(searchspace[0], searchspace[1] + step, step)

    for e_val in e_values:

        result = solve_sdp_at_e(M0_sym, M1_sym, e_val, g, k, tolerance)

        if result is None:
            print(f"Solver failed at e = {e_val}")
            continue

        tstar, x2star = result

        # Compute smallest eigenvalue numerically
        M_numeric = np.array(
            sp.N(M.subs({'e': e_val, 'g': g, 'x2': x2star}), 20)
        ).astype(np.float64)

        eigenvalue = np.min(np.linalg.eigvalsh(M_numeric))

        full_data.append([float(e_val), float(tstar), float(eigenvalue), float(x2star)])

    return allowed_energies(full_data)

def get_x2_low(data,e):
    for item in data:
        if item[0]==e:
            return item[-2]
    return None

def do_reverse_sdp_at_k(searchspace, step, g, k, results, tolerance=1e-8):

    # Build symbolic matrix once
    M = build_matrix(k)
    M0_sym, M1_sym = matrix_decomposition(M)

    full_data = []

    e_values = np.arange(searchspace[0], searchspace[1] + step, step)

    for e_val in e_values:
        x2_low = get_x2_low(results,e_val)
        result = solve_reverse_sdp_at_e(M0_sym, M1_sym, e_val, g, k, x2_low, tolerance)

        if result is None:
            print(f"Solver failed at e = {e_val}")
            continue

        tstar, x2star = result

        # Compute smallest eigenvalue numerically
        M_numeric = np.array(
            sp.N(M.subs({'e': e_val, 'g': g, 'x2': x2star}), 20)
        ).astype(np.float64)

        eigenvalue = np.min(np.linalg.eigvalsh(M_numeric))

        full_data.append([float(e_val), float(tstar), float(eigenvalue), float(x2star)])

    return full_data

# ================================
# Scan allowed x2 interval per energy
# ================================
def scan_energy_x2_intervals(e_range, e_step, x2_range, x2_step, g, k, tolerance=1e-8):

    # Build symbolic matrix once
    M_sym = build_matrix(k)

    results = []

    e_values = np.arange(e_range[0], e_range[1] + e_step, e_step)
    x2_values = np.arange(x2_range[0], x2_range[1] + x2_step, x2_step)

    for e_val in e_values:

        allowed_x2 = []

        for x2_val in x2_values:

            M_eval = M_sym.subs({
                'e': e_val,
                'g': g,
                'x2': x2_val
            })

            # Skip if something remains symbolic
            if M_eval.free_symbols:
                continue

            M_num = np.array(
                sp.matrix2numpy(sp.N(M_eval, 20), dtype=float)
            )

            min_eig = np.min(np.linalg.eigvalsh(M_num))

            if min_eig >= -tolerance:
                allowed_x2.append(x2_val)

        if allowed_x2:
            x2_min = min(allowed_x2)
            x2_max = max(allowed_x2)
            results.append((float(e_val), float(x2_min), float(x2_max)))

    return results

# ================================
# Example usage
# ================================
if __name__ == "__main__":

    searchspace = [-5, 7]
    step = 0.1
    g = -5
    krange = [7,10]
    kstep = 1
    x2range = [0.7,2.5]
    x2step = 0.05
    
    res_dict={}
    eigenvalue_check={}
    upper_res={}
    for k in range(krange[0],krange[1]+1,kstep):
        res_dict[str(k)] = list(do_sdp_at_k(searchspace, step, g, k))
        #eigenvalue_check[str(k)] = list(scan_energy_x2_intervals(searchspace,step,x2range,x2step,g,k))
        upper_res = list(do_reverse_sdp_at_k(searchspace,step,g,k,res_dict[str(k)]))        

    #with open(f'anharmonic_oscillator_no_temp_k_{krange[0]}_to_{krange[-1]}_{kstep}_step_g_minus_5.json', 'w') as outfile:
        #json.dump(res_dict, outfile) 
    
    #with open(f'eigen_value_check_anharmonic_oscillator_no_temp_k_{krange[0]}_to_{krange[-1]}_{kstep}_step_g_minus_5.json', 'w') as outfile:
        #json.dump(eigenvalue_check, outfile)
    
    with open(f'upper_x2_bound_nharmonic_k_{krange[0]}_to_{krange[-1]}_{kstep}_step_g_minus_5.json','w') as outfile:
        json.dump(upper_res,outfile)
    
        
    
    