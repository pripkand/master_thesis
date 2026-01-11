import cvxpy as cp
import numpy as np



# === Variables exactly as listed ===
X    = cp.Variable()                 # real
X2   = cp.Variable()                 # real
P2   = cp.Variable()                 # real
XP2  = cp.Variable(complex=True)     # complex
P3   = cp.Variable()                 # real
X4   = cp.Variable()                 # real
X2P2 = cp.Variable()                 # real
P4   = cp.Variable()                 # real
XP3  = cp.Variable(complex=True)     # complex

# === Matrix written EXACTLY as in the screenshot (no conj added) ===
hardMPime = cp.bmat([
    [1,       X,        0,       X2,        P2,         -1/2,            1/2],
    [X,       X2,       1/2,     0,         0,          -1j*X,           0],
    [0,       1/2,      P2,     -2j*X,      P3,         XP2,             XP2],
    [X2,      0,       2j*X,     X4,        X2P2,       1j*X2/2,         3j*X2/2],
    [P2,      0,        P3,      X2P2,      P4,        -3j*P2 + XP3,    -2j*P2 + XP3],
    [-1/2,   1j*X,      XP2,   -1j*X2/2,    XP3,        1 + X2P2,        1/2 + X2P2],
    [1/2,     0,        XP2,   -3j*X2/2,   -1j*P2 + XP3, 1/2 + X2P2,     1 + X2P2]
])

# === Optimization problem ===
objective = cp.Minimize(X4 + P2)
constraints = [hardMPime >> 0]          # PSD constraint

#prob = cp.Problem(objective, constraints)
#prob.solve()

print("Status:", prob.status)
print("Optimal value:", prob.value)
print("Optimal variables:")
print("X =", X.value)
print("X2 =", X2.value)
print("P2 =", P2.value)
print("XP2 =", XP2.value)
print("P3 =", P3.value)
print("X4 =", X4.value)
print("X2P2 =", X2P2.value)
print("P4 =", P4.value)
print("XP3 =", XP3.value)
