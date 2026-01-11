import numpy as np
import cvxpy as cp

#P2=cp.Variable(name="P2")
P2=cp.Parameter(value=0.5,complex=False)
X4,P3,P4,X3=cp.Variable(name="X4"),cp.Variable(name="P3"),cp.Variable(name="P4"),cp.Variable(name="X3")
XP2,X2P,X2P2,X3P,XP3=cp.Variable(name="XP2",complex=True),cp.Variable(name="X2P",complex=True),cp.Variable(name="X2P2",complex=True),cp.Variable(name="X3P",complex=True),cp.Variable(name="XP3",complex=True)

M = cp.bmat([
    [1,        0,        0,        P2,        1j/2,      -1j/2,      P2],
    [0,        P2,       1j/2,      X3,        X2P,       X2P,        XP2],
    [0,       -1j/2,     P2,        X2P,       XP2,       XP2,        P3],
    [P2,       X3,       X2P,       X4,        X3P,      -1j*P2 + X3P, X2P2],
    [-1j/2,    X2P,      XP2,      -3j*P2 + X3P, 1 + X2P2, 0.5 + X2P2, -1j*P2 + XP3],
    [1j/2,     X2P,      XP2,      -2j*P2 + X3P, 0.5 + X2P2, 1 + X2P2,  XP3],
    [P2,       XP2,      P3,        X2P2,     -2j*P2 + XP3, -3j*P2 + XP3, P4]
])
z_matrices = np.array([cp.Variable((1, 1), name="Z_" + str(i), hermitian=True) for i in range(3 + 1)])
t_matrices = np.array([cp.Variable((1, 1), name="T_" + str(i + 1), hermitian=True) for i in range(3)])

beta=cp.Parameter(nonneg=True)

constant=cp.Constant(np.array([[1]]))
# {t_j,w_j}
# {{0, 1/9}, {0.1550510257, 0.512486}, {0.6449489743, 0.376403}}
constraints=[M>>0,
             cp.bmat([[constant,z_matrices[1]],[z_matrices[1],constant]])>>0, #
             cp.bmat([[z_matrices[1],z_matrices[2]],[z_matrices[2],constant]])>>0, #
             cp.bmat([[z_matrices[2],z_matrices[3]],[z_matrices[3],constant]])>>0,#
            cp.bmat([[z_matrices[-1]-constant-t_matrices[0],cp.Constant(np.array([[0]]))],[cp.Constant(np.array([[0]])),constant]])>>0,
        cp.bmat([[z_matrices[-1]-constant-t_matrices[1],-np.sqrt(0.1550510257)*t_matrices[1]],[-np.sqrt(0.1550510257)*t_matrices[1],constant-0.1550510257*t_matrices[1]]])>>0,
            cp.bmat([[z_matrices[-1] - constant - t_matrices[2], -np.sqrt(0.6449489743) * t_matrices[2]],
                      [-np.sqrt(0.6449489743) * t_matrices[2], constant - 0.6449489743* t_matrices[2]]]) >> 0,
            1/9*t_matrices[0]+0.512486*t_matrices[1]+0.376403*t_matrices[2]==0
             ]

objective=cp.Minimize(2)
problem=cp.Problem(objective, constraints)
for i in np.arange(0.4,1, 0.1):
    P2.value=i
    problem.solve(verbose=True,solver=cp.SCS,warm_start=True)
    print(2*P2.value)
    print(problem.status)
