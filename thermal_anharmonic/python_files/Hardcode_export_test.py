import numpy as np
import cvxpy as cp


P2,X4,P3,P4,X3=cp.Variable(name="P2"),cp.Variable(name="X4"),cp.Variable(name="P3"),cp.Variable(name="P4"),cp.Variable(name="X3")
XP2,X2P,X2P2,X3P,XP3=cp.Variable(name="XP2",complex=True),cp.Variable(name="X2P",complex=True),cp.Variable(name="X2P2",complex=True),cp.Variable(name="X3P",complex=True),cp.Variable(name="XP3",complex=True)

M=cp.bmat([
    [1,0,0,P2,P2,1j/2,-1j/2],
    [0,P2,1j/2,X3,XP2,X2P,X2P],
    [0,-1j/2,P2,X2P,P3,XP2,XP2],
    [P2,X3,X2P,X4,X2P2,X3P,-1J*P2+X3P],
    [P2,XP2,P3,X2P2,P4,-1J*2*P2+XP3,-1J*3*P2+XP3],
    [-1J/2,X2P,XP2,-1j*3*P2+X3P,-1J*P2+XP3,1+X2P2,1/2+X2P2],
    [1J/2,X2P,XP2,-1j*2*P2+X3P,XP3,1/2+X2P2,1*X2P2]
])
z_matrices = np.array([cp.Variable((1, 1), name="Z_" + str(i), hermitian=True) for i in range(3 + 1)])
t_matrices = np.array([cp.Variable((1, 1), name="T_" + str(i + 1), hermitian=True) for i in range(3)])

beta=cp.Parameter(nonneg=True)

constant=cp.Constant(np.array([[1]]))

constraints=[M>>0,cp.bmat([[constant,z_matrices[1]],[z_matrices[1],constant]])>>0,cp.bmat([[z_matrices[1],z_matrices[2]],[z_matrices[2],constant]])>>0,cp.bmat([[z_matrices[2],z_matrices[3]],[z_matrices[3],constant]])>>0,
            cp.bmat([[z_matrices[-1]-constant-t_matrices[0],-np.sqrt(0)*t_matrices[0]],[cp.Constant(np.array([[0]])),constant]])>>0,cp.bmat([[z_matrices[-1]-constant-t_matrices[1],-np.sqrt(0.355051)*t_matrices[1]],[-np.sqrt(0.355051)*t_matrices[1],constant-0.355051*t_matrices[1]]])>>0,
            cp.bmat([[z_matrices[-1] - constant - t_matrices[2], -np.sqrt(0.844949) * t_matrices[2]],
                      [-np.sqrt(0.844949) * t_matrices[2], constant - 0.844949* t_matrices[2]]]) >> 0,
            1/9*t_matrices[0]+0.512486*t_matrices[1]+0.376403*t_matrices[2]==0
             ]

objective=cp.Minimize(2*P2)
problem=cp.Problem(objective, constraints)

problem.solve(verbose=True)

print(problem.status)
print(problem.value)