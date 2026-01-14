import numpy as np
from scipy.linalg import block_diag

x = np.array([[1,2],[3,4]])
y = block_diag(x,x)
print(x.shape)