import numpy as np
from scipy.linalg import block_diag
from collections import defaultdict

class PsdHermitian:

    # Dunders
    def __init__(self, variables_dictionary, domain_chart, name : str, nblock : int = 1, blockstruct=None):
        self._name = name
        self._original_chart = domain_chart
        self._matrix, self._variables = PsdHermitian.clean_up_complexes(variables_dictionary, domain_chart)
        self._size = self._matrix[-1].shape[0]
        self._nblock = nblock
        self._blockstruct = [self._size] if blockstruct is None else blockstruct
        self._mdim = len(self._variables)-1 # Number of variables is the len(variable) minus the "constant" "variable"




    def __str__(self):
        return str(self._value)+"="+self._name

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        if not isinstance(other, PsdHermitian):
            return NotImplemented
        elif self._size != other._size:
            raise ValueError("Addition only supported for equal size matrices")
        else:
            out = defaultdict(lambda: 0)

            for d in (other._matrix, self._matrix):
                for k, v in d.items():
                    out[k] += v

            return PsdHermitian(dict(out),{key:False for key in out.keys()},name=self._name+"+"+other._name,nblock=self._nblock,blockstruct=self._blockstruct)

    def __sub__(self, other):
        if not isinstance(other, PsdHermitian):
            return NotImplemented
        elif self._size!=other._size:
            raise ValueError("Subtraction only supported for equal size matrices")
        else:
            out = defaultdict(lambda: 0)

            for d in (other._matrix, self._matrix):
                for k, v in d.items():
                    out[k] -= v

            return PsdHermitian(dict(out),{key:False for key in out.keys()},name=self._name+"-"+other._name,nblock=self._nblock,blockstruct=self._blockstruct)

    def __neg__(self):
        return PsdHermitian({key:-value for key,value in self._matrix},{key:False for key in self._variables},name= "-"+self._name)

    def __mul__(self, other):
        if not np.isscalar(other):
            return NotImplemented
        else:
            return PsdHermitian({key: other*value for key, value in self._matrix.items()},
                                {key: False for key in self._variables}, name=f"{other}*" + self._name)

    def __rmul__(self, other):
        return self.__mul__(other)

    # Properties
    @property
    def imag(self):
        return {key:value.imaginary for key,value in self._matrix.items()}
    @property
    def real(self):
        return {key:value.real for key,value in self._matrix.items()}
    @property
    def name(self):
        return self._name
    @property
    def shape(self):
        return self._size,self._size
    @property
    def size(self):
        return self._size
    @property
    def variables(self):
        return self._variables
    @property
    def matrix(self):
        return self.matrix
    @property
    def psd_form(self):
        if self._nblock == 1:
            return {
                key: np.block([
                    [mat.real, mat.imag],
                    [-mat.imag, mat.real]
                ])
                for key, mat in self._matrix.items()
            }
        else:
            return NotImplemented
    @property
    def blockstruct(self):
        return self._blockstruct
    @property
    def nblock(self):
        return self._nblock

    # Statics
    @staticmethod
    def real_variables(var_dic,dom_chart):
        out_dict = defaultdict(lambda:np.zeros(next(iter(var_dic)).shape),{key:np.array(value) for key,value in var_dic.items() if not dom_chart[key]})
        variable_names = []

        for key,value in set(var_dic.items())-set("constant"):

            i = value.imaginary
            r = value.real

            if dom_chart[key]:

                out_dict[key+"_real"] = np.array(r+1j*i,dtype=complex)
                out_dict[key+"_imaginary"] = np.array(-i+1j*r,dtype=complex)

        return out_dict, list(out_dict.keys())


    # Methods
    def rename(self,new_name):
        self._name = new_name

    def direct_sum(self,other):
        if not isinstance(other,PsdHermitian):
            return NotImplemented
        else:
            out = defaultdict(lambda:0)
            for d in (self._matrix,other._matrix):
                for key in d.keys():
                    out[key]=block_diag(self._matrix[key],other._matrix[key])


            return PsdHermitian(final_dictionary, dom_chart, name=self._name + "(+)" + other._name,nblock=self.nblock+other.nblock,blockstruct=self.blockstruct+other.blockstruct)


def symmetric_basis(n:int):
    """
    Generate a list of n x n matrices that form a basis for symmetric matrices.
    """
    basis = []

    # Diagonal elements
    for i in range(n):
        mat = np.zeros((n,n))
        mat[i,i] = 1
        basis.append(mat)

    # Off-diagonal elements
    for i in range(n):
        for j in range(i+1, n):
            mat = np.zeros((n,n))
            mat[i,j] = 1
            mat[j,i] = 1
            basis.append(mat)

    return basis

def antisymmetric_basis(n:int):
    """
    Generate a list of n x n matrices that form a basis for antisymmetric matrices.
    """
    basis = []

    # Diagonal elements
    for i in range(n):
        mat = np.zeros((n,n))
        mat[i,i] = 0
        basis.append(mat)

    # Off-diagonal elements
    for i in range(n):
        for j in range(i+1, n):
            mat = np.zeros((n,n))
            mat[i,j] = 1
            mat[j,i] = -1
            basis.append(mat)

    return basis

def generate_hermitian(dimension: int,name: str):
    """
    Generates an array containing PsdHermitian objects that correspond to a hermitian matrix of size dimension with each entry a separate variable
    :param dimension: size of the matrix
    :param name: the name of the hermitian matrix (string)
    :return: Array of PsdHermitian objects
    """
    sym_basis = symmetric_basis(dimension)
    antisym_basis = antisymmetric_basis(dimension)
    return [PsdHermitian(a,name+f"_{i}") for  i,a in enumerate(zip(sym_basis, antisym_basis))]
