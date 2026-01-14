from typing import MutableMapping

import numpy as np
from scipy.linalg import block_diag
from collections import defaultdict

class PsdHermitian(MutableMapping):

    # Dunders
    def __init__(self, variables_dictionary, domain_chart, name : str, nblock : int = 1, blockstruct=None):
        self._name = name
        self._original_chart = domain_chart
        self._matrix, self._variables = PsdHermitian.real_variables(variables_dictionary, domain_chart)
        self._size = next(iter(self._matrix.values())).shape[0]
        self._nblock = nblock
        self._blockstruct = [self._size] if blockstruct is None else blockstruct
        self._mdim = len(self._variables)-1 # Number of variables is the len(variable) minus the "constant" "variable"




    def __str__(self):
        return f"Hermitian Matrix {self._name} with dimension: {self._size} blocks {self._nblock} block structure {self._blockstruct} and number of variables {self._mdim}"

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

            return PsdHermitian(dict(out),{key:False for key in out.keys()},name=self._name+" + "+other._name,nblock=self._nblock,blockstruct=self._blockstruct)

    def __sub__(self, other):
        if not isinstance(other, PsdHermitian):
            return NotImplemented
        elif self._size!=other._size:
            raise ValueError("Subtraction only supported for equal size matrices")
        else:
            out = defaultdict(lambda: 0)

            for k, v in self._matrix.items():
                out[k] += v
            for k, v in other._matrix.items():
                out[k] -= v

            return PsdHermitian(dict(out),{key:False for key in out.keys()},name=self._name+" - "+other._name,nblock=self._nblock,blockstruct=self._blockstruct)

    def __neg__(self):
        return PsdHermitian({key:-value for key,value in self._matrix},{key:False for key in self._variables},name= "-"+self._name,nblock= self._nblock,blockstruct= self._blockstruct)

    def __mul__(self, other):
        if not np.isscalar(other):
            return NotImplemented
        else:
            return PsdHermitian({key: other*value for key, value in self._matrix.items()},
                                {key: False for key in self._variables}, name=f"{other}*" + self._name,nblock= self._nblock, blockstruct= self._blockstruct)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __getitem__(self, item):
        if isinstance(item,str):
            return self._matrix[item]
        elif isinstance(item,int):
            return self._matrix[self._variables[item]]
        else:
            return NotImplemented

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("Key must be a string")

        self._matrix[key] = value
        if key not in self._variables:
            self._variables.append(key)
            self._mdim += 1

    def __delitem__(self, key):
        del self._matrix[key]
        self._variables.remove(key)

    def __len__(self):
        return len(self._matrix)

    def __iter__(self):
        return iter(self._matrix)

    def __contains__(self, key):
        return key in self._matrix.keys()

    # Properties
    @property
    def imag(self):
        return {key:value.imag for key,value in self._matrix.items()}
    @property
    def real(self):
        return {key:value.real for key,value in self._matrix.items()}
    @property
    def name(self):
        return self._name
    @property
    def size(self):
        return next(iter(self._matrix.values())).shape[0]
    @property
    def shape(self):
        mat = next(iter(self._matrix.values()))
        return tuple(list(mat.shape)[1:])
    @property
    def variables(self):
        return self._variables
    @property
    def matrix(self):
        return self._matrix
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
        out_dict = defaultdict(lambda:np.zeros(next(iter(var_dic.values())).shape),{key:np.array(value) for key,value in var_dic.items() if not dom_chart[key]})

        for key, value in var_dic.items():
            if dom_chart[key]:
                i = value.imag
                r = value.real
                out_dict[key + "_real"] = r + 1j * i
                out_dict[key + "_imaginary"] = -i + 1j * r

        return out_dict, list(out_dict.keys())


    # Methods
    def keys(self):
        return self._matrix.keys()

    def values(self):
        return self._matrix.values()

    def items(self):
        return self._matrix.items()

    def rename(self,new_name):
        self._name = new_name

    def direct_sum(self, other):
        if not isinstance(other, PsdHermitian):
            return NotImplemented

        out = {}
        all_keys = set(self._matrix) | set(other._matrix)

        a0 = next(iter(self._matrix.values()))
        b0 = next(iter(other._matrix.values()))

        for key in all_keys:
            a = self._matrix.get(key, np.zeros_like(a0))
            b = other._matrix.get(key, np.zeros_like(b0))
            out[key] = block_diag(a, b)

        return PsdHermitian(
            out,
            {key: False for key in out},
            name=self._name + " (+) " + other._name,
            nblock=self.nblock + other.nblock,
            blockstruct=self.blockstruct + other.blockstruct
        )
    def psd(self,key):
        return self.psd_form[key]

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
