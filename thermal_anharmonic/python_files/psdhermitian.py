import operator
from functools import reduce

import numpy as np
import time
import subprocess as sp

from typing import MutableMapping
from scipy.linalg import block_diag
from collections import defaultdict

class PsdHermitian(MutableMapping):

    # Dunders
    def __init__(self, variables_dictionary, domain_chart, name : str, nblock : int = 1, blockstruct=None):
        self._name = name
        self._original_chart = domain_chart
        self._matrix, self._variables = PsdHermitian.real_variables(variables_dictionary, domain_chart)
        self._nblock = nblock
        self._blockstruct = [self.size] if blockstruct is None else blockstruct
        #self._mdim = len(self._variables)-1 # Number of variables is the len(variable) minus the "constant" "variable"




    def __str__(self):
        return f"Hermitian Matrix {self._name} with dimension: {self.size} blocks {self._nblock} block structure {self._blockstruct} and number of variables {self._mdim}"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        if not isinstance(other, PsdHermitian):
            return NotImplemented
        elif self.size != other.size:
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
        elif self.size!=other.size:
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
            return {key:block_diag([np.block([[block.real,block.imag],[-block.imag,block.real]]) for block in PsdHermitian.extract_diag_blocks(value,self._blockstruct)]) for key,value in self._matrix}
    @property
    def blockstruct(self):
        return self._blockstruct
    @property
    def nblock(self):
        return self._nblock
    @property
    def mdim(self):
        return len(self._matrix)-1

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
    @staticmethod
    def extract_diag_blocks(a, blockstruct):
        blocks = []
        start = 0
        for n in blockstruct:
            end = start + n
            blocks.append(a[start:end, start:end])
            start = end
        return blocks

    # Class methods
    @classmethod
    def from_basis(cls, dimension: int, name: str):
        indices = [(i,j) for i in range(dimension) for j in range(dimension) if i>=j]

        variables = {}
        chart = {}

        for index in indices:
            (i,j) = index
            if i==j:
                var = f"{name}_({i},{j})"
                s = np.zeros((dimension, dimension))
                s[(i,j)] = 1
                s[(j,i)] = 1
                variables[var] = s + 1j * np.zeros((dimension, dimension))
                chart[var]=False
            else:
                var_r = f"Re{name}_({i},{j})"
                var_i = f"Im{name}_{i},{j}"

                s = np.zeros((dimension, dimension))
                a = np.zeros((dimension, dimension))

                s[(i, j)] = 1
                s[(j, i)] = 1

                a[(i,j)] = 1
                a[(j,i)] = -1

                variables[var_r] = s + 1j * np.zeros((dimension, dimension))
                variables[var_i] = np.zeros((dimension, dimension)) + 1j * a

                chart[var_r] = False
                chart[var_i] = False

            variables["constant"] = np.zeros((dimension, dimension), dtype=complex)
            chart["constant"] = False

        return cls(variables, chart, name=name)

    @classmethod
    def ones(cls,size:int,spot:int):
        if spot == 1:
            return PsdHermitian({"constant":np.array([[1,0],[0,0]])},
                {"constant":False},
                name=""
            )
        elif spot == 2:
            return PsdHermitian({"constant":np.array([[0,1],[0,0]])},
                {"constant":False},
                name=""
            )
        elif spot == 3:
            return PsdHermitian({"constant":np.array([[0,0],[1,0]])},
                {"constant":False},
                name=""
            )
        elif spot == 4:
            return PsdHermitian({"constant":np.array([[0,0],[0,1]])},
                {"constant":False},
                name=""
            )
        else:
            return NotImplemented

    @classmethod
    def from_constant(cls, mat, name="C"):
        return cls(
            {"constant": np.asarray(mat, dtype=complex)},
            {"constant": False},
            name=name
        )

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

    def direct_product(self, other):
        if not isinstance(other, PsdHermitian):
            return NotImplemented

        out = {}

        for k1, a in self._matrix.items():
            for k2, b in other._matrix.items():
                if k1 == "constant" and k2 == "constant":
                    key = "constant"
                elif k1 == "constant":
                    key = k2
                elif k2 == "constant":
                    key = k1
                else:
                    key = f"{k1}*{k2}"

                kron = np.kron(a, b)

                if key in out:
                    out[key] += kron
                else:
                    out[key] = kron

        return PsdHermitian(
            out,
            {key: False for key in out},
            name=f"{self._name} ⊗ {other._name}",
            nblock=self._nblock * other._nblock,
            blockstruct=[
                a * b
                for a in self._blockstruct
                for b in other._blockstruct
            ]
        )

class PSDConstraint:

    # Dunders
    def __init__(self, matrix: PsdHermitian):
        self._matrix = matrix

    def __add__(self,other):
        return PSDConstraint(self.matrix.direct_sum(other.matrix))

    def __str__(self):
        return self._matrix.name+" >= 0 "

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return iter(self._matrix)

    def __contains__(self, key):
        return key in self._matrix.keys()

    def __getitem__(self, item):
        if isinstance(item,str):
            return self._matrix[item]
        elif isinstance(item,int):
            return self._matrix[self._variables[item]]
        else:
            return NotImplemented

    # Class methods
    @classmethod
    def from_sides(cls, side_a: PsdHermitian, side_b: PsdHermitian = None):
        if side_b is None:
            return cls(side_a)
        return cls(side_a - side_b)

    # Properties
    @property
    def matrix(self):
        return self._matrix
    @property
    def psd_form(self):
        return self._matrix.psd_form
    @property
    def mdim(self):
        return self._matrix.mdim
    @property
    def nblock(self):
        return self._matrix.nblock
    @property
    def blockstruct(self):
        return self._matrix.blockstruct
    def keys(self):
        return self._matrix.keys()

class EqualityConstraint(PSDConstraint):

    def __init__(self,side_a:PsdHermitian,side_b:PsdHermitian=None):
        if side_b is not None:
            diff= side_a - side_b
        else:
            diff= side_a

        psd_matrix = diff.direct_sum(-diff)
        super().__init__(psd_matrix)

class Problem:

    gmp_library_path = str()
    out_file_path = str()
    in_file_path = str()

    def __init__(self, target: dict, constraints: list | np.ndarray,
                 in_file_name: str = None, out_file_name: str = None):

        if not isinstance(target, dict):
            raise ProblemSetupError("target must be a dictionary")

        if not constraints:
            raise ProblemSetupError("constraints list cannot be empty")

        try:
            self._constraints = reduce(operator.add, constraints)
        except Exception as e:
            raise ProblemSetupError("Failed to reduce constraints") from e

        if not hasattr(self._constraints, "keys"):
            raise ProblemSetupError("Reduced constraints must behave like a mapping")

        self._target = Problem.extend_target(target, self._constraints)

        if not isinstance(self._target, dict):
            raise ProblemSetupError("extend_target must return a dictionary")

        self._solved = False
        self._solved_variables = []
        self._problem_status = ""

        self._in_file_name = (
            in_file_name if in_file_name is not None
            else f"sdpa_infile_{time.time()}.dat"
        )
        self._out_file_name = (
            out_file_name if out_file_name is not None
            else f"sdpa_out_file_{time.time()}.dat"
        )


    def __str__(self):
        return "target function: "+Problem.target_to_string(self._target)+" with constraints: "+Problem.constraints_to_string(self._constraints)

    def solve(self,force_resolve:bool=False):
        if not Problem.gmp_library_path:
            raise ProblemSetupError("GMP library path not set")

        if not Problem.in_file_path:
            raise ProblemSetupError("Input file path not set")

        if not Problem.out_file_path:
            raise ProblemSetupError("Output file path not set")

        # Runs the sdp with the gmp libray
        if self._solved and force_resolve:
            print("problem already solved. Use force_solve=True to solve again")
            return 0
        else:
            try:
                Problem.write_file(
                    Problem.in_file_path + self._in_file_name,
                    self._target,
                    self._constraints
                )
            except Exception as e:
                raise ProblemIOError("Failed writing SDPA input file") from e

            try:
                sp.run(
                    ["./sdpa_gmp",
                     Problem.in_file_path + self._in_file_name,
                     Problem.out_file_path + self._out_file_name],
                    cwd=Problem.gmp_library_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
            except sp.CalledProcessError as e:
                raise ProblemSolveError(
                    f"SDPA failed:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
                ) from e
            return 0
    
    @staticmethod
    def extend_target(target,full_constraints):
        out = target.copy()
        for key in full_constraints.keys():
            if key not in target and key!="constant":
                out[key] = 0
        
        return out
    @staticmethod
    def target_to_string(d):
        parts = []
        for k, v in d.items():
            if v == 0:
                continue
            sign = "-" if v < 0 else "+"
            coeff = abs(v)

            if coeff == 1:
                term = k
            else:
                term = f"{coeff}*{k}"

            parts.append((sign, term))

        if not parts:
            return "0"

        first_sign, first_term = parts[0]
        out = (first_term if first_sign == "+" else "-" + first_term)

        for sign, term in parts[1:]:
            out += f" {sign} {term}"
        return out

    @staticmethod
    def constraints_to_string(constraints):
        out=""
        for item in constraints:
            out += str(item)+", "
        return out

    @staticmethod
    def write_file(out_file,target,constraints):
        if not hasattr(constraints, "mdim"):
            raise ProblemSetupError("constraints must have mdim attribute")

        if len(target) != constraints.mdim:
            print(target)
            raise ProblemSetupError(
                f"Target size ({len(target)}) "
                f"!= constraint mdim ({constraints.mdim})"
            )

        if not hasattr(constraints, "blockstruct"):
            raise ProblemSetupError("constraints missing blockstruct")

        try:
            with open(out_file, "w") as f:
                f.write(f"{len(target)} = mDIM\n")
                f.write(f"{constraints.nblock} = nBLOCK\n")
                f.write(
                    f"{[2*stru for stru in constraints.blockstruct]}"
                    .replace("[", "{").replace("]", "}")
                    + " = BLOCKSTRUCT\n"
                )

                f.write("{")
                f.write(", ".join(str(target[k]) for k in target))
                f.write("}"+f" = {[key for key in target.keys()]}\n")
                Problem.write_matrix_dense(f,constraints["constant"],constraints.blockstruct)
                f.write("=F_constant\n")
                for k in target:
                    if k not in constraints:
                        raise ProblemSetupError(f"Constraint missing key {k}")

                    Problem.write_matrix_dense(
                        f, constraints[k], constraints.blockstruct
                    )
                    f.write(f"=F_{k}\n")
        except OSError as e:
            raise ProblemIOError("Failed writing SDPA file") from e

    @staticmethod
    def write_matrix_dense(f, m, blockstruct):
        print("Started Printing")
        start = 0
        psd_m = np.block([[m.real,m.imag],[-m.imag,m.real]]).real
        for bs in blockstruct:
            block = psd_m[start:start + 2*bs, start:start + 2*bs] # The factor of 2 on bs is because the PSD form of a nxn hermitian matrix is 2nx2n
            start += 2*bs

            for i,row in enumerate(block):
                if i==0:
                    f.write("{ ")
                f.write("{ ")
                f.write(" ".join(f"{x}, " for x in row).replace("[","{").replace("]","}") + " ")
                f.write(" }, ")
                if i==len(block)-1:
                    f.write("}")
        print("Ended Printing")

    @classmethod
    def set_gmp_path(cls,path:str):
        cls.gmp_library_path = path

    @classmethod
    def set_out_file(cls,path:str):
        cls.out_file_path = path

    @classmethod
    def set_in_file(cls, path: str):
        cls.in_file_path = path

## Problem Handling Classes
class ProblemError(Exception):
    pass

class ProblemIOError(ProblemError):
    pass

class ProblemSetupError(ProblemError):
    pass

class ProblemSolveError(ProblemError):
    pass