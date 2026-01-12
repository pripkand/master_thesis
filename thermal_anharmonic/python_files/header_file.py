import cvxpy as cp
import numpy as np
import json
import h5py
import time

from fontTools.ufoLib.utils import deprecated


## These functions are meant to be used with CVXPY
def build_matrix(coefficients,variables):
	"""
	Reconstructs a matrix from its coefficient matrices.
	variables.
	:param coefficients: Takes a dictionary that is formated as: {"constant":[Real Part,Imaginary Part],variable_name:[R,I]...}
	:param variables: Takes a dictionary that is formated as {variable_name:variable,...}
	:return: A matrix that is a function of cvxpy variables
	"""
	matrix_coefficients = {key: np.array(value[0]) + 1j * np.array(value[1]) for key, value in coefficients.items()}
	return  matrix_coefficients["constant"] + cp.sum(
		[matrix_coefficients[key] * variables[key] for key in matrix_coefficients.keys() & variables.keys()])

def run_sdp(temp_range:np.ndarray,input_file:str,output_folder:str,verbose:bool=False,maximize:bool=False,solver_parameters:dict = None)->None:
	"""
	Runs the thermal and saves it to a file
	:param temp_range: The range over which the SDP will run.
	:param input_file: The file to fetch data from.
	:param output_folder: The folder to output the HDF5 file. The output file has the convention <system>_L=<L>_n=<n>_k=<k>_<upper/lower>
	:param verbose: Whether to set the solver as verbose.
	:param maximize: Whether to maximize the objective or to minimize. Default is False
	:param solver_parameters: The solver parameters to be included. If none are supplied, default values are used.
	:return: None
	"""
	start_time = time.time()
	with open(input_file, "r") as f:
		data = json.load(f)

		parameters = data["parameters"]
		variable_domains = data["domains"]

	# Build Variables with names and domains defined by the JSON file data
	variables = {key: cp.Variable(name=key, complex=value) for key, value in variable_domains.items()}

	# Take size parameters from data
	system = parameters["type"]
	L = parameters['L']
	n = parameters["n"]
	quadrature = parameters["quadrature"]
	m = len(quadrature)
	k = parameters["k"]

	# Build Matrices from the JSON file data
	M = build_matrix(data["M"], variables)
	A = build_matrix(data["A"], variables)
	B = build_matrix(data["B"], variables)
	C = build_matrix(data["C"], variables)
	# print("C",C,"B",B,"A",A)
	# Create the inverse temperature parameter
	beta = cp.Parameter(nonneg=True)

	# Define the Z and T matrices as variables where needed.
	z_matrices = np.array([cp.Variable((n, n), name="Z_" + str(i), hermitian=True) for i in range(k + 1)])
	t_matrices = np.array([cp.Variable((n, n), name="T_" + str(i + 1), hermitian=True) for i in range(m)])

	constraints = [M >> 0, z_matrices[0] == B]

	# Append Z block constraints
	for i in range(k):
		expr = cp.bmat([
			[z_matrices[i], z_matrices[i + 1]],
			[z_matrices[i + 1], A]
		])
		constraints.append(expr >> 0)

	# Append T block constraints
	for j in range(m):
		t_j = quadrature[j][0]
		expr = cp.bmat([
			[z_matrices[-1] - A - t_matrices[j], -cp.Constant(np.sqrt(t_j)) * t_matrices[j]],
			[-cp.Constant(np.sqrt(t_j)) * t_matrices[j], A - cp.Constant(t_j) * t_matrices[j]]
		])
		constraints.append(expr >> 0)

	# Append Sum T constraints
	weights = [cp.Constant(quadrature[i][1]) for i in range(m)]
	t_eq = sum(weights[j] * t_matrices[j] for j in range(m)) + cp.Constant(2 ** (-k)) * beta * C
	constraints.append(t_eq == 0)

	# For the Harmonic case H=p^2/2+x^2/2 so for <x^2>=<p^2> => <H>=<p^2>
	objective = variables["P2"] if system.strip().lower()=="harmonic" else 3/2*variables["P2"]

	if verbose: print(objective)

	objective_func = cp.Maximize(objective) if  maximize else cp.Minimize(objective)
	problem = cp.Problem(objective_func, constraints)

	# Set Default Parameters
	if solver_parameters is None :
		solver_parameters = {"max_iters":20000,"eps":1e-6,"eps_abs":1e-7,"eps_rel":1e-7,"acceleration_lookback":10,"normalize":True,"scale":1.0}

	# Prime Storers
	e_values = []
	variable_values = {}
	problem_status = []
	problem_stats_solve_time = {}
	problem_stats_iters = {}
	violation_flags = []
	# Solve for all Inverse Temperatures
	for i,t in enumerate(temp_range):

			if i%5==0:
				print(f'#| L={L} and (m,k)=({m},{k}) | System Type: {system}')
				print(f"##| Progress:{i/len(temp_range)*100}%")

			beta.value = 1/t
			problem.solve(
				verbose=verbose,
				warm_start=True,  solver=cp.SDPA,# **solver_parameters
			)# Solve using SCS with some parameters adjusted. Look at https://www.cvxgrp.org/scs/api/settings.html#settings for particulars

			# Grab Problem Statistics
			problem_stats = problem.solver_stats

			# Progress Print
			if i%5==0:
				print(f"##| Status:{problem.status} | Energy:{problem.value} ",
					  #f"| Elapsed Time:{problem_stats.solve_time:.3f} seconds"
				)

			# Check for Violations
			for con_i,con in enumerate(problem.constraints):
				res=con.violation()
				if res.any()>1e-7:
					if con_i==0:
						violation_flags.append(f"hankel_violation_L={L}_m={m}_k={k}_at_t={t}_residual={res}")
					if con_i == 1:
						violation_flags.append(f"z_block_violation_L={L}_m={m}_k={k}_at_t={t}_residual={res}")
					if con_i == 2:
						violation_flags.append(f"t_block_violation_L={L}_m={m}_k={k}_at_t={t}_residual={res}")
					if con_i == 3:
						violation_flags.append(f"eq_violation_L={L}_m={m}_k={k}_at_t={t}_residual={res}")

			# Save problem status, energy values and variable values
			problem_status.append(problem.status)
			e_values.append(problem.value)

			variable_values["z_values_" + str(t)] = np.asarray([z_matrices[i].value for i in range(len(z_matrices))],)
			variable_values["t_values_" + str(t)] = [t_matrices[i].value for i in range(len(t_matrices))]
			for key,var in variables.items():
				if var.value is not None:
					variable_values["variables_values_"+"key_" + str(t)] = var.value
				else:
					continue



			# Save solve time and max iters
			problem_stats_solve_time["solve_time_"+str(t)]=problem_stats.solve_time
			problem_stats_iters["iters_"+str(t)]=problem_stats.num_iters

	# Export File
	bound = "_upper" if maximize else "_lower"
	out_file_name = output_folder+"/"+system+"_L=" + str(L) + "_m=" + str(m) + "_k=" + str(k) + bound + ".hdf5"
	with h5py.File(out_file_name, 'a') as f:

		f.attrs["System"] = system
		f.attrs["Time Elapsed"] = start_time-time.time()
		f.attrs["Description"] = f"This file contains the bootstrap the {system} oscillator with non-zero temperature with size {L} and relaxation values m={m} and k={k}"
		f.attrs["Size"] = L
		f.attrs["Quadrature Points"] = m
		f.attrs["Logarithmic Quenching"] = k
		f.attrs["Bound"] = bound

		data_sets = ["energy", "status", "temperatures","flags"] + [ name for name in variable_values.keys()]+[name for name in problem_stats_iters.keys()]+[name for name in problem_stats_solve_time.keys()]

		# Delete exiting data sets that will be rewritten
		for data_set in data_sets:
			if data_set in f:
				del f[data_set]

		# Write new datasets
		f.create_dataset("energy", data=e_values)
		f.create_dataset("status", data=problem_status)
		f.create_dataset("temperatures",data=temp_range)
		f.create_dataset(
			"flags",
			data=violation_flags,
			dtype=h5py.string_dtype()
		)
		for key,value in variable_values.items():
			f.create_dataset(key, data=value)
		#for key in problem_stats_iters.keys():
			#f.create_dataset(key, data=problem_stats_iters[key])
		#for key in problem_stats_solve_time.keys():
			#f.create_dataset(key, data=problem_stats_solve_time[key])


	# Print out export file location and name
	print("File Exported as: ",out_file_name)

	return None

## These functions are to make the problem into SDPA format and then solve.
def make_hermitian_psd(coefficients) -> np.ndarray:
	"""
	Takes the real and imaginary parts of a hermitian matrix and returns a matrix of the form [[Real,-Imaginary],[Imaginary,Real]] which is positive semidefinite.
	:param coefficients: Takes a 2 item list that is of the form [Real,Imaginary]
	:return: A matrix of the form [[[Real,0],[0,Real]],[0,-Imaginary],[Imaginary,0]]
	"""
	real, imaginary = coefficients
	return np.array([[real,-imaginary],[imaginary,real]])

class PsdHermitian:

	# Dunders
	def __init__(self, variables_dictionary, domain_chart, name : str):
		self._name = name
		self._reals,self._imaginaries,self._variables = PsdHermitian.clean_up_complexes(variables_dictionary,domain_chart)
		self._size = self._reals.shape[0]


	def __str__(self):
		return str(self._value)+"="+self._name

	def __repr__(self):
		return self.__str__()

	def __add__(self, other):
		if not isinstance(other, PsdHermitian):
			return NotImplemented
		else:
			final_dictionary = {}

			other_keys = set(other._value)
			self_keys = set(self._value)

			dom_chart = {key: False for key in other_keys | self_keys}
			for key in self_keys & other_keys:
				final_dictionary[key] = [self._value[key][0]+other._value[key][0], self._value[key][1]+other._value[key][1]]

			other_diff = other_keys - self_keys
			self_diff = self_keys -other_keys

			for key in other_diff:
				final_dictionary[key] = other._value[key]
			for key in self_diff:
				final_dictionary[key] = self._value[key]

			return PsdHermitian(final_dictionary,dom_chart,name = self._name+"+"+other._name)

	def __sub__(self, other):
		if not isinstance(other, PsdHermitian):
			return NotImplemented
		else:
			final_dictionary = {}

			other_keys = set(other._value)
			self_keys = set(self._value)

			dom_chart = {key: False for key in other_keys | self_keys}
			for key in self_keys & other_keys:
				final_dictionary[key] = [self._value[key][0] - other._value[key][0],
											self._value[key][1] - other._value[key][1]]

			other_diff = other_keys - self_keys
			self_diff = self_keys - other_keys

			for key in other_diff:
				final_dictionary[key] = -1*other._value[key]
			for key in self_diff:
				final_dictionary[key] = self._value[key]

			return PsdHermitian(final_dictionary, dom_chart, name=self._name + "-" + other._name)

	def __neg__(self):
		return PsdHermitian({key:np.array([-value[0],-value[1]]) for key,value in self._value.items()},{key:False for key in self._variables},name= "-"+self._name)

	def __mul__(self, other):
		if not np.isscalar(other):
			return NotImplemented
		else:
			return PsdHermitian({key: np.array([other*value[0], other*value[1]]) for key, value in self._value.items()},
								{key: False for key in self._variables}, name="-" + self._name)

	def __rmul__(self, other):
		return self.__mul__(other)

	# Properties
	@property
	def imag(self):
		return {key:value[1] for key,value in self._value.items()}
	@property
	def real(self):
		return {key:value[0] for key,value in self._value.items()}
	@property
	def name(self):
		return self._name
	@property
	def value(self):
		return self._value

	# Statics
	@staticmethod
	def clean_up_complexes(var_dic,dom_chart):
		reals = []
		imaginaries = []
		variable_names = []

		for key,value in var_dic.items():

			i = np.array(value[1])
			r = np.array(value[0])

			if dom_chart[key]:
				reals.append(r)
				imaginaries.append(i)
				variable_names.append(key+"_real")

				reals.append(-i)
				imaginaries.append(r)
				variable_names.append(key+"_imaginary")
			else:
				reals.append(r)
				imaginaries.append(i)
				variable_names.append(key)
		return np.array(reals),np.array(imaginaries),np.array(variable_names)


	# Methods
	def rename(self,new_name):
		self._name = new_name

	def direct_sum(self,other):
		if not isinstance(other,PsdHermitian):
			return NotImplemented
		else:
			final_dictionary = {}

			self_shape =

			other_keys = set(other.value)
			self_keys = set(self._value)

			dom_chart = {key: False for key in other_keys | self_keys}
			for key in self_keys & other_keys:

				self_r,self_i = self._value[key]
				other_r,other_i = other.value[key]
				zero_self = np.zeros(self_r.shape)
				zero_other = np.zeros(other_r.shape)

				final_dictionary[key] = np.array([
					np.block([[self_r,zero_self],[zero_other,other_r]]),
					np.block([[self_i,zero_self],[zero_other,other_i]])
				])

			other_diff = other_keys - self_keys
			self_diff = self_keys - other_keys

			for key in other_diff:

				other_r, other_i = other.value[key]
				zero = np.zeros(self_r.shape)


				final_dictionary[key] = np.array([
					np.block([[zero,zero],[zero,other_r]]),
					np.block([[zero,zero],[zero,other_i]])
				])
			for key in self_diff:

				self_r,self_i = self._value[key]
				zero = np.zeros(self_r.shape)

				final_dictionary[key] = np.array([
					np.block([[self_r,zero],[zero,zero]]),
					np.block([[self_i,zero],[zero,zero]])
				])

			return PsdHermitian(final_dictionary, dom_chart, name=self._name + "(+)" + other._name)


	@deprecated("This is probably useless")
	def print_hermitian(self):
		return print(self._real + 1j * self._imaginary)

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

def translate_variables(dictionary_of_variables,complexity_chart)-> np.ndarray:
	final_list = []
	for key,value in dictionary_of_variables.items():
		if not complexity_chart[key]:
			final_list.append(PsdHermitian(value, name = key))
		else:
			addition = np.array(value[0])+np.array(value[1])
			subtraction = np.array(value[0])-np.array(value[1])
			zeros = np.zeros(addition.shape)
			final_list.append(PsdHermitian([addition,zeros],name = key+"_real"))
			final_list.append(PsdHermitian([zeros,subtraction],name = key+"_imaginary"))

	return np.array(final_list)

if __name__ == "__main__":
	con =[
			[
				[
					1,
					0,
					0,
					0,
					0,
					0,
					0
				],
				[
					0,
					0,
					0,
					0,
					0,
					0,
					0
				],
				[
					0,
					0,
					0,
					0,
					0,
					0,
					0
				],
				[
					0,
					0,
					0,
					0,
					0,
					0,
					0
				],
				[
					0,
					0,
					0,
					0,
					1,
					0.5,
					0
				],
				[
					0,
					0,
					0,
					0,
					0.5,
					1,
					0
				],
				[
					0,
					0,
					0,
					0,
					0,
					0,
					0
				]
			],
			[
				[
					0,
					0,
					0,
					0,
					0.5,
					-0.5,
					0
				],
				[
					0,
					0,
					0.5,
					0,
					0,
					0,
					0
				],
				[
					0,
					-0.5,
					0,
					0,
					0,
					0,
					0
				],
				[
					0,
					0,
					0,
					0,
					0,
					0,
					0
				],
				[
					-0.5,
					0,
					0,
					0,
					0,
					0,
					0
				],
				[
					0.5,
					0,
					0,
					0,
					0,
					0,
					0
				],
				[
					0,
					0,
					0,
					0,
					0,
					0,
					0
				]
			]
		]
	p2 = [
			[
				[
					0,
					0,
					0,
					1,
					0,
					0,
					1
				],
				[
					0,
					1,
					0,
					0,
					0,
					0,
					0
				],
				[
					0,
					0,
					1,
					0,
					0,
					0,
					0
				],
				[
					1,
					0,
					0,
					0,
					0,
					0,
					0
				],
				[
					0,
					0,
					0,
					0,
					0,
					0,
					0
				],
				[
					0,
					0,
					0,
					0,
					0,
					0,
					0
				],
				[
					1,
					0,
					0,
					0,
					0,
					0,
					0
				]
			],
			[
				[
					0,
					0,
					0,
					0,
					0,
					0,
					0
				],
				[
					0,
					0,
					0,
					0,
					0,
					0,
					0
				],
				[
					0,
					0,
					0,
					0,
					0,
					0,
					0
				],
				[
					0,
					0,
					0,
					0,
					0,
					-1,
					0
				],
				[
					0,
					0,
					0,
					-3,
					0,
					0,
					-1
				],
				[
					0,
					0,
					0,
					-2,
					0,
					0,
					0
				],
				[
					0,
					0,
					0,
					0,
					-2,
					-3,
					0
				]
			]
		]
	x2p = [
			[
				[
					0,
					0,
					0,
					0,
					0,
					0,
					0
				],
				[
					0,
					0,
					0,
					0,
					1,
					1,
					0
				],
				[
					0,
					0,
					0,
					1,
					0,
					0,
					0
				],
				[
					0,
					0,
					1,
					0,
					0,
					0,
					0
				],
				[
					0,
					1,
					0,
					0,
					0,
					0,
					0
				],
				[
					0,
					1,
					0,
					0,
					0,
					0,
					0
				],
				[
					0,
					0,
					0,
					0,
					0,
					0,
					0
				]
			],
			[
				[
					0,
					0,
					0,
					0,
					0,
					0,
					0
				],
				[
					0,
					0,
					0,
					0,
					0,
					0,
					0
				],
				[
					0,
					0,
					0,
					0,
					0,
					0,
					0
				],
				[
					0,
					0,
					0,
					0,
					0,
					0,
					0
				],
				[
					0,
					0,
					0,
					0,
					0,
					0,
					0
				],
				[
					0,
					0,
					0,
					0,
					0,
					0,
					0
				],
				[
					0,
					0,
					0,
					0,
					0,
					0,
					0
				]
			]
		]
	variables = {"constant":con,"P2":p2,"X2P":x2p}
	chart = {
		"constant":False,
		"P2":False,
		"P3":False,
		"P4":False,
		"X2P":True,
		"X2P2":True,
		"X3":False,
		"X3P":True,
		"X4":False,
		"XP2":True,
		"XP3":True
	}

	tr = translate_variables(variables,chart)

	print(tr)



# Next:
# -Ditch the dictionary-> np.array structure in favor of lists.