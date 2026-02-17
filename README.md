# master_thesis
The code for the implementation of my master's thesis


# Project Structure

## Zero Temperature

### `zero-temperature_harmonic_oscillator.nb`
Mathematica implementation of the harmonic oscillator bootstrap at T=0.  
Constructs and simplifies the Hankel moment matrix, solves the SDP, and exports results to JSON.

### `plotting_harmonic_oscillator.ipynb`
Reads exported JSON data and generates plots.

### `zero_temperature_anharmonic_oscillator.py`
Python implementation of the T=0 anharmonic oscillator bootstrap using `sympy`, `cvxpy`, and SDPA.  
Solves the SDP and exports results.

### `plotting_anharmonic_oscillator.ipynb`
Loads output data and produces plots.

---

## Thermal Bootstrap

### `matrix_maker.nb`
Mathematica code generating operator bases, Schwinger–Dyson relations, moment matrices, and Gauss–Radau quadrature data.  
Exports all matrices and parameters to JSON for the thermal SDP.

### `thermal_bootstrap_header_file.py`
Core Python module that:
- Converts exported matrices into CVXPY expressions  
- Constructs and solves the thermal SDP  
- Performs temperature scans  
- Exports results to HDF5  

Includes numerical diagonalization for anharmonic thermal energy comparison.

### `thermal_harmonic_bootstrap.py`
Runs the thermal SDP for the harmonic oscillator (upper and lower bounds).

### `thermal_anharmonic_bootstrap.py`
Runs the thermal SDP for the anharmonic oscillator (upper and lower bounds).

### `plotting_harmonic.py`
Plots harmonic thermal bootstrap results from HDF5 output.

### `plotting_anharmonic.py`
Plots anharmonic thermal bootstrap results and compares with numerical thermal energy.
