Title: Electrical Resistivity Tomography Conditional Diffusion

Overview:
This repository provides tools to generate, process, and visualize Electrical Resistivity Tomography (ERT) data using conditional diffusion techniques and PFLOTRAN-based simulations. It includes:

PFLOTRAN input file generation from parameter sets.
Utilities to gather ERT data (.srv files), preprocess, and visualize.
Plotting utilities for ensemble statistics (mean, mode, quantiles) and difference maps.

Reproducible runs with fixed random seeds.
Plotting functions for ensemble analysis and quantiles.
Ready-to-extend hooks for training/inference if you add ML routines.

Requirements:
Python 3.10+ recommended
Optional: PFLOTRAN installed and accessible for running simulations


Quickstart:
Generate PFLOTRAN input files from parameters:
ertdiff simulate --template examples/template.in --params examples/params.csv --output outputs
Gather ERT data from .srv files:
ertdiff gather-data --dir ./outputs --prefix SIM
Plot ensemble and quantiles from NumPy arrays:
ertdiff plot --input path/to/results.npz --outfig figures/summary.png

Data:
The ERTDataHandler assumes .srv files contain data in column 5 starting after 259 header lines. Adjust in src/ert_diffusion/data.py if your files differ.
PFLOTRAN usage:

This package writes input files (does not run PFLOTRAN). You can run PFLOTRAN externally and then use gather-data to collect outputs.

Disclaimer

This material was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

PACIFIC NORTHWEST NATIONAL LABORATORY

operated by

BATTELLE

for the

UNITED STATES DEPARTMENT OF ENERGY

under Contract DE-AC05-76RL01830
