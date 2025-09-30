Title: Electrical Resistivity Tomography Conditional Diffusion

Overview:
This repository provides tools to generate, process, and visualize Electrical Resistivity Tomography (ERT) data using conditional diffusion techniques and PFLOTRAN-based simulations. It includes:

PFLOTRAN input file generation from parameter sets.
Utilities to gather ERT data (.srv files), preprocess, and visualize.
Plotting utilities for ensemble statistics (mean, mode, quantiles) and difference maps.
Packaging and CLI to make workflows reproducible and user-friendly.
Features:

Modular Python package with CLI.
Reproducible runs with fixed random seeds.
Plotting functions for ensemble analysis and quantiles.
Ready-to-extend hooks for training/inference if you add ML routines.
Requirements:

Python 3.10+ recommended
Optional: PFLOTRAN installed and accessible for running simulations
Dependencies listed in requirements.txt

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
