#!/usr/bin/ipython3
import os
import numpy as np
import pandas as pd
import math
from itertools import islice
from ypstruct import structure
import sys
import matplotlib.pyplot as plt
from subprocess import call, TimeoutExpired
import subprocess
import Table
import GROMACS_FILES
import random
import multiprocessing as mp
import shutil
from scipy.optimize import curve_fit
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
DESTINATION_FOLDER0 = Path("/home/uni08/soleimani/RUNS/Thesis/Membrane_Project/PAPER/4000/")
TEMP1 = 200
TEMPN = 420
INTERVAL = 20

# Helper Functions
def create_directory(path):
    """Create a directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)

def move_files(source, destination):
    """Move files from source to destination."""
    try:
        shutil.move(source, destination)
        logging.info(f"Moved: {source} to {destination}")
    except Exception as e:
        logging.error(f"Failed to move {source}: {e}")

def copy_files(source, destination):
    """Copy files from source to destination."""
    try:
        shutil.copy(source, destination)
        logging.info(f"Copied: {source} to {destination}")
    except Exception as e:
        logging.error(f"Failed to copy {source}: {e}")

def run_command(command, cwd=None, timeout=None):
    """Run a shell command with optional timeout."""
    try:
        subprocess.call(command, shell=True, cwd=cwd, timeout=timeout)
    except TimeoutExpired:
        logging.warning(f"Command timed out: {' '.join(command)}")

# Main Optimization Function
def optimize(x, nfe, lst):
    """Optimize parameters using GROMACS simulations."""
    nfe += 1
    epsilon, rmin, r1, rc, bFC, aFC, Friction = x

    # Generate GROMACS files
    mdp_min = GROMACS_FILES.MINIMIZATION(epsilon, rmin, r1, rc, bFC, aFC, Friction, nfe)
    mdp_nvt = GROMACS_FILES.NVT(epsilon, rmin, r1, rc, bFC, aFC, Friction, nfe)
    mdp_npt = GROMACS_FILES.NPT(epsilon, rmin, r1, rc, bFC, aFC, Friction, nfe)
    ff = GROMACS_FILES.FORCEFIELD(epsilon, rmin, r1, rc, bFC, aFC, Friction, nfe)
    top = GROMACS_FILES.TOPOLOGY(epsilon, rmin, r1, rc, bFC, aFC, Friction, nfe)
    tabulated_files = Table.Tables(epsilon, rmin, rc + 1.2, r1, rc, bFC, aFC, Friction, nfe)

    # Create directories and move files
    destination_folder = DESTINATION_FOLDER0 / "RUNS0"
    create_directory(destination_folder)

    # Run simulations
    try:
        run_command(["bash", "Calculation002.sh", str(epsilon), str(rmin), str(r1), str(rc), str(bFC), str(aFC), str(Friction), str(nfe)], cwd=destination_folder, timeout=1800)
    except Exception as e:
        logging.error(f"Simulation failed: {e}")

    # Calculate cost
    cost = calculate_cost(destination_folder)
    return cost, nfe, lst

def calculate_cost(destination_folder):
    """Calculate the cost based on simulation results."""
    # Placeholder for cost calculation logic
    return 0.0

# Genetic Algorithm Functions
def single_point_crossover(p1, p2, cut):
    """Perform single-point crossover."""
    c1, c2 = p1.deepcopy(), p2.deepcopy()
    c1.position = np.append(p1.position[:cut], p2.position[cut:])
    c2.position = np.append(p2.position[:cut], p1.position[cut:])
    return c1, c2

def mutate(x, mu, sigma):
    """Mutate an individual."""
    y = x.deepcopy()
    flag = np.random.rand(*x.position.shape) <= mu
    ind = np.argwhere(flag)
    y.position[ind] += sigma[ind] * np.random.randn(*ind.shape)
    return y

# Main Script
if __name__ == "__main__":
    # Initialize population
    npop = 48
    pop = [structure() for _ in range(npop)]
    for i in range(npop):
        pop[i].position = np.random.uniform([1.0, 1.0, 1.5, 2.5, 100, 10.0, 2], [10.0, 1.5, 2.5, 3.0, 20000, 20000, 2])
        pop[i].position[2] = pop[i].position[1] + random.uniform(0.5, 1.5)
        pop[i].position[3] = pop[i].position[2] + random.uniform(0.0, 0.5)

    # Run optimization
    bestsol = structure()
    bestsol.cost = np.inf
    for it in range(1):  # Replace with maxit
        for i in range(npop):
            pop[i].cost, _, pop[i].Objectives = optimize(pop[i].position, 0, [0, 0, 0, 0, 0, 0, 0])
            if pop[i].cost < bestsol.cost:
                bestsol = pop[i].deepcopy()

        logging.info(f"Iteration {it}: Best Cost = {bestsol.cost}")

    # Save results
    with open("results.txt", "w") as f:
        f.write(f"Best Solution: {bestsol.position}\nBest Cost: {bestsol.cost}\n")
