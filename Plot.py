import os
import numpy as np
import pandas as pd
import math
from   itertools import islice
from   scipy.optimize import curve_fit
import sys
import itertools 
from itertools import islice
import matplotlib.pyplot as plt






skiprow=26


def Density_Profile():
    if os.path.isfile("density.xvg"):
        fig, ax2 = plt.subplots(figsize=(6,4))
        DP = np.loadtxt("density.xvg", skiprows=skiprow)
        ax2.plot(DP[:,0],DP[:,1], '-o', label="System")
        DP = np.loadtxt("density.xvg", skiprows=skiprow)
        ax2.plot(DP[:,0],DP[:,2], '-o', label="T")
        DP = np.loadtxt("density.xvg", skiprows=skiprow)
        ax2.plot(DP[:,0],DP[:,3], '-o', label="H")
        plt.legend(loc='upper right')
        ax2.set_xlabel('Distance/Angestrum',fontsize=12)
        ax2.set_ylabel('g(r) ',fontsize=12)
        plt.savefig("DP.png")

       

    else:
        print("This simulation with parameters faced error.PP can not be plotted.\n")
        
           


A=Density_Profile()
print(A)




def Enthalpy():
    if os.path.isfile("Enthalpy.xvg"):
        fig, ax2 = plt.subplots(figsize=(6,4))
        DP = np.loadtxt("Enthalpy.xvg", skiprows=skiprow)
        ax2.plot(DP[:,0],DP[:,1], '--')
        plt.legend(loc='upper right')
        ax2.set_xlabel('Time(ps)',fontsize=12)
        ax2.set_ylabel('Enthalpy(kj/mol) ',fontsize=12)
        plt.savefig("Enthalpy.png")

       

    else:
        print("This simulation with parameters faced error.PP can not be plotted.\n")
    



B=Enthalpy()
print(B)































