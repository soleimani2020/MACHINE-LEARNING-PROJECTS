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
from mpl_toolkits.mplot3d import Axes3D


skiprow=26




def Enthalpy_Profile():
    if os.path.isfile("Enthalpy_200.xvg") and os.path.isfile("Enthalpy_220.xvg") and os.path.isfile("Enthalpy_240.xvg") and os.path.isfile("Enthalpy_260.xvg") and os.path.isfile("Enthalpy_280.xvg") and os.path.isfile("Enthalpy_300.xvg") and os.path.isfile("Enthalpy_320.xvg") and os.path.isfile("Enthalpy_340.xvg") and os.path.isfile("Enthalpy_360.xvg") and os.path.isfile("Enthalpy_380.xvg") and os.path.isfile("Enthalpy_400.xvg"):
        All_F=[]
        for Temp in range(400,190,-20):
            F="Enthalpy_"+str(Temp)+".xvg"
            All_F.append(F)
        
        fig, ax2 = plt.subplots(figsize=(6,4))
        for F in All_F:
            DP = np.loadtxt(F, skiprows=skiprow)
            ax2.plot(DP[:,0],DP[:,1], '--', label="T_"+str(F)+"")
        plt.legend(loc='upper right')
        ax2.set_xlabel('Time',fontsize=12)
        ax2.set_ylabel('Enthalpy(kJ/mol) ',fontsize=12)
        #ax2.set_xlim([0, 290000])
        plt.savefig("Enthalpy.png")
        
    else:
        
        f = open("Error.dat", "w")
        f.write("No enthalpy! The simulation faced Error...")
        f.write("\n")   
        f.close() 


A=Enthalpy_Profile()




x0=30000
x1=35000




def Enthalpy_Profile2():
    if os.path.isfile("Enthalpy0_200.xvg") and os.path.isfile("Enthalpy0_220.xvg") and os.path.isfile("Enthalpy0_240.xvg") and os.path.isfile("Enthalpy0_260.xvg") and os.path.isfile("Enthalpy0_280.xvg") and os.path.isfile("Enthalpy0_300.xvg") and os.path.isfile("Enthalpy0_320.xvg") and os.path.isfile("Enthalpy0_340.xvg") and os.path.isfile("Enthalpy0_360.xvg") and os.path.isfile("Enthalpy0_380.xvg") and os.path.isfile("Enthalpy0_400.xvg"):
        All_F=[]
        for Temp in range(400,190,-20):
            F="Enthalpy0_"+str(Temp)+".xvg"
            All_F.append(F)
        
        fig, ax2 = plt.subplots(figsize=(6,4))
        for F in All_F:
            DP = np.loadtxt(F, skiprows=skiprow)
            ax2.plot(DP[:,0],DP[:,1], '--', label="T_"+str(F)+"")
        plt.legend(loc='upper right')
        ax2.set_xlabel('Time',fontsize=12)
        ax2.set_ylabel('Enthalpy(kJ/mol) ',fontsize=12)
        ax2.axvline(x0, color='black',linestyle='--',linewidth=1)
        ax2.axvline(x1, color='black',linestyle='--',linewidth=1)
        #ax2.set_xlim([0, 290000])
        plt.savefig("Enthalpy0.png")
        
    else:
        
        f = open("Error.dat", "w")
        f.write("No enthalpy! The simulation faced Error...")
        f.write("\n")   
        f.close() 


A=Enthalpy_Profile2()



















