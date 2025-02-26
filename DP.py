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





epsilon= sys.argv[1]
rmin   = sys.argv[2]
r1     = sys.argv[3]
rc     = sys.argv[4]
bFC    = sys.argv[5]
aFC    = sys.argv[6]
Friction    = sys.argv[7]
NFE    = sys.argv[8]
Fitness   = sys.argv[9]



skiprow=26


def Density_Profile(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE,Fitness):
    if os.path.isfile("density_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg"):
        fig, ax2 = plt.subplots(figsize=(6,4))
        DP = np.loadtxt("density_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg", skiprows=skiprow)
        ax2.plot(DP[:,0],DP[:,1], '-o', label="System")
        DP = np.loadtxt("density_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg", skiprows=skiprow)
        ax2.plot(DP[:,0],DP[:,2], '-o', label="T")
        DP = np.loadtxt("density_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg", skiprows=skiprow)
        ax2.plot(DP[:,0],DP[:,3], '-o', label="H")
        plt.legend(loc='upper right')
        ax2.set_xlabel('Distance/Angestrum',fontsize=12)
        ax2.set_ylabel('g(r) ',fontsize=12)
        plt.savefig("DP_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+str("_")+str(Fitness)+".png")

       

    else:
        print("This simulation with parameters:"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+" faced error.PP can not be plotted.\n")
        
           


A=Density_Profile(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE,Fitness)
print(A)







































