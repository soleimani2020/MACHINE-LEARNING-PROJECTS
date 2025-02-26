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










def Pressure_Profile(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE):
    if os.path.isfile("PP_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg"):
        fig, ax2 = plt.subplots(figsize=(6,4))
        #PP = np.loadtxt("PP_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+".xvg", skiprows=25)
        #ax2.plot(PP[:,0],PP[:,1], '-o', label="Pxx")
        #PP = np.loadtxt("PP_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+".xvg", skiprows=25)
        #ax2.plot(PP[:,0],PP[:,2], '-o', label="Pyy")
        #PP = np.loadtxt("PP_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+".xvg", skiprows=25)
        #ax2.plot(PP[:,0],PP[:,3], '-o', label="Pzz")
        PP = np.loadtxt("PP_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg", skiprows=25)
        ax2.plot(PP[:,0],PP[:,4], '-o', label="PT-PN")
        plt.legend(loc='upper right')
        ax2.set_xlabel('Distance/Angestrum',fontsize=12)
        ax2.set_ylabel('P ',fontsize=12)
        plt.savefig("PP_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".png")
        
    else:
        print("This simulation with parameters:"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+"faced error.PP can not be plotted.\n")
        
           


A=Pressure_Profile(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)
print(A)












































