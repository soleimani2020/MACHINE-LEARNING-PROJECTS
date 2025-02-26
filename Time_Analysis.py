import glob
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
import statistics
from statistics import mean
from scipy.stats import norm

dir_path ="/home/uni08/soleimani/RUNS/Thesis/Membrane_Project/PAPER/4000/TIME"
Sim_Number = [f for f in glob.glob("TIME.Threetxt*")]
print("Sim_Number:",len(Sim_Number))




def get_txt(name):
    res = []
    L=[]
    for i in range(1,len(Sim_Number)+1):
        S=name+str(i)+""
        L.append(S)
    Ordered_L=L
    return Ordered_L



######
######
######




        
Sample_Num=0
Sample_Pass=1

def Time(name):
    Time_X = get_txt(name)
    All_Files=[]
    for File in Time_X:
        filename =File
        with open(filename) as f:
            next_n_lines = list(islice(f, Sample_Num))
            next_n_lines = list(islice(f, Sample_Pass))         
            np_lines = np.array(next_n_lines)                
            np_lines_lists = np.char.split(np_lines)        
            np_lines_array = np.array(np_lines_lists.tolist())
            raw_data = pd.DataFrame(np_lines_array)
            raw_data_pxx = raw_data.astype({0:'str'})
            All_Files.append(raw_data_pxx[0][0])
    return All_Files




A=Time("TIME.Onetxt")
f = open("Time_One.dat", "w")
for i in A:
    f.write(str(i))
    f.write("\n")   
f.close() 
print("\n")







B=Time("TIME.Twotxt")
f = open("Time_TWO.dat", "w")
for i in B:
    f.write(str(i))
    f.write("\n")   
f.close() 
print("\n")





C=Time("TIME.Threetxt")
f = open("Time_Three.dat", "w")
for i in C:
    f.write(str(i))
    f.write("\n")   
f.close() 
print("\n")



print("Number of finished simulations:",len(C))

NUM=np.arange(len(C))

f = open("Number.dat", "w")
for i in NUM:
    f.write(str(i+1))
    f.write("\n")   
f.close() 
print("\n")





filenames = [ 'Number.dat', 'Time_One.dat','Time_TWO.dat', 'Time_Three.dat']
with open('output.txt', 'w') as writer:
    readers = [open(filename) for filename in filenames]
    for lines in zip(*readers):
        print('     '.join([line.strip() for line in lines]), file=writer)
                
                
               

                
                
                






