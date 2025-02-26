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
import Conversion
import statistics
from statistics import mean
from scipy.stats import norm


epsilon= sys.argv[1]
rmin   = sys.argv[2]
r1     = sys.argv[3]
rc     = sys.argv[4]
bFC    = sys.argv[5]
aFC    = sys.argv[6]
Friction    = sys.argv[7]
NFE    = sys.argv[8]


Sample_Num=25
Sample_Pass=1680000


def Average(lst): 
    return np.mean(lst)

def Time(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE):
    filename ="Pxx_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg"
    with open(filename) as f:
        next_n_lines = list(islice(f, Sample_Num))
        next_n_lines = list(islice(f, Sample_Pass))         
        np_lines = np.array(next_n_lines)                
        np_lines_lists = np.char.split(np_lines)        
        np_lines_array = np.array(np_lines_lists.tolist())
        raw_data = pd.DataFrame(np_lines_array)
        raw_data = raw_data.astype({0:'float64'})
        Time=[]
        for i in range(0,Sample_Pass,1000):
            Time.append(raw_data[0][i])
            #print(raw_data[0][i])
            My_Time = [float(i) for i in Time]
        T = My_Time
    return list(T)



#A=Time(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)
#print(len(A))




def Pressure_Components(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE):
    filename ="Pxx_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg"
    with open(filename) as f:
        next_n_lines = list(islice(f, Sample_Num))
        next_n_lines = list(islice(f, Sample_Pass))         
        np_lines = np.array(next_n_lines)                
        np_lines_lists = np.char.split(np_lines)        
        np_lines_array = np.array(np_lines_lists.tolist())
        raw_data = pd.DataFrame(np_lines_array)
        raw_data_pxx = raw_data.astype({0:'float64'})
        Pxx=[]
        for i in range(0,Sample_Pass,1000):
            Pxx.append(raw_data_pxx[1][i])
            My_Pxx = [float(i) for i in Pxx]
        PXX = list(My_Pxx)
    filename ="Pyy_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg"
    with open(filename) as f:
        next_n_lines = list(islice(f, Sample_Num))
        next_n_lines = list(islice(f, Sample_Pass))         
        np_lines = np.array(next_n_lines)                
        np_lines_lists = np.char.split(np_lines)        
        np_lines_array = np.array(np_lines_lists.tolist())
        raw_data = pd.DataFrame(np_lines_array)
        raw_data_pyy = raw_data.astype({0:'float64'})
        Pyy=[]
        for i in range(0,Sample_Pass,1000):
            Pyy.append(raw_data_pyy[1][i])
            My_Pyy = [float(i) for i in Pyy]
        PYY = list(My_Pyy)
    filename ="Pzz_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg"
    with open(filename) as f:
        next_n_lines = list(islice(f, Sample_Num))
        next_n_lines = list(islice(f, Sample_Pass))         
        np_lines = np.array(next_n_lines)                
        np_lines_lists = np.char.split(np_lines)        
        np_lines_array = np.array(np_lines_lists.tolist())
        raw_data = pd.DataFrame(np_lines_array)
        raw_data_pzz = raw_data.astype({0:'float64'})
        Pzz=[]
        for i in range(0,Sample_Pass,1000):
            Pzz.append(raw_data_pzz[1][i])
            My_Pzz = [float(i) for i in Pzz]
        PZZ = list(My_Pzz)
    return (PXX) , (PYY) , (PZZ)








#A=Pressure_Components(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)[2]
#print((A))






def Box_Area(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE):
    LX=20
    LZ=15
    return LX , LZ  ### Non-periodic dimensions / sourrounded by water 


def Line_Tension(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE):
    Pxx=Pressure_Components(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)[0]
    Pyy=Pressure_Components(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)[1]
    Pzz=Pressure_Components(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)[2]
    LX=Box_Area(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)[0]
    LZ=Box_Area(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)[1]
    LT=[]
    for i in range(0,len(Pxx)):
        GROMACS_UNIT=  -1*LX*LZ*0.5*((Pyy[i])-(0.5*(Pxx[i]+Pzz[i])))  # Ref:Free energy of a trans-membrane pore calculated from atomistic molecular dynamics simulations
        #GROMACS_UNIT= -1*LX*LZ*0.5*((Pyy[i])-(Pxx[i]))
        #GROMACS_UNIT= -1*LX*LZ*0.5*((Pyy[i])-(Pzz[i]))
        TARGET_UNIT=Conversion.LineTension_Unit_Conversion(GROMACS_UNIT)
        LT.append(TARGET_UNIT)  # nm^2.bar =0.1pN
    LT_List=LT
    Mean_LT=Average(LT_List)  #pN
    file_out0='Mean_LT.dat'
    fo = open(file_out0,'w')
    fo.write(str(Mean_LT))
    fo.write("\n")
    fo.close
    return LT_List , Mean_LT


        
def Plot_Lt(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE):
    LT=Line_Tension(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)[0]
    LT=LT[-680:]  
    LT_Min=np.min(LT)
    LT_Max=np.max(LT)
    T=Time(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)
    New_T=[]
    for i in  T:
        New_T.append(i/1000)
    T_ns=New_T[-680:]
    T_min=np.min(T_ns)
    T_max=np.max(T_ns)
    mean = statistics.mean(LT)
    sd = statistics.stdev(LT)
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(T_ns,LT,'o-')
    plt.legend(loc='upper right')
    plt.xlabel('Time/ ns')
    plt.ylabel('Line Tension(pN)')
    Step=int((LT_Max-LT_Min)/10)
    plt.xticks(np.arange(T_min, T_max+5, step=20))
    plt.yticks(np.arange(LT_Min, LT_Max, step=Step))
    plt.xlim([T_min, T_max+5])
    plt.subplot(1, 2, 2)
    plt.plot(LT, norm.pdf(LT, mean, sd),'bs')
    plt.axvline(mean, color='grey',linestyle='--',linewidth=2 , label="Mean:"+str(mean)+". std:"+str(sd)+".")
    plt.axvline(mean-sd, color='black',linestyle='--',linewidth=2)
    plt.axvline(mean+sd, color='black',linestyle='--',linewidth=2)
    plt.legend(loc='upper right')
    plt.savefig("LT_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".png")
    plt.show()

    
    
    
    
    

def AAA(Sample_Num,Sample_Pass,epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE):
    if os.path.isfile("Pxx_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg"):
        with open("Pxx_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg") as f0:
            lines=f0.read().splitlines()
            last_line = lines[-1]
            if "252000.000000" in last_line:
                B=Plot_Lt(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)
                LT=Line_Tension(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)[1]
                return LT
            else:
                return 10000
    else:
        return 20000
                
            

            
       
LT=0#AAA(Sample_Num,Sample_Pass,epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)
print(LT)































