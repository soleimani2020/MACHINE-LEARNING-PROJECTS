import os
import numpy as np
import pandas as pd
import math
from   itertools import islice
import sys
import itertools 
from itertools import islice
import matplotlib.pyplot as plt
import sys
import statistics
from statistics import mean
from scipy.stats import norm
import Conversion




epsilon= sys.argv[1]
rmin   = sys.argv[2]
r1     = sys.argv[3]
rc     = sys.argv[4]
bFC    = sys.argv[5]
aFC    = sys.argv[6]
Friction    = sys.argv[7]
NFE    = sys.argv[8]



Sample_Num=24
Sample_Pass=1691-24    ### Number of samples
Lipid_Num=816          ### Number of lipids in "One" leaftlet




###############################
###############################
############################### 


def Average(lst): 
    return np.mean(lst)

def Time(Sample_Num,Sample_Pass):
    filename ="LX_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg"
    if os.path.isfile(filename):
        with open("LX_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg") as f0:
            lines=f0.read().splitlines()
            last_line = lines[-1]
            if "252000.000000" in last_line:
                with open(filename) as f:
                    next_n_lines = list(islice(f, Sample_Num))
                    next_n_lines = list(islice(f, Sample_Pass))         
                    np_lines = np.array(next_n_lines)                
                    np_lines_lists = np.char.split(np_lines)        
                    np_lines_array = np.array(np_lines_lists.tolist())
                    raw_data = pd.DataFrame(np_lines_array)
                    raw_data = raw_data.astype({0:'float64'})
                    Time=[]
                    for i in range(0,Sample_Pass):
                        Time.append(raw_data[0][i])
                        My_Time = [float(i) for i in Time]
                    T = My_Time
                return list(T) 
            return 10000  # due to simulation Error
    return 20000  #  due to segmentation fault



###############################
###############################
############################### 


def Membrane_BOX_LX(Sample_Num,Sample_Pass):
    filename ="LX_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg"
    if os.path.isfile(filename):
        with open("LX_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg") as f0:
            lines=f0.read().splitlines()
            last_line = lines[-1]
            if "252000.000000" in last_line:
                with open(filename) as f:
                    next_n_lines = list(islice(f, Sample_Num))
                    next_n_lines = list(islice(f, Sample_Pass))         
                    np_lines = np.array(next_n_lines)                
                    np_lines_lists = np.char.split(np_lines)        
                    np_lines_array = np.array(np_lines_lists.tolist())
                    raw_data = pd.DataFrame(np_lines_array)
                    raw_data_pxx = raw_data.astype({0:'float64'})
                    Pxx=[]
                    for i in range(0,Sample_Pass):
                        Pxx.append(raw_data_pxx[1][i])
                        My_Pxx = [float(i) for i in Pxx]
                    LXX = list(My_Pxx)
                    
                return LXX
                    
                    
            return 10000
        
    return 20000

                    
 

###############################
###############################
############################### 


def Membrane_BOX_LY(Sample_Num,Sample_Pass):
    filename ="LX_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg"
    if os.path.isfile(filename):
        with open("LX_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg") as f0:
            lines=f0.read().splitlines()
            last_line = lines[-1]
            if "252000.000000" in last_line:
                with open(filename) as f:
                    next_n_lines = list(islice(f, Sample_Num))
                    next_n_lines = list(islice(f, Sample_Pass))         
                    np_lines = np.array(next_n_lines)                
                    np_lines_lists = np.char.split(np_lines)        
                    np_lines_array = np.array(np_lines_lists.tolist())
                    raw_data = pd.DataFrame(np_lines_array)
                    raw_data_pyy = raw_data.astype({0:'float64'})
                    Pyy=[]
                    for i in range(0,Sample_Pass):
                        Pyy.append(raw_data_pyy[1][i])
                        My_Pyy = [float(i) for i in Pyy]
                    LYY = list(My_Pyy)
                    
                return LYY
                    
                    
            return 10000
        
    return 20000

                    
 
 
###############################
###############################
############################### 
 
 
def Membrane_Area(Sample_Num,Sample_Pass):
    MEM_BOX_LX=Membrane_BOX_LX(Sample_Num,Sample_Pass)
    MEM_BOX_LY=Membrane_BOX_LY(Sample_Num,Sample_Pass)
    products=[]
    APL=[]
    for num1, num2 in zip(MEM_BOX_LX, MEM_BOX_LY):
        products.append(num1 * num2)
        APL.append((num1 * num2)/Lipid_Num)
    
    return products , APL

    
###############################
###############################
###############################
# Area per lipid calculation 

def Plot_APL(Sample_Num,Sample_Pass,epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE):
    MEM_BOX_LX=Membrane_BOX_LX(Sample_Num,Sample_Pass)
    MEM_BOX_LY=Membrane_BOX_LY(Sample_Num,Sample_Pass)
    if (MEM_BOX_LX or MEM_BOX_LY)==10000:
        APL=10000
    elif (MEM_BOX_LX or MEM_BOX_LY)==20000:
        APL=20000
    else:
        APL=Membrane_Area(Sample_Num,Sample_Pass)[1]
        APL=APL[-667:]
        T=Time(Sample_Num,Sample_Pass)
        New_T=[]
        for i in  T:
            New_T.append(i/1000)
        T_ns=New_T[-667:]
        T_min=np.min(T_ns)
        T_max=np.max(T_ns)
        mean = statistics.mean(APL)
        sd = statistics.stdev(APL)
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(T_ns,APL,'bo')
        plt.legend(loc='upper right')
        plt.xlabel('Time/ ns')
        plt.ylabel('APL($nm^2$)')
        plt.xticks(np.arange(T_min, T_max+5, step=20))
        plt.ylim([0.2, 0.9])
        plt.xlim([T_min, T_max+5])
        plt.subplot(1, 2, 2)
        plt.plot(APL, norm.pdf(APL, mean, sd),'bo')
        plt.axvline(mean, color='grey',linestyle='--',linewidth=2 , label="Mean:"+str(mean)+".std:"+str(sd)+".")
        plt.axvline(mean-sd, color='black',linestyle='--',linewidth=2)
        plt.axvline(mean+sd, color='black',linestyle='--',linewidth=2)
        plt.xlim([mean-2*sd, mean+2*sd])
        plt.legend(loc='upper right')
        plt.savefig("APL_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".png")
        APL=mean
    
        return APL
    return APL


A=Plot_APL(Sample_Num,Sample_Pass,epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)


###############################
###############################
###############################
# Area Area_Compressibility calculation based on the projected area of the simulation box 

def Area_Compressibility1(Sample_Num,Sample_Pass,epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE):
    MEM_BOX_LX=Membrane_BOX_LX(Sample_Num,Sample_Pass)
    MEM_BOX_LY=Membrane_BOX_LY(Sample_Num,Sample_Pass)
    if (MEM_BOX_LX or MEM_BOX_LY)==10000:       # Segmentation fault 
        TARGET_UNIT_KA=10000
        APL_Mean=10000
        
    elif (MEM_BOX_LX or MEM_BOX_LY)==20000:     # Simulation Error 
        TARGET_UNIT_KA=20000
        APL_Mean=20000
    
    else:
        
        APL_Mean=Plot_APL(Sample_Num,Sample_Pass,epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)
        Projected_Area=Membrane_Area(Sample_Num,Sample_Pass)[0]
        #f = open("Projected_Area.dat", "w")
        #for i in Projected_Area:
        #    f.write(str(i))
        #    f.write("\n")   
        #f.close() 
        Projected_Area_pow=[float(i)*float(i) for i in Projected_Area]
        #f = open("Projected_Area_pow.dat", "w")
        #for i in Projected_Area_pow:
        #    f.write(str(i))
        #    f.write("\n")   
        #f.close() 
        Ave_Projected_Area_pow = Average(Projected_Area_pow)
        #print("<Projected_Area^2>:\n",Ave_Projected_Area_pow)
        #f = open("Ave_Projected_Area_pow.dat", "w")
        #f.write(str(Ave_Projected_Area_pow))
        #f.write("\n")   
        #f.close() 
        Ave_Projected_Area = Average(Projected_Area)
        #print("<Projected_Area>:\n",Ave_Projected_Area)
        #f = open("Ave_Projected_Area.dat", "w")
        #f.write(str(Ave_Projected_Area))
        #f.write("\n")   
        #f.close() 
        Pow_Projected_Area=Ave_Projected_Area**2
        #print("<Projected_Area>^2:\n",Pow_Projected_Area)
        #f = open("Pow_Projected_Area.dat", "w")
        #f.write(str(Pow_Projected_Area))
        #f.write("\n")   
        #f.close() 
        #print("Experimental value for Area Compressibility is 230 [mN/m]. \n" )
        GROMACS_UNIT_KA=(Ave_Projected_Area)/(Ave_Projected_Area_pow-Pow_Projected_Area)  #kT/(nm^2)
        #f = open("GROMACS_UNIT_KA.dat", "w")
        #f.write(str(GROMACS_UNIT_KA))
        #f.write("\n")   
        #f.close() 
        TARGET_UNIT_KA=Conversion.AreaCompressibility_Unit_Conversion(GROMACS_UNIT_KA)    # mN/m
        return TARGET_UNIT_KA , APL_Mean
    
    return TARGET_UNIT_KA , APL_Mean


###############################
###############################
###############################
# Area Area_Compressibility calculation based on area per lipid 

def Area_Compressibility2(Sample_Num,Sample_Pass,epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE):
    MEM_BOX_LX=Membrane_BOX_LX(Sample_Num,Sample_Pass)
    MEM_BOX_LY=Membrane_BOX_LY(Sample_Num,Sample_Pass)
    if (MEM_BOX_LX or MEM_BOX_LY)==10000:       # Simulation Error  
        TARGET_UNIT_KA=10000
        APL_Mean=10000
        
    elif (MEM_BOX_LX or MEM_BOX_LY)==20000:     # Segmentation fault 
        TARGET_UNIT_KA=20000
        APL_Mean=20000
    
    else:
        
        
        APL_Mean=Plot_APL(Sample_Num,Sample_Pass,epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)
        Lipid_Num_total=816*2
        Lipid_Num_PLT=816
        APL=Area_Per_Lipid(Sample_Num,Sample_Pass,epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)[0]
        APL_POW2=[x**2 for x in APL]
        APL_POW2_Ave=np.mean(APL_POW2)
        APL_Ave=np.mean(APL)
        APL_Ave_POW2=APL_Ave**2
        GROMACS_UNIT_KA=(APL_Ave)/((Lipid_Num_PLT)*(APL_POW2_Ave-APL_Ave_POW2))
        TARGET_UNIT_KA=Conversion.AreaCompressibility_Unit_Conversion(GROMACS_UNIT_KA)
        

        return TARGET_UNIT_KA , APL_Mean

    return TARGET_UNIT_KA , APL_Mean



###############################
###############################
###############################
# Area compressibility calculation based on fluctuation in LX


def Area_Compressibility3(Sample_Num,Sample_Pass,epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE):
    MEM_BOX_LX=Membrane_BOX_LX(Sample_Num,Sample_Pass)
    MEM_BOX_LY=Membrane_BOX_LY(Sample_Num,Sample_Pass)
    if (MEM_BOX_LX or MEM_BOX_LY)==10000:       # Simulation Error 
        TARGET_UNIT_KA=10000
        APL_Mean=10000
        
    elif (MEM_BOX_LX or MEM_BOX_LY)==20000:     # Segmentation fault  
        TARGET_UNIT_KA=20000
        APL_Mean=20000
    
    else:
        APL_Mean=Plot_APL(Sample_Num,Sample_Pass,epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)
        Lipid_Num_total=816*2
        Lipid_Num_PLT=816
        LX0=Membrane_BOX_LX(Sample_Num,Sample_Pass)
        LX0=LX0[300:]
        A=[]
        B=[]
        C=[]
        D=[]
        E=[]
        F=[]
        for i in range(len(LX0),10,-1):
            LX=LX0[0:i]
            #print(LX)
            Mean_LX=np.mean(LX)
            Mean_LX_Pow2=Mean_LX**2
            LX_pow2=[x**2 for x in LX]
            LX_pow2_mean=np.mean(LX_pow2)
            GROMACS_UNIT_KA=(1)/((4)*(LX_pow2_mean-Mean_LX_Pow2))
            TARGET_UNIT_KA=Conversion.AreaCompressibility_Unit_Conversion(GROMACS_UNIT_KA)
            #print(TARGET_UNIT_KA)
            A.append(Mean_LX)
            B.append(Mean_LX_Pow2)
            C.append(LX_pow2)
            D.append(LX_pow2_mean)
            E.append(GROMACS_UNIT_KA)
            F.append(TARGET_UNIT_KA)
        My_A=A
        My_B=B
        My_C=C
        My_D=D
        My_E=E
        My_F=list(reversed(F))
        return (My_F)
    return TARGET_UNIT_KA , APL_Mean


#A=Area_Compressibility3(Sample_Num,Sample_Pass,epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)
#print(A)


#f = open("AC.dat", "w")
#for i in A:
#    f.write(str(i))
#    f.write("\n")   
#f.close() 



###############################
###############################
###############################
# Plotting Area compressibility  as a function of time  

def Plot_AC(Sample_Num,Sample_Pass,epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE):
    MEM_BOX_LX=Membrane_BOX_LX(Sample_Num,Sample_Pass)
    MEM_BOX_LY=Membrane_BOX_LY(Sample_Num,Sample_Pass)
    if (MEM_BOX_LX or MEM_BOX_LY)==10000:
        TARGET_UNIT_KA=10000
        APL_Mean=10000
    elif (MEM_BOX_LX or MEM_BOX_LY)==20000:
        TARGET_UNIT_KA=20000
        APL_Mean=20000
    else:
        
        APL_Mean=Plot_APL(Sample_Num,Sample_Pass,epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)
        AC=Area_Compressibility3(Sample_Num,Sample_Pass,epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)
        AC=AC[-667:]  #667
        AC_Min=np.min(AC)
        AC_Max=np.max(AC)
        T=Time(Sample_Num,Sample_Pass)        
        New_T=[]
        for i in range(0,len(T)):  
            New_T.append(T[i]/1000)
        T_ns=New_T[-667:]
        T_min=np.min(T_ns)
        T_max=np.max(T_ns)
        mean = statistics.mean(AC)
        sd = statistics.stdev(AC)
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(T_ns,AC,'ro')
        plt.legend(loc='upper right')
        plt.xlabel('Time/ ns')
        plt.ylabel('AC($mN/m$)')
        plt.xticks(np.arange(T_min, T_max+5, step=20))
        plt.yticks(np.arange(AC_Min-100, AC_Max+100, step=50))
        
        #plt.ylim([AC_Min-50, AC_Max+50])
        plt.xlim([T_min, T_max+5])
        plt.subplot(1, 2, 2)
        plt.plot(AC, norm.pdf(AC, mean, sd),'ro')
        plt.axvline(mean, color='grey',linestyle='--',linewidth=2 , label="Mean:"+str(mean)+".std:"+str(sd)+".")
        plt.axvline(mean-sd, color='black',linestyle='--',linewidth=2)
        plt.axvline(mean+sd, color='black',linestyle='--',linewidth=2)
        plt.legend(loc='upper right')
        plt.savefig("AC_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".png")
        TARGET_UNIT_KA=mean
    
        return TARGET_UNIT_KA , APL_Mean
    return TARGET_UNIT_KA , APL_Mean





###############################
###############################
###############################



#Area Compressibility Calculation
#A=Plot_AC(Sample_Num,Sample_Pass,epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)
#print(A[0])
  

#Area Per Lipid Calculation  
#print(A[1])
    




A=0#Area_Compressibility(Sample_Num,Sample_Pass,epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)[0]
print(A)
  


#Area Per Lipid Calculation  
B=0#Area_Compressibility(Sample_Num,Sample_Pass,epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)[1]
print(B)
    



