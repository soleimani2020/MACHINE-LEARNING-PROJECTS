#! /usr/bin/ipython3

from   itertools import islice
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from   scipy.optimize import curve_fit
import sys
from matplotlib.pyplot import subplots
import subprocess
from subprocess import call


epsilon= sys.argv[1]
rmin   = sys.argv[2]
r1     = sys.argv[3]
rc     = sys.argv[4]
bFC    = sys.argv[5]
aFC    = sys.argv[6]
Friction    = sys.argv[7]
NFE    = sys.argv[8]
TEMP1=200 
TEMPN=420
INTERVAL=20


###
### Directories
###


destination_folder0='/home/uni08/soleimani/RUNS/Thesis/Membrane_Project/PAPER/4000/'
destination_folder1=destination_folder0+'RUNS0/'
destination_folder2=destination_folder1+'R'+str("-")+str(epsilon)+str("-")+str(rmin)+str("-")+str(r1)+str("-")+str(rc)+str("-")+str(bFC)+str("-")+str(aFC)+str("-")+str(Friction)+str("-")+str(NFE)+''
destination_folder3=destination_folder2+'/Phase_Transition/T-'
destination_folder4=destination_folder2+'/Phase_Transition/'  


###
###
###



First_sample=TEMP1
Last_sample=TEMPN
Interval=INTERVAL
Sample_NUM=int((Last_sample-First_sample)/(Interval))

nan='nan'
nan1='-nan'
def nan_To_num(Mesh_List):
    New_Mesh_list=[]
    for i in Mesh_List:
        if (i != nan)  and (i != nan1):
            New_Mesh_list.append(i)
        else:
            New_Mesh_list.append(1000000000000000)
            
    return  list(New_Mesh_list)



def GET_H0():
    H_List=[]
    FOLDERS=np.arange(First_sample,Last_sample,Interval) 
    for it in FOLDERS:
        destination_folder0=destination_folder3+str(it)+'/Enthalpy0.xvg'
        destination_folder1=destination_folder3+str(it)+'/log_Enthelpy.txt'
        print(destination_folder0)
        if os.path.isfile(destination_folder0):
            with open(destination_folder0) as f0:
                lines=f0.read().splitlines()
                last_line = lines[-1]
                if ("36000.000000" in last_line)  and ("nan" not in last_line) and ("-nan" not in last_line):
                    print("YES. File --Enthalpy0.xvg-- Completion Condition Satisfied!")
                    if os.path.isfile(destination_folder1) != 0 :
                        with open(destination_folder1) as f:
                            next_n_lines = list(islice(f, 5))
                            next_n_lines = list(islice(f,6))         
                            np_lines = np.array(next_n_lines)
                            np_lines_lists = np.char.split(np_lines)
                            np_lines_array = np.array(np_lines_lists.tolist())
                            st=((np_lines_array[1][1]))
                            H_List.append(st)
                        My_List0=(H_List)
                        My_List=nan_To_num(My_List0)
                        destination_folder5=destination_folder3+str(it)+'/'
                        C=subprocess.call('mv Enthalpy0.xvg Enthalpy0_'+str(it)+'.xvg',shell=True,cwd=destination_folder5)
                        C=subprocess.call('cp Enthalpy0_'+str(it)+'.xvg  ../            ',shell=True,cwd=destination_folder5)
                        

H=GET_H0()




def GET_H():
    H_List=[]
    FOLDERS=np.arange(First_sample,Last_sample,Interval) 
    for it in FOLDERS:
        destination_folder0=destination_folder3+str(it)+'/Enthalpy.xvg'
        destination_folder1=destination_folder3+str(it)+'/log_Enthelpy.txt'
        print(destination_folder0)
        if os.path.isfile(destination_folder0):
            with open(destination_folder0) as f0:
                lines=f0.read().splitlines()
                last_line = lines[-1]
                if ("34950.000000" in last_line)  and ("nan" not in last_line) and ("-nan" not in last_line):
                    print("YES. File --Enthalpy0.xvg-- Completion Condition Satisfied!")
                    if os.path.isfile(destination_folder1) != 0 :
                        with open(destination_folder1) as f:
                            next_n_lines = list(islice(f, 5))
                            next_n_lines = list(islice(f,6))         
                            np_lines = np.array(next_n_lines)
                            np_lines_lists = np.char.split(np_lines)
                            np_lines_array = np.array(np_lines_lists.tolist())
                            st=((np_lines_array[1][1]))
                            H_List.append(st)
                        My_List0=(H_List)
                        My_List=nan_To_num(My_List0)
                        destination_folder5=destination_folder3+str(it)+'/'
                        C=subprocess.call('mv Enthalpy.xvg Enthalpy_'+str(it)+'.xvg',shell=True,cwd=destination_folder5)
                        C=subprocess.call('cp Enthalpy_'+str(it)+'.xvg  ../            ',shell=True,cwd=destination_folder5)
                        
                    
                    
                        
                    else:
                        
                        f = open("Report.dat", "w")
                        f.write("File Exists and Completed but STH Wrong in log_Enthelpy.txt !...")
                        f.write("\n")   
                        f.close() 
                        Empty_List=[1000000000000000] * Sample_NUM
                        return Empty_List
                        
                    
                else:
                    
                    f = open("Report.dat", "w")
                    f.write("File Exists But Not Completed or inf Enthalpy !...")
                    f.write("\n")   
                    f.close()
                    Empty_List=[1000000000000000] * Sample_NUM
                    return Empty_List

        else:
            
            f = open("Report.dat", "w")
            f.write("File Doesn't Exist!")
            f.write("\n")   
            f.close()
            Empty_List=[1000000000000000] * Sample_NUM
            return Empty_List
        
        
    return My_List
        

    
H=GET_H()


f = open("Enthalpy.dat", "w")
for i in H:
    f.write(str(i))
    f.write("\n")   
f.close() 


    
def Temperature():
    T_list=np.arange(First_sample,Last_sample,Interval)
    f = open("Temp.dat", "w")
    for i in T_list:
        f.write(str(i))
        f.write("\n")   
    f.close() 
    return T_list

T=Temperature()
Sample_Num=len(T)    
    
    
    
###
### Temperature Vs Enthalpy
###
    
    
    
filenames = ['Temp.dat', 'Enthalpy.dat']
with open('Result.txt', 'w') as writer:
    readers = [open(filename) for filename in filenames]
    for lines in zip(*readers):
        print('     '.join([line.strip() for line in lines]), file=writer)





### Extract data 

def Snapshot():
    filename_temp = "Result.txt"
    filename=filename_temp
    with open(filename) as f:
        next_n_lines = list(islice(f,0))
        next_n_lines = list(islice(f,Sample_Num))
        np_lines = np.array(next_n_lines)    
        np_lines_lists = np.char.split(np_lines) 
        np_lines_array = np.array(np_lines_lists.tolist())
        raw_data = pd.DataFrame(np_lines_array)
        raw_data = raw_data.astype({0:'float64', 1:'float64'}) 
        Mydata = (raw_data).values
        Temperature = list(Mydata[:,0])
        Enthalpy = list(Mydata[:,1])
    return Temperature , Enthalpy  


R=Snapshot()
x=R[0]
Enthalpy=R[1]
p00=np.mean(Enthalpy)
p04=np.mean(x)
Delta_H=np.diff(Enthalpy)
Delta_H=Delta_H[0]
p01=Delta_H



### Derivative Calculation 

def D(xlist,ylist):
    yprime=np.diff(ylist)/np.diff(xlist)
    xprime=[]
    for i in range(len(yprime)):
        xtemp=(xlist[i+1]+xlist[i])/2
        xprime=np.append(xprime,xtemp)
    
    return xprime , yprime
        
        
xprime , yprime = D(x,Enthalpy)
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(xprime,yprime,"-o",label="1st-Derivative")
plt.xlabel('dT')
plt.ylabel('dH')

#xprime2 , yprime2 =  D(xprime,yprime)
#ax.plot(xprime2,yprime2,"-o",label="2nd-Derivative")
#plt.xticks(np.arange(x_min-20, x_max+20, step=20))
#plt.yticks(np.arange(y_min, y_max, step=100))
#plt.legend()




### Finding the transition area 

def Slop(input_list,x):
    if input_list[0]==0 and input_list[1]==0 and input_list[2]==0 and input_list[3]==0:
        T_m=1000
        T_m_Zone=[1000,1010]
        return T_m , T_m_Zone
    else:
        max = input_list[0]
        index = 0
        All_B=[]
        for i in range(0,len(yprime)):
            if input_list[i] > max:
                max = input_list[i]
                index = i
                All_B.append(input_list[index])
                My_All=input_list[-1]
                T_m=(x[index]+x[index+1])/2
                #print(f"Transition zone is in the temperature zone between "+str(x[index])+" and "+str(x[index+1])+".") 
                T_m_Zone=[x[index],x[index+1]]

            elif input_list[0] > input_list[1] and input_list[0] > input_list[2] and input_list[0] > input_list[3] :
                T_m=210.1234
                T_m_Zone=[200,210]
        return T_m , T_m_Zone


Expected_Tm=Slop(yprime,x)[0]
print("Expected_Tm:\n",Expected_Tm)

f = open("Expected_Tm.dat", "w")
f.write(str(Expected_Tm))
f.write("\n")   
f.close()



Expected_Tm_zone=Slop(yprime,x)[1]
print("Expected_Tm_Zone:\n",Expected_Tm_zone)

f = open("Expected_Tm_Zone.dat", "w")
f.write(str(Expected_Tm_zone))
f.write("\n")   
f.close()



###
### Fitting Phase Transition Temperature 
###




def FIT():
    R=Snapshot()
    if (R[1][0]==1000000000000000):
        f = open("T_m.dat", "w")
        f.write(str(0))
        f.write("\n")   
        f.close() 
        
    elif Expected_Tm == 210.1234 :
        f = open("T_m.dat", "w")
        f.write(str(Expected_Tm))
        f.write("\n")   
        f.close()
    
    else:
        try:
            
            def model2(x,A1,DeltaA,x0,k,c):
                FH =A1+c*x+(DeltaA)/(1+np.exp(k*(x-x0)))
                return FH

            p0 = [p00,p01,Expected_Tm,0,0] #  initial guess
            popt, pcov = curve_fit(model2,x,Enthalpy,p0, maxfev = 100000)
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(x,Enthalpy,'bo')

            ###
            xstart=First_sample
            xstop=Last_sample
            increment=0.1
            xmodel=np.arange(xstart,xstop,increment)
            A1=popt[0]
            DeltaA=popt[1]
            x0=popt[2]
            k=popt[3]
            c=popt[4]
            plt.axvline(x0, color='black',linestyle='--',linewidth=1)
            ymodel=model2(xmodel,A1,DeltaA,x0,k,c)
            plt.plot(xmodel,ymodel,'r', label="T_M:"+str(x0)+".")
            plt.legend(loc='lower right')
            plt.xlabel('Temperature(k)')
            plt.ylabel('Enthalpy(kJ/mol)')
            plt.xticks(np.arange(First_sample, Last_sample, step=Interval))
            plt.savefig("Phase_Transition_Temperature.png")
            if  TEMP1 < x0 < TEMPN :
                f = open("T_m.dat", "w")
                f.write(str(x0))
                f.write("\n")   
                f.close() 
            else:
                
                f = open("T_m.dat", "w")
                f.write(str(Expected_Tm))
                f.write("\n")   
                f.close() 
        
        except:
            
            f = open("T_m.dat", "w")
            f.write(str(0))
            f.write("\n")   
            f.close() 
        pass
        
       

        


T_m=FIT()



        
        
###
### Add Phase Transition Temperature to other targets 
###
        
try:        
            
    P=os.system("cp   T_m.dat ../../  ")    
    O=subprocess.call('cat T_m.dat >>   RESULTS'+str("_")+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+'.txt  ',shell=True,cwd=destination_folder2)    
        
except:
    print("Fit Not working..")
    pass

###
### Plot Enthalpy 
###
    

O=subprocess.call('python3 ./Plot_ALL_H.py  ',shell=True,cwd=destination_folder4)    
    

    



    
    
    
    
    





























    
    
    





