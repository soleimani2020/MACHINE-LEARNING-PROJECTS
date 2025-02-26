#! /usr/bin/ipython3
import os
import numpy as np
import pandas as pd
from   itertools import islice
from   ypstruct  import structure 
import sys
import itertools 
from itertools import islice
import subprocess
import random
import time




epsilon= sys.argv[1]
rmin   = sys.argv[2]
r1     = sys.argv[3]
rc     = sys.argv[4]
bFC    = sys.argv[5]
aFC    = sys.argv[6]
Friction    = sys.argv[7]
NFE    = sys.argv[8]




def Enthalpy_Gen0(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE):
    file_name = "Enthalpy0.sh"
    completeName = os.path.join( file_name)
    file = open(completeName,"w")
    file.write('echo  "18"  | gmx energy -f npt_'+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+'.edr  -o  Enthalpy0.xvg                                                      \n')
    file.write("                                                                                                                                         \n")
    file.close()




def Enthalpy_Gen(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE):
    file_name = "Enthalpy.sh"
    completeName = os.path.join( file_name)
    file = open(completeName,"w")
    file.write('echo  "18"  | gmx energy -f npt_'+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+'.edr  -o  Enthalpy.xvg    -b 30000 -e 35000        > log_Enthelpy.txt                                                                                          \n')
    file.write("                                                                                                                                         \n")
    file.close()





def Calc_Enthalpy(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE):
    file_name = "calculation.sh"
    completeName = os.path.join( file_name)
    file = open(completeName,"w")
    file.write('echo  "0"  "2"  "3"  | gmx density -f npt_'+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+'.trr -n  index-'+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+'.ndx  -s npt_'+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+'.tpr -o density.xvg -ng 3 -sl 1000                                                                                                                                       \n')
    file.write("#########################                                                                                                                \n")
    file.write("> RESULTS.txt                                                                                                                            \n")
    file.write("outputString1=$(python  ./Enthalpy.py  )                                                                                                 \n")                                                                                                                                                                                                                                                                               
    file.write("echo ${outputString1} >> RESULTS.txt                                                                                                     \n")
    file.write("#########################                                                                                                                \n")
    file.write("rm \#*                                                                                                                                   \n")
    file.close()



def Calc_Enthalpy2(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE):
    file_name = "Enthalpy.py"
    completeName = os.path.join(file_name)
    file = open(completeName,"w")
    file.write("import os                                                                                                                                                   \n")
    file.write("import numpy as np                                                                                                                                          \n")
    file.write("import pandas as pd                                                                                                                                         \n")
    file.write("from   itertools import islice                                                                                                                              \n")
    file.write("import math                                                                                                                                                 \n")
    file.write("                                                                                                                                                            \n")
    file.write("                                                                                                                                                            \n")
    file.write("def GET_Enthalpy():                                                                                                                                         \n")
    file.write("    with open('log_Enthelpy.txt') as f:                                                                                                                     \n")                                                                                                                             
    file.write("        next_n_lines = list(islice(f, 5))                                                                                                                   \n")
    file.write("        next_n_lines = list(islice(f,6))                                                                                                                    \n")
    file.write("        np_lines = np.array(next_n_lines)                                                                                                                   \n")
    file.write("        np_lines_lists = np.char.split(np_lines)                                                                                                            \n")                                                                                                                                    
    file.write("        np_lines_array = np.array(np_lines_lists.tolist())                                                                                                  \n")                                                                                                                               
    file.write("        st=float((np_lines_array[1][1]))                                                                                                                    \n")
    file.write("    return st                                                                                                                                               \n")
    file.write("                                                                                                                                                            \n")
    file.write("                                                                                                                                                            \n")
    file.write("A=GET_Enthalpy()                                                                                                                                            \n")
    file.write("print(A)                                                                                                                                                    \n")
    file.close()




A=Enthalpy_Gen0(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)
B=subprocess.call('chmod +x ./Enthalpy0.sh',shell=True)
C=subprocess.call('bash  ./Enthalpy0.sh',shell=True)



A=Enthalpy_Gen(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)
B=subprocess.call('chmod +x ./Enthalpy.sh',shell=True)
C=subprocess.call('bash  ./Enthalpy.sh',shell=True)

    

A=Calc_Enthalpy(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)
B=subprocess.call('chmod +x ./calculation.sh',shell=True)
C=subprocess.call('bash  ./calculation.sh',shell=True)

       

A=Calc_Enthalpy2(epsilon,rmin,r1,rc,bFC,aFC,Friction,NFE)
C=subprocess.call('python3  ./Enthalpy.py',shell=True)
    

C=subprocess.call('python3  ./Plot.py',shell=True)





