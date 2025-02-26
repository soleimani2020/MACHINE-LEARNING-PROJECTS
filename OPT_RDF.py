import numpy as np
import pandas as pd
import os
import numpy as np
from   itertools import islice
import sys
from scipy.stats import wasserstein_distance







epsilon= sys.argv[1]
rmin   = sys.argv[2]
r1     = sys.argv[3]
rc     = sys.argv[4]
bFC    = sys.argv[5]
aFC    = sys.argv[6]
Friction    = sys.argv[7]
NFE    = sys.argv[8]



FILENAME1="Normalised_g_Atomistic.xvg"
First_Sample=0
Last_Sample=1000



def Get_DATA_SubX_ATOMISTIC():
    filename =FILENAME1
    with open(filename) as f:
        next_n_lines = list(islice(f, First_Sample))
        next_n_lines = list(islice(f, Last_Sample))         
        np_lines = np.array(next_n_lines)                
        np_lines_lists = np.char.split(np_lines)        
        np_lines_array = np.array(np_lines_lists.tolist())
        raw_data = pd.DataFrame(np_lines_array)
        raw_data = raw_data.astype({0:'float64',1:'float64' })
        #print(raw_data)
        All=[]
        for i in range(0,len(raw_data)):
            All.append(raw_data[1][i])
        Result=All
    return (Result)



FILENAME2="Normalised_g_T_T_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg"
First_Sample=0
Last_Sample=1000


                     
def Get_DATA_SubX_CG():
    filename =FILENAME2
    with open(filename) as f:
        next_n_lines = list(islice(f, First_Sample))
        next_n_lines = list(islice(f, Last_Sample))         
        np_lines = np.array(next_n_lines)                
        np_lines_lists = np.char.split(np_lines)        
        np_lines_array = np.array(np_lines_lists.tolist())
        raw_data = pd.DataFrame(np_lines_array)
        raw_data = raw_data.astype({0:'float64',1:'float64' })
        #print(raw_data)
        All=[]
        for i in range(0,len(raw_data)):
            All.append(raw_data[1][i])
        Result=All
    return (Result)





def COST_DP():
    if os.path.isfile("g_T_T_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg"):
        Atomistic=Get_DATA_SubX_ATOMISTIC()
        CG=Get_DATA_SubX_CG()
        W2=wasserstein_distance(Atomistic, CG)
        #print(W2)
        if W2<=0.3:
            return 1
        elif   0.3 <W2 <=0.6:
            return 0.8
        elif   0.6 <W2 <=0.9:
            return 0.6
        elif   0.9 <W2 <=1.2:
            return 0.4
        elif   1.2 <W2 <=1.5:
            return 0.2
        elif   1.5 <W2 <=9:
            return 0
        else:   
            return 0
    else:
        return 0
        
    


COST=0#COST_DP()
print(COST)












