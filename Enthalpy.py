import os                                                                                                                                                   
import numpy as np                                                                                                                                          
import pandas as pd                                                                                                                                         
from   itertools import islice                                                                                                                              
import math                                                                                                                                                 
                                                                                                                                                            
                                                                                                                                                            
def GET_Enthalpy():                                                                                                                                         
    with open('log_Enthelpy.txt') as f:                                                                                                                     
        next_n_lines = list(islice(f, 5))                                                                                                                   
        next_n_lines = list(islice(f,6))                                                                                                                    
        np_lines = np.array(next_n_lines)                                                                                                                   
        np_lines_lists = np.char.split(np_lines)                                                                                                            
        np_lines_array = np.array(np_lines_lists.tolist())                                                                                                  
        st=float((np_lines_array[1][1]))                                                                                                                    
    return st                                                                                                                                               
                                                                                                                                                            
                                                                                                                                                            
A=GET_Enthalpy()                                                                                                                                            
print(A)                                                                                                                                                    
