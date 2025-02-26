import os
import numpy as np



def LineTension_Unit_Conversion(GROMACS_OUTPUT):
    # GROMACS OUTPUT UNIT: nm^2*bar
    # TARGET UNIT : pN
    # 1 pN=10nm^2*bar
    # 1 nm^2*bar=0.1pN
    
    Target_Unit=GROMACS_OUTPUT*0.1
    return Target_Unit
    




def AreaCompressibility_Unit_Conversion(GROMACS_OUTPUT):
    # GROMACS OUTPUT UNIT: kT/(nm^2)
    # TARGET UNIT : mN/m
    # kT=4.114 	pN⋅nm
    # 1 pN = 1.0E-9 mN
    # kT/(nm^2)=4.114 pN⋅nm/nm
    # kT/(nm^2)=4.114 mN/m 
    Target_Unit=GROMACS_OUTPUT*4.11447
    return Target_Unit
    




def Bending_Rigidity_Unit_Conversion(GROMACS_OUTPUT):
    # GROMACS OUTPUT UNIT: kJ/mol
    # TARGET UNIT : kT
    # kT=2.4773464 kJ/mol
    Target_Unit=GROMACS_OUTPUT/2.4773464
    return Target_Unit
    
















