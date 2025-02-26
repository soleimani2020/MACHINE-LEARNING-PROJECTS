#! /usr/bin/ipython3
import os
import numpy as np
import pandas as pd
import math
from   itertools import islice
import sys
import itertools 
import matplotlib.pyplot as plt






def get_abc(alpha,r1,rc):
    A = -((alpha+4)*rc-(alpha+1)*r1)/(rc**(alpha+2)*(rc-r1)**2)
    A = alpha*A
    B = ((alpha+3)*rc-(alpha+1)*r1)/(rc**(alpha+2)*(rc-r1)**3)
    B = alpha*B
    C = (1/rc**alpha) - (A/3)*(rc-r1)**3 - (B/4)*(rc-r1)**4
    return A,B,C

def get_s(r,alpha,r1,rc):
    A,B,C = get_abc(alpha,r1,rc)
    S = A*(r-r1)**2 + B*(r-r1)**3
    return S

def get_switched_force(r,alpha,r1,rc):
    unswitched = alpha/r**(alpha+1)
    switched = unswitched + get_s(r,alpha,r1,rc)
    switched[r<r1] = unswitched[r<r1]
    switched[rc<=r] = 0.0
    switched[r<0.04] = 0.0
    return switched

def get_switched_potential(r,alpha,r1,rc):
    unswitched = 1/r**alpha
    A,B,C = get_abc(alpha,r1,rc)
    result = (1/r**alpha) - (A/3)*(r-r1)**3 - (B/4)*(r-r1)**4 - C
    result[r>rc] = 0.0
    result[r<r1] = unswitched[r<r1] - C
    result[r<0.04] = 0.0
    return result

def v_vdw(r,A,C):
    v_vdw = A/r**6 - C/r**0.5
    return v_vdw

def F_vdw(r,A,C):
    F_vdw = (6*A)/(r**7) - (0.5*C)/(r**1.5)
    return F_vdw

def v_vdw_switch(r,r1,rc,A,C):
    vs6 = get_switched_potential(r,alpha=6,r1=r1,rc=rc)
    vshalf = get_switched_potential(r,alpha=0.5,r1=r1,rc=rc)
    v_vdw_switch = A*vs6 - C*vshalf
    v_vdw_switch=list(v_vdw_switch)
    return A*vs6 ,C*vshalf, v_vdw_switch

def F_vdw_switch(r,r1,rc,A,C):
    Fs6 = get_switched_force(r,alpha=6,r1=r1,rc=rc)
    Fshalf = get_switched_force(r,alpha=0.5,r1=r1,rc=rc)
    F_vdw_switch = A*Fs6 - C*Fshalf
    return  A*Fs6 , C*Fshalf , F_vdw_switch

def Parameter_TT(epsilon,rmin):
    A=  (epsilon/5.5)*0.5*(rmin)**6
    C= (epsilon/5.5)*6*(rmin)**0.5
    return  A ,C

def v_vdw_HH(r,A):
    v_vdw = (0.4*A)/(r**6)
    return v_vdw

def F_vdw_HH(r,A):
    F_vdw = (0.4*6*A)/(r**7)
    return F_vdw

def v_vdw_HH_switch(r,r1,rc,A):
    vs6 = get_switched_potential(r,alpha=6,r1=r1,rc=rc)
    v_vdw_switch = 0.4*A*vs6 
    v_vdw_switch=list(v_vdw_switch)
    return v_vdw_switch

def F_vdw_HH_switch(r,r1,rc,A):
    Fs6 = get_switched_force(r,alpha=6,r1=r1,rc=rc)
    F_vdw_switch = 0.4*A*Fs6 
    return F_vdw_switch

def Parameter_HH(epsilon,rmin):
    A=  (epsilon/5.5)*0.5*(rmin)**6
    return  A 

def Tables(epsilon,rmin,Max_distance,r1,rc,bFC,aFC,Friction,NFE):
    A =  Parameter_TT(epsilon,rmin)[0]
    C =  Parameter_TT(epsilon,rmin)[1]
    r = np.arange(0.0,Max_distance,0.002)
    analytic_noswitch_potential = v_vdw(r=r,A=A,C=C)
    My_G=-1*v_vdw_switch(r=r,A=A,C=C,r1=r1,rc=rc)[1]                         # <0
    My_H=v_vdw_switch(r=r,A=A,C=C,r1=r1,rc=rc)[0]                            # >0
    My_minus_G_Prime=-1*F_vdw_switch(r=r,A=A,C=C,r1=r1,rc=rc)[1]             # <0
    My_Minus_H_Prime=F_vdw_switch(r=r,A=A,C=C,r1=r1,rc=rc)[0]                # >0
    analytic_switch_potential = v_vdw_switch(r=r,A=A,C=C,r1=r1,rc=rc)[2]
    analytic_switch_force_H =F_vdw_switch(r=r,A=A,C=C,r1=r1,rc=rc)[0]
    analytic_switch_force_G =F_vdw_switch(r=r,A=A,C=C,r1=r1,rc=rc)[1]
    analytic_switch_force =F_vdw_switch(r=r,A=A,C=C,r1=r1,rc=rc)[2]
    analytic_noswitch_force = F_vdw(r=r,A=A,C=C)
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(r,analytic_noswitch_potential,'k.-',label='Normal potential')
    plt.plot(r,analytic_switch_potential,'g--',linewidth=3,label='My shifted potential')
    leg = ax.legend(fancybox=True)
    ax.set_ylim(-4,.1)
    ax.set_ylim(-8,4)
    ax.set_xlim(0,3)
    plt.axvline(r1 , color='black',linestyle='--',linewidth=2)
    plt.axvline(rc , color='black',linestyle='--',linewidth=2)
    plt.axvline(rmin , color='red',linestyle='--',linewidth=2)
    plt.axhline(0 , color='black',linestyle='--',linewidth=2)
    ax.set_ylabel('$V_{LJ}$')
    ax.set_xlabel('r (nm)')
    plt.savefig("Plot_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".png")
    plt.grid(True)
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(r,analytic_switch_potential,'g--',linewidth=3,label='My shifted potential')
    plt.axvline(r1 , color='black',linestyle='--',linewidth=2)
    plt.axvline(rc , color='black',linestyle='--',linewidth=2)
    leg = ax.legend(fancybox=True)
    leg.get_frame().set_alpha(0.5)
    ax.set_ylim(-4,.1)
    ax.set_ylim(-0.25,0.25)
    ax.set_xlim(r1-0.2,rc+0.2)
    ax.set_ylabel('$V_{LJ}$')
    ax.set_xlabel('r (nm)')
    plt.grid(True)
    plt.savefig("Plot_Zoom_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".png")
    # For electrostatic interactions 
    Distance = np.arange(0.0,Max_distance,0.002)
    F=[]
    for r in Distance:
        if r<0.04:
            f=0
            F.append(f)
        else:
            f=(1/r)
            F.append(f)
    My_F=F
    Minus_F_Prime=[]
    for r in Distance:
        if r<0.04:
            f_prime=0
            Minus_F_Prime.append(f_prime)
        else:
            f_prime=-(1/r)**2
            Minus_F_Prime.append(-f_prime)
    My_Minus_F_Prime=Minus_F_Prime
    # dictionary of lists 
    dict = {'r': Distance, 'F': My_F, 'MFP': My_Minus_F_Prime , 'G': My_G, 'MGP': My_minus_G_Prime, 'H': My_H  ,'MHP': My_Minus_H_Prime} 
    df = pd.DataFrame(dict)
    #print(df)
    Molecule_IDit = pd.DataFrame(df, columns= ['r'])
    Number_DATA_Memit=len(Molecule_IDit)
    Membrane_Datait= pd.DataFrame(df, columns= ['r','F','MFP','G','MGP','H','MHP','Switch_V'])
    Mem_Datait = Membrane_Datait.to_numpy()
    Mem_Datait = pd.DataFrame(Mem_Datait)
    Mem_Datait = Mem_Datait.astype({0:'float', 1:'float', 2:'float', 3:'float', 4:'float', 5:'float' ,6:'float'})
    Membrane_Data_it =  np.zeros ([Number_DATA_Memit,7], dtype=object)
    for i in range(0,7):
        for j in range(0,Number_DATA_Memit):
            Membrane_Data_it[j][i]=Mem_Datait[i][j]
    My_Membrane_Data_it=Membrane_Data_it
    #np.savetxt('table_T_T_'+str(epsilon_value)+'_'+str(rmin_value)+'.xvg', My_Membrane_Data_it, delimiter=" ", fmt="%2f %8f %13f %12f %12f %9f %8f ")    
    np.savetxt('table_T'+str("-")+str(epsilon)+str("-")+str(rmin)+str("-")+str(r1)+str("-")+str(rc)+str("-")+str(bFC)+str("-")+str(aFC)+str("-")+str(Friction)+str("-")+str(NFE)+str("_")+'T'+str("-")+str(epsilon)+str("-")+str(rmin)+str("-")+str(r1)+str("-")+str(rc)+str("-")+str(bFC)+str("-")+str(aFC)+str("-")+str(Friction)+str("-")+str(NFE)+'.xvg', My_Membrane_Data_it, delimiter=" ", fmt="%2f %8f %13f %12f %12f %9f %8f ")
    # End of table T-T
    Distance=np.arange(0.0,Max_distance,0.002)
    F=[]
    for r in Distance:
        if r<0.04:
            f=0
            F.append(f)
        else:
            f=(1/r)
            F.append(f)
    My_F=F
    Minus_F_Prime=[]
    for r in Distance:
        if r<0.04:
            f_prime=0
            Minus_F_Prime.append(f_prime)
        else:
            f_prime=-(1/r)**2
            Minus_F_Prime.append(-f_prime)
    My_Minus_F_Prime=Minus_F_Prime
    G=[]
    for r in Distance:
        if r<0.04:
            g=0
            G.append(g)
        else:
            g=0
            G.append(g)
    My_G=G
    Minus_G_Prime=[]
    for r in Distance:
        if r<0.04:
            g_prime=0
            Minus_G_Prime.append(g_prime)
        else:
            g_prime=0
            Minus_G_Prime.append(-g_prime)

    My_minus_G_Prime=Minus_G_Prime
    A=Parameter_HH(epsilon,rmin)    
    My_H=v_vdw_HH_switch(r=Distance,A=A,r1=r1,rc=rc)
    My_Minus_H_Prime=F_vdw_HH_switch(r=Distance,A=A,r1=r1,rc=rc)
    dict = {'r': Distance, 'F': My_F, 'MFP': My_Minus_F_Prime , 'G': My_G, 'MGP': My_minus_G_Prime, 'H': My_H  ,'MHP': My_Minus_H_Prime} 
    df = pd.DataFrame(dict)
    Molecule_IDit = pd.DataFrame(df, columns= ['r'])
    Number_DATA_Memit=len(Molecule_IDit)
    Membrane_Datait= pd.DataFrame(df, columns= ['r','F','MFP','G','MGP','H','MHP'])
    Mem_Datait = Membrane_Datait.to_numpy()
    Mem_Datait = pd.DataFrame(Mem_Datait)
    Mem_Datait = Mem_Datait.astype({0:'float', 1:'float', 2:'float', 3:'float', 4:'float', 5:'float' ,6:'float' })
    Membrane_Data_it =  np.zeros ([Number_DATA_Memit,7], dtype=object)
    for i in range(0,7):
        for j in range(0,Number_DATA_Memit):
            Membrane_Data_it[j][i]=Mem_Datait[i][j]
    My_Membrane_Data_it=Membrane_Data_it
    
    np.savetxt('table'+str("-")+str(epsilon)+str("-")+str(rmin)+str("-")+str(r1)+str("-")+str(rc)+str("-")+str(bFC)+str("-")+str(aFC)+str("-")+str(Friction)+str("-")+str(NFE)+'.xvg', My_Membrane_Data_it, delimiter=" ", fmt="%2f %8f %13f %12f %12f %9f %8f ") 
    
    
    
    np.savetxt('table_H'+str("-")+str(epsilon)+str("-")+str(rmin)+str("-")+str(r1)+str("-")+str(rc)+str("-")+str(bFC)+str("-")+str(aFC)+str("-")+str(Friction)+str("-")+str(NFE)+str("_")+'H'+str("-")+str(epsilon)+str("-")+str(rmin)+str("-")+str(r1)+str("-")+str(rc)+str("-")+str(bFC)+str("-")+str(aFC)+str("-")+str(Friction)+str("-")+str(NFE)+'.xvg', My_Membrane_Data_it, delimiter=" ", fmt="%2f %8f %13f %12f %12f %9f %8f ")
    






def Tables_p(epsilon,rmin,Max_distance,r1,rc,bFC,aFC,Friction,NFE):
    Distance = np.arange(0.0,Max_distance,0.002)
    F=[]
    for r in Distance:
        if r<0.04:
            f=0
            F.append(f)
        else:
            f=(1/r)
            F.append(f)
    My_F=F
    Minus_F_Prime=[]
    for r in Distance:
        if r<0.04:
            f_prime=0
            Minus_F_Prime.append(f_prime)
        else:
            f_prime=-(1/r)**2
            Minus_F_Prime.append(-f_prime)
    My_Minus_F_Prime=Minus_F_Prime
    G=[]
    for r in Distance:
        if r<0.04:
            g=0
            G.append(g)
        else:
            g=0
            G.append(g)
    My_G=G
    Minus_G_Prime=[]
    for r in Distance:
        if r<0.04:
            g_prime=0
            Minus_G_Prime.append(g_prime)
        else:
            g_prime=0
            Minus_G_Prime.append(g_prime)
            
    My_minus_G_Prime=Minus_G_Prime
    H=[]
    for r in Distance:
        if r<0.04:
            h=0
            H.append(h)
        else:
            h=0
            H.append(h)
    My_H=H
    Minus_H_Prime=[]
    for r in Distance:
        if r<0.04:
            h_prime=0
            Minus_H_Prime.append(h_prime)
        else:
            h_prime=0
            Minus_H_Prime.append(h_prime)
    My_Minus_H_Prime=Minus_H_Prime
    
    dict = {'r': Distance, 'F': My_F, 'MFP': My_Minus_F_Prime , 'G': My_G, 'MGP': My_minus_G_Prime, 'H': My_H  ,'MHP': My_Minus_H_Prime} 
    df = pd.DataFrame(dict)
    Molecule_IDit = pd.DataFrame(df, columns= ['r'])
    Number_DATA_Memit=len(Molecule_IDit)
    Membrane_Datait= pd.DataFrame(df, columns= ['r','F','MFP','G','MGP','H','MHP'])
    Mem_Datait = Membrane_Datait.to_numpy()
    Mem_Datait = pd.DataFrame(Mem_Datait)
    Mem_Datait = Mem_Datait.astype({0:'float', 1:'float', 2:'float', 3:'float', 4:'float', 5:'float' ,6:'float' })
    Membrane_Data_it =  np.zeros ([Number_DATA_Memit,7], dtype=object)
    for i in range(0,7):
        for j in range(0,Number_DATA_Memit):
            Membrane_Data_it[j][i]=Mem_Datait[i][j]
    My_Membrane_Data_it=Membrane_Data_it
    np.savetxt('Ptable'+str("-")+str(epsilon)+str("-")+str(rmin)+str("-")+str(r1)+str("-")+str(rc)+str("-")+str(bFC)+str("-")+str(aFC)+str("-")+str(Friction)+str("-")+str(NFE)+'.xvg', My_Membrane_Data_it, delimiter=" ", fmt="%2f %8f %13f %12f %12f %9f %8f ") 
    





























    



