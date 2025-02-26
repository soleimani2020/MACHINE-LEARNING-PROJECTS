###### ! /usr/bin/ipython3
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
import random
from numpy import linspace , meshgrid , c_
from matplotlib.pyplot import subplots
import scipy.interpolate as interp
import numpy as np
import numpy.fft as FF
import scipy.interpolate as interp
from scipy import ndimage
from scipy.interpolate import griddata
import numpy as np





epsilon= sys.argv[1]
rmin   = sys.argv[2]
r1     = sys.argv[3]
rc     = sys.argv[4]
bFC    = sys.argv[5]
aFC    = sys.argv[6]
Friction    = sys.argv[7]
NFE    = sys.argv[8]


def Average(lst): 
    return np.mean(lst) 


# Simulation box of the frames 
def Sim_BOX(it):
    filename_temp = "conf_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+str("_")+str(it)+".gro"
    filename=filename_temp
    with open(filename) as f:
        lines=f.read().splitlines()
        last_line = np.array((lines[-1]))
        last_line = (np.char.split(last_line))
        np_lines_array = np.array(last_line.tolist())
        LX=float(np_lines_array[0])
        LY=float(np_lines_array[1])
        return LX,LY

    
    
def Sim_BOX_Matrix(initial_TS,final_TS,Interval):
    initial_TS=0
    Interval=1
    Result2=  np.zeros([final_TS,2], dtype=float) 
    for it in range(initial_TS,final_TS,Interval):
        lx=Sim_BOX(it)[0]
        ly=Sim_BOX(it)[1]
        Result2[it][0] = lx
        Result2[it][1] = ly

    return Result2 , np.array(Result2)
    
      

def Snapshot1(it):
    filename_temp = "conf_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+str("_")+str(it)+".gro"
    filename=filename_temp
    with open(filename) as f:
        next_n_lines = list(islice(f,2))
        next_n_lines = list(islice(f,4896))
        np_lines = np.array(next_n_lines)    
        np_lines_lists = np.char.split(np_lines) 
        np_lines_array = np.array(np_lines_lists.tolist())
        raw_data = pd.DataFrame(np_lines_array)
        raw_data = raw_data.astype({0:'str', 1:'str', 2:'str', 3:'float64', 4:'float64',5:'float64', 6:'float64', 7:'float64',8:'float64'})
        #print(raw_data)
        # Select Membrane Trajectory 
        Mydata = (raw_data).values
        x_data = list(Mydata[:,3])
        y_data = list(Mydata[:,4])
        z_data = list(Mydata[:,5])
    return x_data , y_data , z_data



    
def Snapshot2(it):
    filename_temp = "conf_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+str("_")+str(it)+".gro"
    filename=filename_temp
    with open(filename) as f:
        next_n_lines = list(islice(f,10001))
        next_n_lines = list(islice(f,9585))
        np_lines = np.array(next_n_lines)    
        np_lines_lists = np.char.split(np_lines) 
        np_lines_array = np.array(np_lines_lists.tolist())
        raw_data = pd.DataFrame(np_lines_array)
        raw_data = raw_data.astype({0:'str', 1:'str', 2:'float64', 3:'float64', 4:'float64' })
        #print(raw_data)
        # Select Membrane Trajectory 
        Mydata = (raw_data).values
        x_data = list(Mydata[:,2])
        y_data = list(Mydata[:,3])
        z_data = list(Mydata[:,4])
    return x_data , y_data , z_data




def Matrxi_Trajectory(initial_TS,final_TS,Interval):
    Num_Particle=4896
    Result2=  np.zeros([final_TS,Num_Particle,3], dtype=float) 
    for it in range(initial_TS,final_TS,Interval):
        #print(it)
        X_SS1=Snapshot1(it)[0]
        #X_SS2=Snapshot2(it)[0]
        Y_SS1=Snapshot1(it)[1]
        #Y_SS2=Snapshot2(it)[1]
        Z_SS1=Snapshot1(it)[2]
        #Z_SS2=Snapshot2(it)[2]
        #X_SS1.extend(X_SS2)
        #Y_SS1.extend(Y_SS2)
        #Z_SS1.extend(Z_SS2)
        XXX=X_SS1
        YYY=Y_SS1
        ZZZ=Z_SS1
        data = [XXX,YYY,ZZZ]
        df = pd.DataFrame(data)
        df = df.transpose()
        raw_data = df.astype({0:'float64', 1:'float64', 2:'float64'})
        #print(raw_data)
        Mydata = (raw_data).values
        x_data = list(Mydata[:,0])
        y_data = list(Mydata[:,1])
        z_data = list(Mydata[:,2])        
        for i in range(0,Num_Particle):
            Result2[it][i][0] = x_data[i]
            Result2[it][i][1] = y_data[i]
            Result2[it][i][2] = z_data[i]

    return Result2 , np.array(Result2)

 



def planar_modal_analysis (particle_traj_list,box_dim_list,n_modes,initial_TS,final_TS,interp_method="linear"):
    # The particle_traj_list is assumed to be in shape of (n_steps, n_particles, 3)        
    n_particles = particle_traj_list.shape[1]
    qL_max = 2.0 * np.pi * (n_modes - 1) / 2.0
    #print("qL_max = ", qL_max)
    n_q = (n_modes - 1) // 2
    dqL = qL_max / float(n_q)
    sampled_qL = []
    sampled_hpow = []
    sampled_hpow_A = []
    sampled_h=[]
    sampled_abs=[]

    
    
    Result0=  np.zeros([n_modes,n_modes,final_TS], dtype=float) 

    
    for i in range(particle_traj_list.shape[0]):
        ###print(i)
        pos = particle_traj_list[i, :, :]
        #print(pos[0::1, 0]) # list of X for each snapshot
        #print(pos[0::1, 1]) # list of Y for each snapshot
        #print(pos[0::1, 2]) # list of Z for each snapshot
        #print(pos[0::1]) # Trajectory : X Y Z size :(19584, 3) 
        box_L = box_dim_list[i, :]
        l_x, l_y = box_L[0:2]  
        #print(l_x)
        #print(l_y)
        xx = np.linspace(0.0, l_x, n_modes)
        yy = np.linspace(0.0, l_y, n_modes)
        x_2D, y_2D = np.meshgrid(xx, yy)
        
        
        grid_x_2D_flat=x_2D.ravel()
        grid_y_2D_flat=y_2D.ravel()
        xy_space = c_[grid_x_2D_flat,grid_y_2D_flat]
        
        # To visualize the meshgrid 
        #print("Mesh grid in 2D , top view")
        #fig , ax = subplots(figsize=(5,5))
        #ax.scatter(xy_space[:,0],xy_space[:,1])
        
        h_2D = np.zeros_like(x_2D)
        n_layers = 1
        for nn in range(n_layers):
            condx1 = pos[nn::1, 0] < 0.4 * l_x   
            condx2 = pos[nn::1, 0] > 0.6 * l_x   
            condy1 = pos[nn::1, 1] < 0.4 * l_y   
            condy2 = pos[nn::1, 1] > 0.6 * l_y  
            edge = []
            edge.append(pos[nn::1][condx1] + [l_x, 0.0, 0.0]) # if condx1 True then condx1+l_x
            edge.append(pos[nn::1][condx2] - [l_x, 0.0, 0.0]) # if condx2 True then condx2-l_x
            edge.append(pos[nn::1][condy1] + [0.0, l_y, 0.0]) # if condy1 True then condy1+l_y
            edge.append(pos[nn::1][condy2] - [0.0, l_y, 0.0]) # if condy2 True then condy2-l_y
            edge.append(pos[nn::1][condx1 * condy1] + [l_x, l_y, 0.0])   # if condx1&condy1 True then condx1+l_x & condy1+l_y
            edge.append(pos[nn::1][condx1 * condy2] + [l_x, -l_y, 0.0])  # if condx1&condy2 True then condx1+l_x & condy2-l_y
            edge.append(pos[nn::1][condx2 * condy1] + [-l_x, l_y, 0.0])  # if condx2&condy1 True then condx2-l_x & condy1+l_y
            edge.append(pos[nn::1][condx2 * condy2] + [-l_x, -l_y, 0.0]) # if condx2&condy2 True then condx2-l_x & condy2-l_y
            
            #print(edge)
            #print("\n")
            
            x_ = pos[nn::1, 0]
            y_ = pos[nn::1, 1]
            h_ = pos[nn::1, 2]
            
            for _edge in edge:
                x_ = np.append(x_, _edge[:, 0])
                y_ = np.append(y_, _edge[:, 1])
                h_ = np.append(h_, _edge[:, 2])

                
            __h_2D = interp.griddata ((x_, y_), h_, (x_2D, y_2D), method=interp_method)
            #print(__h_2D)
            
            #print("Simulation trajectory , Side view")
            #fig = plt.figure()
            #ax = plt.axes(projection='3d')
            #ax.scatter(x_, y_, h_, c='red',s=0.2)
            #ax.set_xlabel('x')
            #ax.set_ylabel('y')
            #ax.set_zlabel('z')
            

            Mean=np.mean(__h_2D) # Mean of interpolated vals
            
            h_2D += __h_2D - np.mean(__h_2D)  # Subtracting each interpolated val from the Mean
            
            

            

        h_2D /= float(n_layers) # Interpolated vals for ith snapshot
        #print(h_2D)
        
        
        #print("Interpolated trajectory , Side view")
        #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        #surf = ax.plot_surface(x_2D,y_2D, __h_2D, cmap=cm.coolwarm,linewidth=1)
        #fig.colorbar(surf, shrink=0.5)
        #plt.show()
        
        
        ### FFT Section 
        hff = FF.fftshift(FF.fft2(h_2D, [n_modes, n_modes])) / (float(n_modes) ** 2) # Fourier modes for ith snapshot
        #print(hff)
        h_pow = np.absolute(hff) ** 2   ### a+bj : abs:|a|^2+|b|^2
        #print(h_pow)

        

        freq_xL = FF.fftshift(FF.fftfreq(n_modes, 1.0)) * n_modes
        freq_yL = FF.fftshift(FF.fftfreq(n_modes, 1.0)) * n_modes
        
        freq_x_2D, freq_y_2D = np.meshgrid(freq_xL, freq_yL)
        area = l_x * l_y
        qL = np.sqrt((freq_x_2D/l_x) ** 2 + (freq_y_2D/l_y) ** 2) * 2.0 * np.pi # Wave vectors
        #print(qL)
        #print("\n")
        _ind_f = qL / dqL # Normalised wave vector
        #print(_ind_f)

        
        
        
        
        #Given an interval, values outside the interval are clipped to the interval edges. 
        _ind = np.clip((qL ), 0, 100)  
        #print(_ind)
        
        #Find the unique elements of an array.
        index = np.unique(qL)
        #print(index)

        # Calculate the mean of the values of an array at labels.
        frame_qL = ndimage.mean(qL, _ind, index)
        frame_h = ndimage.mean(np.real(hff), _ind, index)
        frame_hpow = ndimage.mean(np.real(h_pow), _ind, index)
        frame_hpow_A = ndimage.mean(np.real(h_pow) / area, _ind, index)
        frame_abs= ndimage.mean(h_pow, _ind, index)

        
        
        sampled_qL.append(frame_qL[1:-1])
        #print(sampled_qL)
        #print(frame_qL[1:-1])  # exclude the first and the last element
        sampled_h.append(frame_h[1:-1])
        sampled_hpow.append(frame_hpow[1:-1])
        sampled_hpow_A.append(frame_hpow_A[1:-1])
        sampled_abs.append(frame_abs[1:-1])

        

           
    result = [(Result0) ,np.transpose(np.array(sampled_qL)), np.transpose(np.array(sampled_h)), np.transpose(np.array(sampled_hpow)), np.transpose(np.array(sampled_hpow_A)), np.transpose(np.array(sampled_abs))]
    
    return result





def Fluctuation_Spectrum_kappa():
    if os.path.isfile("LX_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg"):
        with open("LX_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".xvg") as f0:
            lines=f0.read().splitlines()
            last_line = lines[-1]
            if "252000.000000" in last_line:
                initial_TS =0
                final_TS   =1680
                Interval   =1
                n_modes    =200  # To sidestep some annoying technical subtleties, we will assume that M is even

            
            
                My_Result=planar_modal_analysis(Matrxi_Trajectory(initial_TS,final_TS,Interval)[0],Sim_BOX_Matrix(initial_TS,final_TS,Interval)[0],n_modes,initial_TS,final_TS)

                A=My_Result[1]
                #print(A)
                #print("\n")
                B=My_Result[2]
                #print(B)
                #print("\n")
                C=My_Result[3]
                #print(C)
                #print("\n")
                D=My_Result[4]
                #print(D)
                E=My_Result[5]
                #print(E)

                print("\n")
                print("\n")
                print("\n")


                sampled_qL=My_Result[1]
                ave = np.zeros([1,n_modes])
                for i in range(0,n_modes):
                    ave[0][i]= (np.mean(sampled_qL[i]))
                Myave_q=ave[0]
                #print(Myave_q)


                sampled_h=My_Result[2]
                ave = np.zeros([1,n_modes])
                for i in range(0,n_modes):
                    ave[0][i]= (np.mean(sampled_h[i]))
                Myave_Intensity=ave[0]
                #print(Myave_Intensity)


                sampled_hpow=My_Result[3]
                ave = np.zeros([1,n_modes])
                for i in range(0,n_modes):
                    ave[0][i]= (np.mean(sampled_hpow[i]))
                Myave_hpow=ave[0]
                #print(Myave_hpow)



                sampled_hpow_A=My_Result[4]
                ave = np.zeros([1,n_modes])
                for i in range(0,n_modes):
                    ave[0][i]= (np.mean(sampled_hpow_A[i]))
                Myave_hpow_A=ave[0]
                #print(Myave_hpow_A)


                sampled_abs=My_Result[5]
                ave = np.zeros([1,n_modes])
                for i in range(0,n_modes):
                    ave[0][i]= (np.mean(sampled_abs[i]))
                Myave_abs=ave[0]
                #print(Myave_abs)
                #print(ave[0][0])
                #print(ave[0][-1])
                
                Last_mode=ave[0][-1]
                
                
                if Last_mode < 0.0009 :   ### To filter bad fluctuation spectrums
                    
                    ### Fitting Kappa ###
                        
                    #print("If the system is a tensionless membrane  , we extract kappa by fitting  q^-4  to the data simulation.") 
                    KT= 1 
                    a0=0.66 # Area per lipid - nm^2
                    Num_Lipid_Perleaflet=816
                    Num_Bead=Num_Lipid_Perleaflet  # Per leaflet


                    def model2(q,Kappa):
                        FH = KT/((Num_Bead*a0)*(Kappa*q**4))
                        return FH

                    fit = curve_fit(model2, Myave_q,Myave_abs, maxfev=10000)
                    ans,cov=fit
                    fit_Kappa=ans
                    Simulated_BR=fit_Kappa[0]
                    #print('Kappa:' ,fit_Kappa)

                    fig, ax = plt.subplots(figsize=(6,4))
                    ax.plot(Myave_q,Myave_abs,'bo', label="Kappa:"+str(Simulated_BR)+".")
                    Min_vector = np.min(Myave_q)
                    Max_vector = np.max(Myave_q)
                    t = np.linspace(Min_vector,Max_vector)
                    plt.plot( t, model2(t, fit_Kappa))

                    ax.set_xlabel('q [1/nm]',fontsize=12)
                    ax.set_ylabel('<$|u(q)|^2$>',fontsize=12)
                    plt.legend(loc='upper right')
                    plt.savefig("BR_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".png")
                    #plt.savefig("BR.png")
                    #plt.show()




                    def model2(q,Kappa):
                        FH = KT/((Num_Bead*a0)*(Kappa*q**4))
                        return FH

                    fit = curve_fit(model2, Myave_q,Myave_abs, maxfev=10000)
                    ans,cov=fit
                    fit_Kappa=ans
                    Simulated_BR=fit_Kappa[0]
                    #print('Kappa:' ,fit_Kappa)

                    fig, ax = plt.subplots(figsize=(6,4))
                    ax.plot(Myave_q,Myave_abs,'bo', label="Kappa:"+str(Simulated_BR)+".")
                    Min_vector = np.min(Myave_q)
                    Max_vector = np.max(Myave_q)
                    t = np.linspace(Min_vector,Max_vector)
                    plt.loglog( t, model2(t, fit_Kappa))

                    ax.set_xlabel('q [1/nm]',fontsize=12)
                    ax.set_ylabel('<$|u(q)|^2$>',fontsize=12)
                    plt.legend(loc='upper right')
                    plt.savefig("BR_loglog_"+str(epsilon)+str("_")+str(rmin)+str("_")+str(r1)+str("_")+str(rc)+str("_")+str(bFC)+str("_")+str(aFC)+str("_")+str(Friction)+str("_")+str(NFE)+".png")
                    #plt.savefig("BR_Log.png")
                    #plt.show()
                
                    return Simulated_BR  # KT
            
                else:
                    return 100000
            
            else:
                return 200000
            
    else:
        return 300000
                
            
           
            
            
            
            
A=0#Fluctuation_Spectrum_kappa()      
print(A)
            
            


