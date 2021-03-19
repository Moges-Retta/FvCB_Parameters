# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:03:59 2021

@author: Moges Retta
Estimate parameters of photosynthesis kinectics based on Yin and Struik 2009
Plant, cell and Environment, 2009,32(5) pp.448-64
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class Estimate_FvCB_parameters:
    def __init__(self,gas_exch_measurement):
        self.gas_exch_measurement=gas_exch_measurement
        

    def plot_Rd(self,Rds,points):
            fig, ax = plt.subplots(2,2,constrained_layout=True)
            replicates = Rds['Replicate'].values
            count = 0
            for replicate in replicates:
                Rd = Rds['Rd'].values
                slope = Rds['Slope'].values

                X = points['X'][count]
                Y = points['Y'][count]
                R2 = Rds['R2'].values
                ypredict = slope[count]*X+Rd[count]
                
                if Rd[count]>0:
                    rd_text = ' + ' + str(np.round(Rd[count],2))                    
                else:
                    rd_text = ' - ' + str(np.round(abs(Rd[count]),2))
                text = 'y = ' + str(np.round(slope[count],3)) + '*x '+\
                rd_text+' \n R2 = '+str(np.round(R2[count],3))
                
                if replicate == 1:
                    ax[0][0].plot(X, Y, 'ko')           
                    ax[0][0].plot(X, ypredict, color='black', linewidth=3) 
                    ax[0][0].set_ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)')
                    ax[0][0].text(min(X)+1,max(Y)-1,text)
                    
                elif replicate==2:
                    ax[0][1].plot(X, Y, 'ko')           
                    ax[0][1].plot(X, ypredict, color='black', linewidth=3)
                    ax[0][1].text(min(X)+1,max(Y)-1,text)
                     
                elif replicate==3:
                    ax[1][0].plot(X, Y, 'ko')         
                    ax[1][0].plot(X, ypredict, color='black', linewidth=3)
                    ax[1][0].set_xlabel('\u03A6$_{PSII}$ * $I_{inc}$/4 (µmol $m^{-2}$ $s^{-1}$)') 
                    ax[1][0].set_ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)')
                    ax[1][0].text(min(X)+1,max(Y)-1,text)
                    
                else:
                    ax[1][1].plot(X, Y, 'ko')           
                    ax[1][1].plot(X, ypredict, color='black', linewidth=3) 
                    ax[1][1].set_xlabel('\u03A6$_{PSII}$ * $I_{inc}$/4 (µmol $m^{-2}$ $s^{-1}$)')
                    ax[1][1].text(min(X)+1,max(Y)-1,text)
                count+=1           
            plt.show()
            
# Estimate Rd and calibration factor s
    def estimate_Rd(self): 
        
        """
        Estimate Rd as intercept of linear regression between 
        A and PhiPSII*Iinc/4
        """
        
        AI = self.gas_exch_measurement.get_AI_data()
        AI=AI[AI['Irradiance']<230]
        replicates = AI['Replicate'].values
        replicates=np.unique(replicates)
        df = pd.DataFrame([],columns=['Replicate','Rd','Std.err','R2','Slope']) 
        df2 = pd.DataFrame([],columns=['Replicate','X','Y'] ) 
        
        count = 0
        for replicate in replicates:
            A_I_r= AI[AI['Replicate']==replicate]
            I = A_I_r['Irradiance'].values            
            PhiPS2 = A_I_r['PhiPS2'].values
            A = A_I_r['Net CO2 assimilation rate'].values
            X = PhiPS2*I/4
            slope, intercept, r_value, p_value, std_err  = stats.linregress(X, A) #slope, intercept, r, p, se 
            df.loc[count,'Rd'] = intercept
            df.loc[count,'Std.err'] = std_err
            df.loc[count,'R2'] = r_value**2
            df.loc[count,'Replicate'] = replicate
            df.loc[count,'Slope'] = slope
            
            df2.loc[count,'X'] = X
            df2.loc[count,'Y'] = A
            df2.loc[count,'Replicate'] = replicate
            count+=1
        self.plot_Rd(df,df2)
            
            
            