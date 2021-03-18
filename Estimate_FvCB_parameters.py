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
        

    def plot_predictions(self,slope,intercept,X,y):
                
            ypredict = slope*X+intercept
            plt.scatter(X, y,  color='black')
            plt.plot(X, ypredict, color='blue', linewidth=3)            
            plt.xticks(())
            plt.yticks(())  
            plt.xlabel('\u03A6$_{PSII}$ * $I_{inc}$/4 (µmol $m^{-2}$ $s^{-1}$)')
            plt.ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)')
            plt.show()
            
# Estimate Rd and calibration factor s
    def estimate_Rd(self): 
        
        """
        Estimate Rd as intercept of linear regression between A and PhiPSII*Iinc/4
        """
        
        AI = self.gas_exch_measurement.get_AI_data()
        AI=AI[AI['Irradiance']<230]
        replicates = AI['Replicate'].values
        replicates=np.unique(replicates)
        df = pd.DataFrame([],columns=['Rd','Std.err','R2'] ) 
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
            self.plot_predictions(slope,intercept,X,A)
            count+=1
            
            
            