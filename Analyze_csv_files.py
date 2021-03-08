# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:04:47 2021

@author: retta001
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FORMAT = ['Photo','Cond','Ci','Fv/Fm', 'PhiPS2','CO2S','PARi']
#PATH = (r'\\WURNET.NL\Homes\retta001\My Documents\Project\2021\GasExchange\')
species_code = ['Hi','Bn']
treatment =['HL','LL']

def make_data(response,Oxygen,species_code,treatment,measurement_days):
    all_data=[]
    for day in measurement_days:
        file_name = 'HL-LL_Day'+str(day)+'_'+species_code+'_'+treatment+'.csv'
        data = pd.read_csv (file_name)
        if response=='AI' and Oxygen==21:
            AI= data[data['Meas']=='LRC_21']
            AI = AI[FORMAT]
            all_data.append(AI)
        elif response=='AI' and Oxygen==2:
            AI= data[data['Meas']=='LRC_2']
            AI = AI[FORMAT]
            all_data.append(AI)
        elif response=='ACI' and Oxygen==2:
            ACI= data[data['Meas']=='CO2_2']
            ACI = ACI[FORMAT]
            all_data.append(ACI)     
        else:            
            ACI= data[data['Meas']=='CO2_21']
            ACI = ACI[FORMAT] 
            all_data.append(ACI)
    return all_data   

    
def plot_response(curve,data,measurement_days):
    fig, ax = plt.subplots()
    plt.rcParams["figure.figsize"] = (10,8)
    plt.rcParams.update({'font.size': 16})
    plt.ylabel("Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)",fontsize=24)
    if curve=='ACI':
            for day in range(0,len(measurement_days)):
                ACI = data[day]
                Ci = ACI["Ci"].values
                A = ACI['Photo'].values
                ax.scatter(np.sort(Ci,axis=None), np.sort(A,axis=None),label='Day'+str(measurement_days[day]))
            plt.xlabel("Intercellular $CO_2$ (µmol $mol^{-1}$)",fontsize=24)  
    else:
        for day in range(0,len(measurement_days)):
            AI = data[day]
            I = AI["PARi"].values
            A = AI['Photo'].values
            ax.scatter(np.sort(I,axis=None), np.sort(A,axis=None),label='Day'+str(measurement_days[day]))
        plt.xlabel("Irradiance (µmol $mol^{-1}$)",fontsize=24)  
    ax.tick_params(labelsize='medium', width=2)
    ax.legend(loc='lower right', fontsize='x-large')     
    
    
# B.Nigra LL
measurement_days = [3,9,10]
A_I_BN_LL = make_data('AI',21,'Bn','LL',measurement_days)
A_CI_BN_LL = make_data('ACI',21,'Bn','LL',measurement_days)
plot_response('AI',A_I_BN_LL,measurement_days)
plot_response('ACI',A_CI_BN_LL,measurement_days)

# H.Icana LL
measurement_days = [4,6,12]
A_I_Hi_LL = make_data('AI',21,'Hi','LL',measurement_days)
A_CI_Hi_LL = make_data('ACI',21,'Hi','LL',measurement_days)
plot_response('AI',A_I_Hi_LL,measurement_days)
plot_response('ACI',A_CI_Hi_LL,measurement_days)

# B.Nigra HL
measurement_days = [8,13]
A_I_BN_HL = make_data('AI',21,'Bn','HL',measurement_days)
A_CI_BN_HL = make_data('ACI',21,'Bn','HL',measurement_days)
plot_response('AI',A_I_BN_HL,measurement_days)
plot_response('ACI',A_CI_BN_HL,measurement_days)

# H.Icana HL
measurement_days = [5,11]
A_I_Hi_HL = make_data('AI',21,'Hi','HL',measurement_days)
A_CI_Hi_HL = make_data('ACI',21,'Hi','HL',measurement_days)
plot_response('AI',A_I_Hi_HL,measurement_days)
plot_response('ACI',A_CI_Hi_HL,measurement_days)