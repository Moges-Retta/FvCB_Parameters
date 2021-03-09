# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:04:47 2021

@author: retta001
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FORMAT = ['Photo','Cond','Ci','Fv/Fm', 'PhiPS2','CO2S','PARi']
PATH = (r'\\WURNET.NL\Homes\retta001\My Documents\Project\2021\GasExchange\\')
species_code = ['Hi','Bn']
treatment =['HL','LL']

def make_data(response,Oxygen,species_code,treatment,measurement_days):
    all_data=[]
    for day in measurement_days:
        file_name = 'HL-LL_Day'+str(day)+'_'+species_code+'_'+treatment+'.csv'
        data = pd.read_csv (file_name)
        if response=='Light' and Oxygen==21:
            AI= data[data['Meas']=='LRC_21']
            AI = AI[FORMAT]
            all_data.append(AI)
        elif response=='Light' and Oxygen==2:
            AI= data[data['Meas']=='LRC_2']
            AI = AI[FORMAT]
            all_data.append(AI)
        elif response=='CO2' and Oxygen==2:
            ACI= data[data['Meas']=='CO2_2']
            ACI = ACI[FORMAT]
            all_data.append(ACI)     
        else:            
            ACI= data[data['Meas']=='CO2_21']
            ACI = ACI[FORMAT] 
            all_data.append(ACI)
    return all_data   

    
def plot_response(treatment,data,measurement_days):

    plt.rcParams["figure.figsize"] = (15,12)
    plt.rcParams.update({'font.size': 16})
    symbols = ['ko','k^','ks']
    if treatment=='CO2':
            fig, ax = plt.subplots(2,2)
            for i in range(0,len(measurement_days)):
                symbol=symbols[i]
                ACI = data[i]
                Ci = ACI["Ci"].values
                A = ACI['Photo'].values
                gs = ACI['Cond'].values
                PhiPS2 = ACI['PhiPS2'].values.astype(float)
                FvFm = ACI['Fv/Fm'].values
                ax[0][0].plot(np.sort(Ci,axis=None), np.sort(A,axis=None),symbol,fillstyle='none',markersize=8)
                ax[0][1].plot(np.sort(Ci,axis=None), np.sort(gs,axis=None),symbol,fillstyle='none',markersize=8)
                ax[1][0].plot(np.sort(Ci,axis=None), np.sort(PhiPS2,axis=None),symbol,fillstyle='none',markersize=8)
                ax[1][1].plot(np.sort(Ci,axis=None), np.sort(FvFm,axis=None),'ko',label='Day'+str(measurement_days[i]),fillstyle='none',markersize=8)
                ax[0][0].set_ylabel("Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)")
                ax[0][1].set_ylabel("Stomatal conductance (mol $m^{-2}$ $s^{-1}$)")
                ax[1][0].set_ylabel("\u03A6$_{PSII}$ (-)")
                ax[1][1].set_ylabel("Fv/Fm (-)")
                ax[1][0].set_xlabel("Intercellular $CO_2$ (µmol $mol^{-1}$)")
                ax[1][1].set_xlabel("Intercellular $CO_2$ (µmol $mol^{-1}$)")
                ax[1][1].set_ylim(bottom=0.7)
    else:
        fig, ax = plt.subplots(2,2)
        for i in range(0,len(measurement_days)):
            symbol=symbols[i]
            AI = data[i]
            I = AI["PARi"].values
            A = AI['Photo'].values
            gs = AI['Cond'].values
            PhiPS2 = AI['PhiPS2'].values.astype(float)
            FvFm = AI['Fv/Fm'].values
            ax[0][0].plot(np.sort(I,axis=None), np.sort(A,axis=None),symbol,fillstyle='none',markersize=8)
            ax[0][1].plot(np.sort(I,axis=None), np.sort(gs,axis=None),symbol,fillstyle='none',markersize=8)
            ax[1][0].plot(np.sort(I,axis=None), np.sort(PhiPS2,axis=None),symbol,fillstyle='none',markersize=8)
            ax[1][1].plot(np.sort(I,axis=None), np.sort(FvFm,axis=None),symbol,label='Day'+str(measurement_days[i]),fillstyle='none',markersize=8)
            ax[0][0].set_ylabel("Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)")
            ax[0][1].set_ylabel("Stomatal conductance (mol $m^{-2}$ $s^{-1}$)")
            ax[1][0].set_ylabel("\u03A6$_{PSII}$ (-)")
            ax[1][1].set_ylabel("Fv/Fm (-)")
            ax[1][0].set_xlabel("Irradiance (µmol $m^{-2}$ $s^{-1}$)")
            ax[1][1].set_xlabel("Irradiance (µmol $m^{-2}$ $s^{-1}$)")
            ax[1][1].set_ylim(bottom=0.7)
#    ax.tick_params(labelsize='medium', width=5)
            ax[1][1].legend(loc='lower right', fontsize='x-large')     
    
    

#FORMAT = ['Photo','Cond','Ci','Fv/Fm', 'PhiPS2','CO2S','PARi']

def replicates_to_Excel(data_frame,species,Oxygen,curve,treatment):
    columns = ['Replicate','Species','Treatment','Measurement type','Oxygen level','Net CO2 assimilation rate','Intercellular CO2 concentration','PhiPS2','Irradiance','Stomatal conductance for CO2']
    Gas_Exchange_data = pd.DataFrame([],columns=columns )
    for i in range(0,len(data_frame)):
        AI = data_frame[i]
        I = AI['PARi'].values
        A = AI['Photo'].values
        gs = AI['Cond'].values
        Ci = AI['Ci'].values
        PhiPS2 = AI['PhiPS2'].values
        df1 = pd.DataFrame([],columns=columns )
        df1['Replicate'] = [i+1]*len(data_frame[i])
        df1['Species'] = species
        df1['Treatment'] = treatment
        df1['Measurement type'] = curve
        df1['Oxygen level'] = [Oxygen]*len(data_frame[i])
        df1['Net CO2 assimilation rate'] = A[:]
        df1['Intercellular CO2 concentration'] = Ci[:]
        df1['PhiPS2'] = PhiPS2[:]
        df1['Irradiance'] = I[:]
        df1['Stomatal conductance for CO2'] = gs[:]
        Gas_Exchange_data=Gas_Exchange_data.append(df1)
    return Gas_Exchange_data
  
columns = ['Replicate','Species','Treatment','Measurement type','Oxygen level','Net CO2 assimilation rate','Intercellular CO2 concentration','PhiPS2','Irradiance','Stomatal conductance for CO2']
Gas_Exchange_data = pd.DataFrame([],columns=columns )   

# B.Nigra LL
measurement_days = [3,9,10]
A_I_BN_LL = make_data('Light',21,'Bn','LL',measurement_days)
A_CI_BN_LL = make_data('CO2',21,'Bn','LL',measurement_days)
plot_response('Light',A_I_BN_LL,measurement_days)
plot_response('CO2',A_CI_BN_LL,measurement_days)
df_I=replicates_to_Excel(A_I_BN_LL,'B.Nigra',0.21,'A-I curve','LL')
df_CI=replicates_to_Excel(A_CI_BN_LL,'B.Nigra',0.21,'A-CI curve','LL')
Gas_Exchange_data=Gas_Exchange_data.append(df_I)
Gas_Exchange_data=Gas_Exchange_data.append(df_CI)

A_I_BN_LL = make_data('Light',2,'Bn','LL',measurement_days)
A_CI_BN_LL = make_data('CO2',2,'Bn','LL',measurement_days)
plot_response('Light',A_I_BN_LL,measurement_days)
plot_response('CO2',A_CI_BN_LL,measurement_days)
df_I=replicates_to_Excel(A_I_BN_LL,'B.Nigra',0.02,'A-I curve','LL')
df_CI=replicates_to_Excel(A_CI_BN_LL,'B.Nigra',0.02,'A-CI curve','LL')
Gas_Exchange_data=Gas_Exchange_data.append(df_I)
Gas_Exchange_data=Gas_Exchange_data.append(df_CI)
  

# H.Incana LL
measurement_days = [4,6,12]
A_I_Hi_LL = make_data('AI',21,'Hi','LL',measurement_days)
A_CI_Hi_LL = make_data('ACI',21,'Hi','LL',measurement_days)
plot_response('Light',A_I_Hi_LL,measurement_days)
plot_response('CO2',A_CI_Hi_LL,measurement_days)
df_I=replicates_to_Excel(A_I_Hi_LL,'H.Incana',0.21,'A-I curve','LL')
df_CI=replicates_to_Excel(A_CI_Hi_LL,'H.Incana',0.21,'A-CI curve','LL')
Gas_Exchange_data=Gas_Exchange_data.append(df_I)
Gas_Exchange_data=Gas_Exchange_data.append(df_CI)

A_I_Hi_LL = make_data('Light',2,'Hi','LL',measurement_days)
A_CI_Hi_LL = make_data('CO2',2,'Hi','LL',measurement_days)
plot_response('Light',A_I_Hi_LL,measurement_days)
plot_response('CO2',A_CI_Hi_LL,measurement_days)
df_I=replicates_to_Excel(A_I_Hi_LL,'H.Incana',0.02,'A-I curve','LL')
df_CI=replicates_to_Excel(A_CI_Hi_LL,'H.Incana',0.02,'A-CI curve','LL')
Gas_Exchange_data=Gas_Exchange_data.append(df_I)
Gas_Exchange_data=Gas_Exchange_data.append(df_CI)


# B.Nigra HL
measurement_days = [8,13]
A_I_BN_HL = make_data('AI',21,'Bn','HL',measurement_days)
A_CI_BN_HL = make_data('ACI',21,'Bn','HL',measurement_days)
plot_response('Light',A_I_BN_HL,measurement_days)
plot_response('CO2',A_CI_BN_HL,measurement_days)
df_I=replicates_to_Excel(A_I_BN_HL,'B.Nigra',0.21,'A-I curve','HL')
df_CI=replicates_to_Excel(A_CI_BN_HL,'B.Nigra',0.21,'A-CI curve','HL')
Gas_Exchange_data=Gas_Exchange_data.append(df_I)
Gas_Exchange_data=Gas_Exchange_data.append(df_CI)

A_I_BN_HL = make_data('Light',2,'Bn','HL',measurement_days)
A_CI_BN_HL = make_data('CO2',2,'Bn','HL',measurement_days)
plot_response('Light',A_I_BN_HL,measurement_days)
plot_response('CO2',A_CI_BN_HL,measurement_days)
df_I=replicates_to_Excel(A_I_BN_HL,'B.Nigra',0.02,'A-I curve','HL')
df_CI=replicates_to_Excel(A_CI_BN_HL,'B.Nigra',0.02,'A-CI curve','HL')
Gas_Exchange_data=Gas_Exchange_data.append(df_I)
Gas_Exchange_data=Gas_Exchange_data.append(df_CI)
#
# H.Incana HL
measurement_days = [5,11]
A_I_Hi_HL = make_data('AI',21,'Hi','HL',measurement_days)
A_CI_Hi_HL = make_data('ACI',21,'Hi','HL',measurement_days)
plot_response('Light',A_I_Hi_HL,measurement_days)
plot_response('CO2',A_CI_Hi_HL,measurement_days)
df_I=replicates_to_Excel(A_I_Hi_HL,'H.Incana',0.21,'A-I curve','HL')
df_CI=replicates_to_Excel(A_CI_Hi_HL,'H.Incana',0.21,'A-CI curve','HL')
Gas_Exchange_data=Gas_Exchange_data.append(df_I)
Gas_Exchange_data=Gas_Exchange_data.append(df_CI)

A_I_Hi_HL = make_data('Light',2,'Hi','HL',measurement_days)
A_CI_Hi_HL = make_data('CO2',2,'Hi','HL',measurement_days)
plot_response('Light',A_I_Hi_HL,measurement_days)
plot_response('CO2',A_CI_Hi_HL,measurement_days)
df_I=replicates_to_Excel(A_I_Hi_HL,'H.Incana',0.02,'A-I curve','HL')
df_CI=replicates_to_Excel(A_CI_Hi_HL,'H.Incana',0.02,'A-CI curve','HL')
Gas_Exchange_data=Gas_Exchange_data.append(df_I)
Gas_Exchange_data=Gas_Exchange_data.append(df_CI)

#Gas_Exchange_data.to_excel(PATH + 'Gas_Exchange_data.xlsx', index = False)
