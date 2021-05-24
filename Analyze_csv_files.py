# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:04:47 2021

@author: retta001
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

FORMAT = ['Photo','Cond','Ci',"CO2R",'CO2S','PhiPS2','CO2S','PARi','Trmmol','BLCond','VpdL']
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
    plt.rcParams.update({'font.size': 14})
    symbols = ['ko','k^','ks','k<','k>']
    if treatment=='CO2':
            fig, ax = plt.subplots(2,2,constrained_layout=True)
            for i in range(0,len(measurement_days)):
                symbol=symbols[i]
                ACI = data[i]
                Ci = ACI["Ci"].values
                A = ACI['Photo'].values
                gs = ACI['Cond'].values
                PhiPS2 = ACI['PhiPS2'].values.astype(float)
                CO2R = ACI["CO2R"].values.astype(float)
#                CO2S = ACI["CO2S"].values.astype(float)                
                ax[0][0].plot(Ci, A,symbol,fillstyle='none',markersize=8)
                ax[0][1].plot(Ci, gs,symbol,fillstyle='none',markersize=8)
                ax[1][0].plot(Ci, PhiPS2,symbol,fillstyle='none',markersize=8)
                ax[1][1].plot(CO2R,Ci, symbol,label='Day'+str(measurement_days[i]),fillstyle='none',markersize=8)
                ax[0][0].set_ylabel("Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)")
                ax[0][0].set_xlabel("Intercellular $CO_2$ (µmol $mol^{-1}$)")
                
                ax[0][1].set_ylabel("Stomatal conductance (mol $m^{-2}$ $s^{-1}$)")
                ax[0][1].set_xlabel("Intercellular $CO_2$ (µmol $mol^{-1}$)")
                
                ax[1][0].set_ylabel("\u03A6$_{PSII}$ (-)")
               
                ax[1][1].set_ylabel("Intercellular $CO_2$ (µmol $mol^{-1}$)")
                ax[1][0].set_xlabel("Intercellular $CO_2$ (µmol $mol^{-1}$)")
                ax[1][1].set_xlabel("External $CO_2$ (µmol $mol^{-1}$)")
#                ax[1][1].set_ylim(bottom=0.7)
                ax[1][1].legend(loc='best', fontsize='x-large')     
                
    else:
        fig, ax = plt.subplots(2,2,constrained_layout=True)
        for i in range(0,len(measurement_days)):
            symbol=symbols[i]
            AI = data[i]
            I = AI["PARi"].values
            A = AI['Photo'].values
            gs = AI['Cond'].values
            PhiPS2 = AI['PhiPS2'].values.astype(float)
            Ci = AI["Ci"].values
            ax[0][0].plot(I, A,symbol,fillstyle='none',markersize=8)
            ax[0][1].plot(I, gs,symbol,fillstyle='none',markersize=8)
            ax[1][0].plot(I, PhiPS2,symbol,fillstyle='none',markersize=8)
            ax[1][1].plot(I, Ci,symbol,label='Day'+str(measurement_days[i]),fillstyle='none',markersize=8)
            ax[0][0].set_ylabel("Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)")
            ax[0][1].set_ylabel("Stomatal conductance (mol $m^{-2}$ $s^{-1}$)")
            ax[1][0].set_ylabel("\u03A6$_{PSII}$ (-)")
            ax[1][1].set_ylabel("Intercellular $CO_2$ (µmol $mol^{-1}$)")
            ax[1][0].set_xlabel("Irradiance (µmol $m^{-2}$ $s^{-1}$)")
            ax[1][1].set_xlabel("Irradiance (µmol $m^{-2}$ $s^{-1}$)")
#            ax[1][1].set_ylim(bottom=0.7)
#    ax.tick_params(labelsize='medium', width=5)
            ax[1][1].legend(loc='best', fontsize='x-large')     
    

#FORMAT = ['Photo','Cond','Ci','Fv/Fm', 'PhiPS2','CO2S','PARi']

def replicates_to_Excel(data_frame,species,Oxygen,curve,treatment):
    columns = ['Replicate','Species','Treatment','Measurement type','Oxygen level','Net CO2 assimilation rate','Intercellular CO2 concentration','PhiPS2','Irradiance','Stomatal conductance for CO2','CO2S','CO2R','Trmmol','BLCond','VpdL']
    Gas_Exchange_data = pd.DataFrame([],columns=columns )
    for i in range(0,len(data_frame)):
        AI = data_frame[i]
        I = AI['PARi'].values
        A = AI['Photo'].values
        gs = AI['Cond'].values
        Ci = AI['Ci'].values
        CO2S = AI['CO2S'].values
        CO2R = AI['CO2R'].values
        
        Trmmol = AI['Trmmol'].values
        BLCond = AI['BLCond'].values
        vpdL = AI['VpdL'].values
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
        df1['CO2S'] = CO2S[:]
        df1['CO2R'] = CO2R[:]        
        df1['Trmmol'] = Trmmol[:]
        df1['BLCond'] = BLCond[:]
        df1['VpdL'] = vpdL[:]
        
        Gas_Exchange_data=Gas_Exchange_data.append(df1)
    return Gas_Exchange_data
  
columns = ['Replicate','Species','Treatment','Measurement type','Oxygen level','Net CO2 assimilation rate','Intercellular CO2 concentration','PhiPS2','Irradiance','Stomatal conductance for CO2','CO2S','CO2R','Trmmol','BLCond','VpdL']
Gas_Exchange_data = pd.DataFrame([],columns=columns )   

# B.Nigra LL
measurement_days = [10,14,19,23]
A_I_BN_LL = make_data('Light',21,'Bn','LL',measurement_days)
A_CI_BN_LL = make_data('CO2',21,'Bn','LL',measurement_days)

#clean data
data = A_I_BN_LL[2]
data=data.drop([5]) # duplicate data for Pari = 200
A_I_BN_LL[2]=[]
A_I_BN_LL[2]=data

#clean data
data = A_CI_BN_LL[0]
data=data.drop([21]) # duplicate data for Pari = 200
A_CI_BN_LL[0]=[]
A_CI_BN_LL[0]=data

data = A_CI_BN_LL[1]
data=data.drop([21]) # duplicate data for Pari = 200
A_CI_BN_LL[1]=[]
A_CI_BN_LL[1]=data

data = A_CI_BN_LL[2]
data=data.drop([22]) # duplicate data for Pari = 200
A_CI_BN_LL[2]=[]
A_CI_BN_LL[2]=data

data = A_CI_BN_LL[3]
data=data.drop([21]) # duplicate data for Pari = 200
A_CI_BN_LL[3]=[]
A_CI_BN_LL[3]=data

#plot_response('Light',A_I_BN_LL,measurement_days)
#plot_response('CO2',A_CI_BN_LL,measurement_days)
df_I=replicates_to_Excel(A_I_BN_LL,'B.Nigra',0.21,'A-I curve','LL')
df_CI=replicates_to_Excel(A_CI_BN_LL,'B.Nigra',0.21,'A-CI curve','LL')
Gas_Exchange_data=Gas_Exchange_data.append(df_I)
Gas_Exchange_data=Gas_Exchange_data.append(df_CI)
#
A_I_BN_LL = make_data('Light',2,'Bn','LL',measurement_days)
A_CI_BN_LL = make_data('CO2',2,'Bn','LL',measurement_days)

#plot_response('Light',A_I_BN_LL,measurement_days)
#plot_response('CO2',A_CI_BN_LL,measurement_days)
df_I=replicates_to_Excel(A_I_BN_LL,'B.Nigra',0.02,'A-I curve','LL')
df_CI=replicates_to_Excel(A_CI_BN_LL,'B.Nigra',0.02,'A-CI curve','LL')
Gas_Exchange_data=Gas_Exchange_data.append(df_I)
Gas_Exchange_data=Gas_Exchange_data.append(df_CI)
#  
#
# H.Incana LL
measurement_days = [6,12,17,21]
A_I_Hi_LL = make_data('Light',21,'Hi','LL',measurement_days)
A_CI_Hi_LL = make_data('CO2',21,'Hi','LL',measurement_days)

#clean data
data = A_I_Hi_LL[0]
data=data.drop([4,6,8,10,12,14,16,18,20,22,24]) # duplicate data
A_I_Hi_LL[0]=[]
A_I_Hi_LL[0]=data

#clean data
data = A_CI_Hi_LL[0]
data=data.drop([27,29,31,34,36,37,38,40,42,44,46,48,50,52,54]) # duplicate data
A_CI_Hi_LL[0]=[]
A_CI_Hi_LL[0]=data

#clean data
data = A_CI_Hi_LL[1]
data=data.drop([17,22]) # duplicate data
A_CI_Hi_LL[1]=[]
A_CI_Hi_LL[1]=data

data = A_CI_Hi_LL[2]
data=data.drop([20,22]) # duplicate data
A_CI_Hi_LL[2]=[]
A_CI_Hi_LL[2]=data

data = A_CI_Hi_LL[3]
data=data.drop([21]) # duplicate data
A_CI_Hi_LL[3]=[]
A_CI_Hi_LL[3]=data


#plot_response('Light',A_I_Hi_LL,measurement_days)
#plot_response('CO2',A_CI_Hi_LL,measurement_days)
df_I=replicates_to_Excel(A_I_Hi_LL,'H.Incana',0.21,'A-I curve','LL')
df_CI=replicates_to_Excel(A_CI_Hi_LL,'H.Incana',0.21,'A-CI curve','LL')
Gas_Exchange_data=Gas_Exchange_data.append(df_I)
Gas_Exchange_data=Gas_Exchange_data.append(df_CI)
#

A_I_Hi_LL = make_data('Light',2,'Hi','LL',measurement_days)
A_CI_Hi_LL = make_data('CO2',2,'Hi','LL',measurement_days)
#
data = A_I_Hi_LL[0]
data=data.drop([69]) # duplicate data
A_I_Hi_LL[0]=[]
A_I_Hi_LL[0]=data

data = A_CI_Hi_LL[0]
data=data.drop([58,60,62,64,67]) # duplicate data
A_CI_Hi_LL[0]=[]
A_CI_Hi_LL[0]=data

data = A_CI_Hi_LL[2]
data=data.drop([38]) # duplicate data
A_CI_Hi_LL[2]=[]
A_CI_Hi_LL[2]=data

#
#plot_response('Light',A_I_Hi_LL,measurement_days)
#plot_response('CO2',A_CI_Hi_LL,measurement_days)
df_I=replicates_to_Excel(A_I_Hi_LL,'H.Incana',0.02,'A-I curve','LL')
df_CI=replicates_to_Excel(A_CI_Hi_LL,'H.Incana',0.02,'A-CI curve','LL')
Gas_Exchange_data=Gas_Exchange_data.append(df_I)
Gas_Exchange_data=Gas_Exchange_data.append(df_CI)
#
#
# B.Nigra HL
measurement_days = [7,8,13,20]
A_I_BN_HL = make_data('Light',21,'Bn','HL',measurement_days)
A_CI_BN_HL = make_data('CO2',21,'Bn','HL',measurement_days)

data = A_I_BN_HL[0]
data=data.drop([10,11]) # duplicate data
A_I_BN_HL[0]=[]
A_I_BN_HL[0]=data

data = A_CI_BN_HL[3]
data=data.drop([21,22]) # duplicate data
A_CI_BN_HL[3]=[]
A_CI_BN_HL[3]=data

data = A_CI_BN_HL[0]
data=data.drop([23]) # duplicate data
A_CI_BN_HL[0]=[]
A_CI_BN_HL[0]=data

data = A_CI_BN_HL[1]
data=data.drop([21]) # duplicate data
A_CI_BN_HL[1]=[]
A_CI_BN_HL[1]=data

data = A_CI_BN_HL[2]
data=data.drop([21]) # duplicate data
A_CI_BN_HL[2]=[]
A_CI_BN_HL[2]=data

#plot_response('Light',A_I_BN_HL,measurement_days)
#plot_response('CO2',A_CI_BN_HL,measurement_days)
df_I=replicates_to_Excel(A_I_BN_HL,'B.Nigra',0.21,'A-I curve','HL')
df_CI=replicates_to_Excel(A_CI_BN_HL,'B.Nigra',0.21,'A-CI curve','HL')
Gas_Exchange_data=Gas_Exchange_data.append(df_I)
Gas_Exchange_data=Gas_Exchange_data.append(df_CI)

A_I_BN_HL = make_data('Light',2,'Bn','HL',measurement_days)
A_CI_BN_HL = make_data('CO2',2,'Bn','HL',measurement_days)

data = A_I_BN_HL[2]
data=data.drop([43]) # duplicate data
A_I_BN_HL[2]=[]
A_I_BN_HL[2]=data

#plot_response('Light',A_I_BN_HL,measurement_days)
#plot_response('CO2',A_CI_BN_HL,measurement_days)

df_I=replicates_to_Excel(A_I_BN_HL,'B.Nigra',0.02,'A-I curve','HL')
df_CI=replicates_to_Excel(A_CI_BN_HL,'B.Nigra',0.02,'A-CI curve','HL')
Gas_Exchange_data=Gas_Exchange_data.append(df_I)
Gas_Exchange_data=Gas_Exchange_data.append(df_CI)
#
#
# H.Incana HL
measurement_days = [5,11,18,22]
A_I_Hi_HL = make_data('Light',21,'Hi','HL',measurement_days)
A_CI_Hi_HL = make_data('CO2',21,'Hi','HL',measurement_days)

data = A_I_Hi_HL[0]
data=data.drop([3,4,6,8,10,12,14,15,17,19,21,23,24,26,27,29,30]) # duplicate data
A_I_Hi_HL[0]=[]
A_I_Hi_HL[0]=data

data = A_I_Hi_HL[1]
data=data.drop([7,11]) # duplicate data
data=data.drop([5]) # duplicate data
A_I_Hi_HL[1]=[]
A_I_Hi_HL[1]=data

data = A_CI_Hi_HL[0]
data=data.drop([33,35,37,39,42,43,44,50,51,53,55,57,59,60]) # duplicate data
A_CI_Hi_HL[0]=[]
A_CI_Hi_HL[0]=data

data = A_CI_Hi_HL[1]
data=data.drop([24]) # duplicate data
A_CI_Hi_HL[1]=[]
A_CI_Hi_HL[1]=data

data = A_CI_Hi_HL[2]
data=data.drop([22]) # duplicate data
A_CI_Hi_HL[2]=[]
A_CI_Hi_HL[2]=data

data = A_CI_Hi_HL[3]
data=data.drop([21]) # duplicate data
A_CI_Hi_HL[3]=[]
A_CI_Hi_HL[3]=data

#plot_response('Light',A_I_Hi_HL,measurement_days)
#plot_response('CO2',A_CI_Hi_HL,measurement_days)
df_I=replicates_to_Excel(A_I_Hi_HL,'H.Incana',0.21,'A-I curve','HL')
df_CI=replicates_to_Excel(A_CI_Hi_HL,'H.Incana',0.21,'A-CI curve','HL')
Gas_Exchange_data=Gas_Exchange_data.append(df_I)
Gas_Exchange_data=Gas_Exchange_data.append(df_CI)
#
A_I_Hi_HL = make_data('Light',2,'Hi','HL',measurement_days)
A_CI_Hi_HL = make_data('CO2',2,'Hi','HL',measurement_days)

data = A_I_Hi_HL[1]
data=data.drop([41]) # duplicate data
A_I_Hi_HL[1]=[]
A_I_Hi_HL[1]=data

data = A_CI_Hi_HL[0]
data=data.drop([62,64,66,68,70]) # duplicate data
A_CI_Hi_HL[0]=[]
A_CI_Hi_HL[0]=data

data = A_CI_Hi_HL[2]
data=data.drop([35]) # duplicate data
A_CI_Hi_HL[2]=[]
A_CI_Hi_HL[2]=data

#plot_response('Light',A_I_Hi_HL,measurement_days)
#plot_response('CO2',A_CI_Hi_HL,measurement_days)
df_I=replicates_to_Excel(A_I_Hi_HL,'H.Incana',0.02,'A-I curve','HL')
df_CI=replicates_to_Excel(A_CI_Hi_HL,'H.Incana',0.02,'A-CI curve','HL')
Gas_Exchange_data=Gas_Exchange_data.append(df_I)
Gas_Exchange_data=Gas_Exchange_data.append(df_CI)

#vpdL
vpdL = Gas_Exchange_data['VpdL'].values
vpdL_m = vpdL.mean(axis=0)
vpdL_e = vpdL.std(axis=0)

vpdL_Bn = Gas_Exchange_data[Gas_Exchange_data['Species']=='B.Nigra']
vpdL_Bn_LL = vpdL_Bn[vpdL_Bn['Treatment']=='LL']
vpdL_Bn_HL = vpdL_Bn[vpdL_Bn['Treatment']=='HL']

vpdL_Bn_LL=vpdL_Bn_LL['VpdL'].values
vpdL_Bn_HL=vpdL_Bn_HL['VpdL'].values

vpdL_Bn_LL_m = vpdL_Bn_LL.mean(axis=0)
vpdL_Bn_LL_e = vpdL_Bn_LL.std(axis=0)
vpdL_Bn_HL_m = vpdL_Bn_HL.mean(axis=0)
vpdL_Bn_HL_e = vpdL_Bn_HL.std(axis=0)

vpdL_Hi = Gas_Exchange_data[Gas_Exchange_data['Species']=='H.Incana']
vpdL_Hi_LL = vpdL_Hi[vpdL_Hi['Treatment']=='LL']
vpdL_Hi_HL = vpdL_Hi[vpdL_Hi['Treatment']=='HL']

vpdL_Hi_LL=vpdL_Hi_LL['VpdL'].values
vpdL_Hi_HL=vpdL_Hi_HL['VpdL'].values

vpdL_Hi_LL_m = vpdL_Hi_LL.mean(axis=0)
vpdL_Hi_LL_e = vpdL_Hi_LL.std(axis=0)
vpdL_Hi_HL_m = vpdL_Hi_HL.mean(axis=0)
vpdL_Hi_HL_e = vpdL_Hi_HL.std(axis=0)


#
#Gas_Exchange_data.to_excel(PATH + 'Gas_Exchange_data.xlsx', index = False)
