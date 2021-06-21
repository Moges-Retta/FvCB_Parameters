# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 20:02:52 2021

@author: Moges Retta
"""
from Gas_exchange_measurement import Gas_exchange_measurement
from Estimate_FvCB_parameters import Estimate_FvCB_parameters
import pandas as pd
import numpy as np
from scipy import stats

PATH = (r'\\WURNET.NL\Homes\retta001\My Documents\Project\2021\GasExchange\\')

#Correct the raw data for leakage
species = ['B.Nigra','H.Incana']
treatments = ['HL','LL']

Gas_Exchange_data_corr = pd.DataFrame([])
for plant in species:
    for treatment in treatments:
        gas_exch_measurement = Gas_exchange_measurement(0.21,plant,treatment)
        df = gas_exch_measurement.correct_leak(plant,treatment)
        Gas_Exchange_data_corr=Gas_Exchange_data_corr.append(df)
        
# Gas_Exchange_data_corr.to_excel(PATH + 'Gas_Exchange_data_leak_corr.xlsx', index = False)

Jmax_values = pd.DataFrame([], columns=['Species','Treatment','Replicate','Jmax','theta','Jmax_err','theta_err'])
Vcmax_values = pd.DataFrame([], columns=['Species','Treatment','Replicate','Vcmax','Tp','Vcmax_err','Tp_err','Sigma_gm','Sigma_gm_err'])
Rd_values = pd.DataFrame([], columns=['Species','Treatment','Replicate','Rd','Slope','Std.err'])
Rd_values_O2 = Rd_values;
Phi2LL_values = pd.DataFrame([], columns=['Species','Treatment','Replicate','Phi2LL','Phi2LL_err'])

cols = ['Plant','Treatment','Vcmax','Rd','Sco','Tp','Jmax','k2LL','theta',\
        'Vcmax_err','Rd_err','Sco_err','Tp_err','Jmax_err','theta_err','bH','bH_err',\
        'Std.err_bH','Std.err_bL','Sigma_gm','Sigma_gm_err']

df_params=pd.DataFrame([],columns=cols)

def Parameter_tabel(row,species,treatment,Rd,Rd_err,Jmax,sco,bH_bL,vcmax_var_gm):
    df_params.loc[row,'Rd']=Rd
    df_params.loc[row,'Rd_err']=Rd_err
    df_params.loc[row,'k2LL']=s*phi2LL[0][0]
    df_params.loc[row,'Jmax']=Jmax[0][0]
    df_params.loc[row,'Jmax_err']=Jmax[1]
    df_params.loc[row,'theta']=Jmax[0][1]
    df_params.loc[row,'theta_err']=Jmax[2]
    df_params.loc[row,'Plant']=species
    df_params.loc[row,'Treatment']=treatment
    df_params.loc[row,'Sco']=sco['Sco'].values
    df_params.loc[row,'Sco_err']=sco['Std.err'].values
    df_params.loc[row,'bH']=bH_bL['bH'].values
    df_params.loc[row,'Std.err_bH']=bH_bL['Std.err_bH'].values
    df_params.loc[row,'bL']=bH_bL['bH'].values
    df_params.loc[row,'Std.err_bL']=bH_bL['Std.err_bL'].values
    df_params.loc[row,'Vcmax']=vcmax_var_gm['Vcmax'].values
    df_params.loc[row,'Vcmax_err']=vcmax_var_gm['Vcmax_err'].values
    df_params.loc[row,'Tp']=vcmax_var_gm['Tp'].values
    df_params.loc[row,'Tp_err']=vcmax_var_gm['Tp_err'].values
    df_params.loc[row,'Sigma_gm']=vcmax_var_gm['Sigma_gm'].values
    df_params.loc[row,'Sigma_gm_err']=vcmax_var_gm['Sigma_gm_err'].values
    if row==4:
        df_params.to_excel(PATH +'Parameters_all.xlsx', index = False)

def Rd_tabel(Rds,species,treatment):
        df = pd.DataFrame([], columns=['Species','Treatment','Replicate','Rd','Slope','Std.err'])
        df['Species']=[species]*4
        df['Treatment']=[treatment]*4
        df['Rd']=Rds['Rd'].values
        df['Replicate']=Rds['Replicate'].values
        df['Std.err']=Rds['Std.err'].values
        df['Slope']=Rds['Slope'].values
        
        return df
    
def Jmax_tabel(Jmaxs,species,treatment):
        df = pd.DataFrame([], columns=['Species','Treatment','Replicate','Jmax','theta','Jmax_err','theta_err'])
        df['Species']=[species]*4
        df['Treatment']=[treatment]*4
        df['Jmax']=Jmaxs['Jmax'].values
        df['Replicate']=Jmaxs['Replicate'].values
        df['Jmax_err']=Jmaxs['Jmax_err'].values
        df['theta']=Jmaxs['theta'].values
        df['theta_err']=Jmaxs['theta_err'].values        
        return df

def Vcmax_tabel(Vcmaxs,species,treatment):
        df = pd.DataFrame([], columns=['Species','Treatment','Replicate','Vcmax','Tp','Vcmax_err','Tp_err','Sigma_gm','Sigma_gm_err'])
        df['Species']=[species]*4
        df['Treatment']=[treatment]*4
        df['Vcmax']=Vcmaxs['Vcmax'].values
        df['Replicate']=Vcmaxs['Replicate'].values
        df['Tp']=Vcmaxs['Tp'].values
        df['Vcmax_err']=Vcmaxs['Vcmax_err'].values
        df['Tp_err']=Vcmaxs['Tp_err'].values 
        df['Sigma_gm']=Vcmaxs['Sigma_gm'].values        
        df['Sigma_gm_err']=Vcmaxs['Sigma_gm_err'].values              
        return df


def Phi2LL_tabel(Phi2LLs,species,treatment):
        df = pd.DataFrame([], columns=['Species','Treatment','Replicate','Phi2LL','Phi2LL_err'])
        df['Species']=[species]*4
        df['Treatment']=[treatment]*4
        df['Phi2LL']=Phi2LLs['Phi2LL'].values
        df['Replicate']=Phi2LLs['Replicate'].values
        df['Phi2LL_err']=Phi2LLs['Phi2LL_err'].values
        
        return df    


# compare photosynthesis, gs and PhiPS2

gas_exch_measurement = Gas_exchange_measurement(0.21,'','')
ave_gas_Exchange_data = gas_exch_measurement.make_avareage_data() #all data
gas_exch_measurement.compare_A(ave_gas_Exchange_data)
gas_exch_measurement.compare_gs(ave_gas_Exchange_data)
gas_exch_measurement.compare_PhiPSII(ave_gas_Exchange_data)


# Estimate FvCB kinetics parameters

species = 'B.Nigra'
treatment = 'LL'
O = 0.21
gas_exch_measurement = Gas_exchange_measurement(O,species,treatment)
df_ave = gas_exch_measurement.get_average_values('ACI')
#gas_exch_measurement.plot_ave_A_CI(df_ave)
#gas_exch_measurement.plot_ave_gs_CI(df_ave)

df_ave = gas_exch_measurement.get_average_values('AI')
#gas_exch_measurement.plot_ave_A_I(df_ave)
#gas_exch_measurement.plot_ave_gs_I(df_ave)

gas_exch_measurement.set_O2(0.02)
parameters = Estimate_FvCB_parameters(gas_exch_measurement)
Rd_Bn_LL = parameters.estimate_Rd()
Rd=Rd_tabel(Rd_Bn_LL,species,treatment)
Rd_values=Rd_values.append(Rd)

gas_exch_measurement.set_O2(0.21)
parameters = Estimate_FvCB_parameters(gas_exch_measurement)
Rd_Hi_LL_O2 = parameters.estimate_Rd()
R_O2 = Rd_tabel(Rd_Hi_LL_O2,species,treatment)
Rd_values_O2=Rd_values_O2.append(R_O2)


#Rd_Bn_LL_common = parameters.estimate_Rd_common()
#Rd_common = np.mean(Rd_Bn_LL_common['Rd'].values,axis=0)
#Rd_common_err = np.nanstd(Rd_Bn_LL_common['Rd'].values/2,axis=0)

#gas_exch_measurement.set_O2(0.21)
#parameters = Estimate_FvCB_parameters(gas_exch_measurement)
#Rd_Bn_LL_O2 = parameters.estimate_Rd()

Rd = np.mean(Rd_Bn_LL['Rd'].values,axis=0)
s = np.nanmean(Rd_Bn_LL['Slope'].values,axis=0)
Rd_err = np.nanstd(Rd_Bn_LL['Rd'].values/2,axis=0)


bH_bL = parameters.estimate_bH_bL(Rd_Bn_LL['Rd'].values)
sco = parameters.estimate_Sco(Rd_Bn_LL['Rd'].values)
        
O = 0.21
gas_exch_measurement.set_O2(O)
parameters = Estimate_FvCB_parameters(gas_exch_measurement)
phi2LL_individual = parameters.estimate_individual_phi2LL()
phi2LL = parameters.estimate_phi2LL()
#phi2LLs=Phi2LL_tabel(phi2LL_individual,species,treatment)
#Phi2LL_values=Phi2LL_values.append(phi2LLs)

k2_Bn_LL = parameters.calculate_k2(Rd_Bn_LL['Slope'])

inputs = {'s':Rd_Bn_LL['Slope'].values,'phi2LL':phi2LL_individual['Phi2LL'].values}
Jmax_individual  = parameters.estimate_individual_Jmax(inputs)
inputs = {'s':s,'PHI2LL':phi2LL[0][0]}
Jmax  = parameters.estimate_Jmax(inputs)
#Jmaxs=Jmax_tabel(Jmax_individual,species,treatment)
#Jmax_values=Jmax_values.append(Jmaxs)

inputs = {'Rd':Rd_Bn_LL['Rd'].values,'Jmax':Jmax[0][0],'Theta':Jmax[0][1],\
          'k2LL':Rd_Bn_LL['Slope'].values*phi2LL_individual['Phi2LL'].values,'Sco':sco['Sco'].values}
vcmax_full = parameters.estimate_Vcmax(inputs)
vcmax_var_gm = parameters.estimate_Vcmax_var_gm(inputs)
vcmax_const_gm = parameters.estimate_Vcmax_constant_gm(inputs)

inputs = {'Rd':Rd_Bn_LL['Rd'].values,'Jmax':Jmax_individual['Jmax'].values,\
          'Theta':Jmax_individual['theta'].values,\
          'k2LL':Rd_Bn_LL['Slope'].values*phi2LL_individual['Phi2LL'].values,'Sco':sco['Sco'].values}
vcmax_individual = parameters.estimate_individual_Vcmax(inputs)
Vcmaxs=Vcmax_tabel(vcmax_individual,species,treatment)
Vcmax_values=Vcmax_values.append(Vcmaxs)


# NRH procedure, input Rd,phi2ll,lump: estimates: vcmax, Jmax, Tp, sigma
gm = parameters.NRH_A_gm(Rd)

inputs = {'s':gm.loc['lump','estimate'],'PHI2LL':phi2LL[0][0]}
Jmax  = parameters.estimate_Jmax(inputs)

inputs = {'Rd':[Rd]*4,'Jmax':Jmax[0][0],'Theta':Jmax[0][1],'gms':[gm.loc['gm','estimate']]*4,
          'k2LL':[gm.loc['lump','estimate']*phi2LL[0][0]]*4,'Sco':sco['Sco'].values}

vcmax_jmax = parameters.estimate_Vcmax_Jmax(inputs)

Parameter_tabel(0,species,treatment,Rd,Rd_err,Jmax,sco,bH_bL,vcmax_var_gm)


species = 'B.Nigra'
treatment = 'HL'
O = 0.21
gas_exch_measurement = Gas_exchange_measurement(O,species,treatment)
df_ave = gas_exch_measurement.get_average_values('ACI')
#gas_exch_measurement.plot_ave_A_CI(df_ave)
#gas_exch_measurement.plot_ave_gs_CI(df_ave)

df_ave = gas_exch_measurement.get_average_values('AI')
#gas_exch_measurement.plot_ave_A_I(df_ave)
#gas_exch_measurement.plot_ave_gs_I(df_ave)

gas_exch_measurement.set_O2(0.02)
parameters = Estimate_FvCB_parameters(gas_exch_measurement)
Rd_Bn_LL = parameters.estimate_Rd()
Rd = np.mean(Rd_Bn_LL['Rd'].values,axis=0)
s = np.nanmean(Rd_Bn_LL['Slope'].values,axis=0)
Rd_err = np.nanstd(Rd_Bn_LL['Rd'].values/2,axis=0)
Rd=Rd_tabel(Rd_Bn_LL,species,treatment)
Rd_values=Rd_values.append(Rd)

gas_exch_measurement.set_O2(0.21)
parameters = Estimate_FvCB_parameters(gas_exch_measurement)
Rd_Hi_LL_O2 = parameters.estimate_Rd()
R_O2 = Rd_tabel(Rd_Hi_LL_O2,species,treatment)
Rd_values_O2=Rd_values_O2.append(R_O2)


#parameters.compare_k2(k2_Bn_HL,k2_Bn_LL,'HL','LL')
bH_bL = parameters.estimate_bH_bL(Rd_Bn_LL['Rd'].values)
sco = parameters.estimate_Sco(Rd_Bn_LL['Rd'].values)
#sco_common = parameters.estimate_Sco(Rd_Bn_LL_common['Rd'].values)

#Rd_Bn_LL_common = parameters.estimate_Rd_common()
#Rd_common = np.mean(Rd_Bn_LL_common['Rd'].values,axis=0)
#Rd_common_err = np.nanstd(Rd_Bn_LL_common['Rd'].values/2,axis=0)

#Rd_Bn_LL_abs_adj = parameters.estimate_abs_adjusted_Rd()
#Rd_abs_adj = np.mean(Rd_Bn_LL_abs_adj['Rd'].values,axis=0)
#Rd_abs_adj_err = np.nanstd(Rd_Bn_LL_abs_adj['Rd'].values/2,axis=0)

O = 0.21
gas_exch_measurement.set_O2(O)
parameters = Estimate_FvCB_parameters(gas_exch_measurement)
phi2LL_individual = parameters.estimate_individual_phi2LL()
phi2LL = parameters.estimate_phi2LL()
#phi2LLs=Phi2LL_tabel(phi2LL_individual,species,treatment)
#Phi2LL_values=Phi2LL_values.append(phi2LLs)
k2_Bn_HL = parameters.calculate_k2(Rd_Bn_LL['Slope'])

Rd_Bn_LL_O2 = parameters.estimate_Rd()

inputs = {'s':Rd_Bn_LL['Slope'].values,'phi2LL':phi2LL_individual['Phi2LL'].values}
Jmax_individual  = parameters.estimate_individual_Jmax(inputs)
inputs = {'s':s,'PHI2LL':phi2LL[0][0]}
Jmax  = parameters.estimate_Jmax(inputs)
#Jmaxs=Jmax_tabel(Jmax_individual,species,treatment)
#Jmax_values=Jmax_values.append(Jmaxs)

inputs = {'Rd':Rd_Bn_LL['Rd'].values,'Jmax':Jmax[0][0],'Theta':Jmax[0][1],\
          'k2LL':Rd_Bn_LL['Slope'].values*phi2LL_individual['Phi2LL'].values,'Sco':sco['Sco'].values}
vcmax_full = parameters.estimate_Vcmax(inputs)
vcmax_var_gm = parameters.estimate_Vcmax_var_gm(inputs)
vcmax_const_gm = parameters.estimate_Vcmax_constant_gm(inputs)

inputs = {'Rd':Rd_Bn_LL['Rd'].values,'Jmax':Jmax_individual['Jmax'].values,\
          'Theta':Jmax_individual['theta'].values,\
          'k2LL':Rd_Bn_LL['Slope'].values*phi2LL_individual['Phi2LL'].values,'Sco':sco['Sco'].values}
vcmax_individual = parameters.estimate_individual_Vcmax(inputs)
Vcmaxs=Vcmax_tabel(vcmax_individual,species,treatment)
Vcmax_values=Vcmax_values.append(Vcmaxs)

# NRH procedure, input Rd,phi2ll,lump: estimates: vcmax, Jmax, Tp, sigma
gm = parameters.NRH_A_gm(Rd)

inputs = {'Rd':[Rd]*4,'Jmax':Jmax[0][0],'Theta':Jmax[0][1],'gms':[gm.loc['gm','estimate']]*4,
          'k2LL':[gm.loc['lump','estimate']*phi2LL[0][0]]*4,'Sco':3.259}

vcmax_jmax = parameters.estimate_Vcmax_Jmax(inputs)

Parameter_tabel(1,species,treatment,Rd,Rd_err,Jmax,sco,bH_bL,vcmax_var_gm)


species = 'H.Incana'
treatment = 'LL'
O = 0.21
gas_exch_measurement = Gas_exchange_measurement(O,species,treatment)
df_ave = gas_exch_measurement.get_average_values('ACI')
#gas_exch_measurement.plot_ave_A_CI(df_ave)
#gas_exch_measurement.plot_ave_gs_CI(df_ave)

df_ave = gas_exch_measurement.get_average_values('AI')
#gas_exch_measurement.plot_ave_A_I(df_ave)
#gas_exch_measurement.plot_ave_gs_I(df_ave)

gas_exch_measurement.set_O2(0.02)
parameters = Estimate_FvCB_parameters(gas_exch_measurement)
Rd_Bn_LL = parameters.estimate_Rd()
Rd = np.mean(Rd_Bn_LL['Rd'].values,axis=0)
s = np.nanmean(Rd_Bn_LL['Slope'].values,axis=0)
Rd_err = np.nanstd(Rd_Bn_LL['Rd'].values/2,axis=0)
Rd=Rd_tabel(Rd_Bn_LL,species,treatment)
Rd_values=Rd_values.append(Rd)
#Rd_Bn_LL_common = parameters.estimate_Rd_common()
#Rd_common = np.mean(Rd_Bn_LL_common['Rd'].values,axis=0)
#Rd_common_err = np.nanstd(Rd_Bn_LL_common['Rd'].values/2,axis=0)
#
#Rd_Bn_LL_abs_adj = parameters.estimate_abs_adjusted_Rd()
#Rd_abs_adj = np.mean(Rd_Bn_LL_abs_adj['Rd'].values,axis=0)
#Rd_abs_adj_err = np.nanstd(Rd_Bn_LL_abs_adj['Rd'].values/2,axis=0)


O = 0.21
gas_exch_measurement.set_O2(O)
parameters = Estimate_FvCB_parameters(gas_exch_measurement)
phi2LL_individual = parameters.estimate_individual_phi2LL()
phi2LL = parameters.estimate_phi2LL()
#phi2LLs=Phi2LL_tabel(phi2LL_individual,species,treatment)
#Phi2LL_values=Phi2LL_values.append(phi2LLs)

k2_Hi_LL = parameters.calculate_k2(Rd_Bn_LL['Slope'])


inputs = {'s':Rd_Bn_LL['Slope'].values,'phi2LL':phi2LL_individual['Phi2LL'].values}
Jmax_individual  = parameters.estimate_individual_Jmax(inputs)
inputs = {'s':s,'PHI2LL':phi2LL[0][0]}
Jmax  = parameters.estimate_Jmax(inputs)
#Jmaxs=Jmax_tabel(Jmax_individual,species,treatment)
#Jmax_values=Jmax_values.append(Jmaxs)
bH_bL = parameters.estimate_bH_bL(Rd_Bn_LL['Rd'].values)
sco = parameters.estimate_Sco(Rd_Bn_LL['Rd'].values)
#sco_common = parameters.estimate_Sco(Rd_Bn_LL_common['Rd'].values)

inputs = {'Rd':Rd_Bn_LL['Rd'].values,'Jmax':Jmax[0][0],'Theta':Jmax[0][1],\
          'k2LL':Rd_Bn_LL['Slope'].values*phi2LL_individual['Phi2LL'].values,'Sco':sco['Sco'].values}
vcmax_full = parameters.estimate_Vcmax(inputs)
vcmax_var_gm = parameters.estimate_Vcmax_var_gm(inputs)
vcmax_const_gm = parameters.estimate_Vcmax_constant_gm(inputs)

inputs = {'Rd':Rd_Bn_LL['Rd'].values,'Jmax':Jmax_individual['Jmax'].values,\
          'Theta':Jmax_individual['theta'].values,\
          'k2LL':Rd_Bn_LL['Slope'].values*phi2LL_individual['Phi2LL'].values,'Sco':sco['Sco'].values}
vcmax_individual = parameters.estimate_individual_Vcmax(inputs)
Vcmaxs=Vcmax_tabel(vcmax_individual,species,treatment)
Vcmax_values=Vcmax_values.append(Vcmaxs)

# NRH procedure, input Rd,phi2ll,lump: estimates: vcmax, Jmax, Tp, sigma
gm = parameters.NRH_A_gm(Rd)

inputs = {'Rd':[Rd]*4,'Jmax':Jmax[0][0],'Theta':Jmax[0][1],'gms':[gm.loc['gm','estimate']]*4,
          'k2LL':[gm.loc['lump','estimate']*phi2LL[0][0]]*4,'Sco':3.259}

vcmax_jmax = parameters.estimate_Vcmax_Jmax(inputs)

Parameter_tabel(2,species,treatment,Rd,Rd_err,Jmax,sco,bH_bL,vcmax_var_gm)

species = 'H.Incana'
treatment = 'HL'
O = 0.21
gas_exch_measurement = Gas_exchange_measurement(O,species,treatment)
df_ave = gas_exch_measurement.get_average_values('ACI')
#gas_exch_measurement.plot_ave_A_CI(df_ave)
#gas_exch_measurement.plot_ave_gs_CI(df_ave)

df_ave = gas_exch_measurement.get_average_values('AI')
#gas_exch_measurement.plot_ave_A_I(df_ave)
#gas_exch_measurement.plot_ave_gs_I(df_ave)

gas_exch_measurement.set_O2(0.02)
parameters = Estimate_FvCB_parameters(gas_exch_measurement)
Rd_Bn_LL = parameters.estimate_Rd()
Rd = np.mean(Rd_Bn_LL['Rd'].values,axis=0)
s = np.nanmean(Rd_Bn_LL['Slope'].values,axis=0)
Rd_err = np.nanstd(Rd_Bn_LL['Rd'].values/2,axis=0)
Rd=Rd_tabel(Rd_Bn_LL,species,treatment)
Rd_values=Rd_values.append(Rd)
#Rd_Bn_LL_common = parameters.estimate_Rd_common()
#Rd_common = np.mean(Rd_Bn_LL_common['Rd'].values,axis=0)
#Rd_common_err = np.nanstd(Rd_Bn_LL_common['Rd'].values/2,axis=0)

O = 0.21
gas_exch_measurement.set_O2(O)
parameters = Estimate_FvCB_parameters(gas_exch_measurement)
phi2LL_individual = parameters.estimate_individual_phi2LL()
phi2LL = parameters.estimate_phi2LL()
#phi2LLs=Phi2LL_tabel(phi2LL_individual,species,treatment)
#Phi2LL_values=Phi2LL_values.append(phi2LLs)

k2_Hi_HL = parameters.calculate_k2(Rd_Bn_LL['Slope'])


inputs = {'s':Rd_Bn_LL['Slope'].values,'phi2LL':phi2LL_individual['Phi2LL'].values}
Jmax_individual  = parameters.estimate_individual_Jmax(inputs)
inputs = {'s':s,'PHI2LL':phi2LL[0][0]}
Jmax  = parameters.estimate_Jmax(inputs)
Jmaxs = Jmax_tabel(Jmax_individual,species,treatment)
Jmax_values = Jmax_values.append(Jmaxs)

sco = parameters.estimate_Sco(Rd_Bn_LL['Rd'].values)
bH_bL = parameters.estimate_bH_bL(Rd_Bn_LL['Rd'].values)
#sco_common = parameters.estimate_Sco(Rd_Bn_LL_common['Rd'].values)

inputs = {'Rd':Rd_Bn_LL['Rd'].values,'Jmax':Jmax[0][0],'Theta':Jmax[0][1],\
          'k2LL':Rd_Bn_LL['Slope'].values*phi2LL_individual['Phi2LL'].values,'Sco':sco['Sco'].values}

vcmax_full = parameters.estimate_Vcmax(inputs)
vcmax_var_gm = parameters.estimate_Vcmax_var_gm(inputs)
vcmax_const_gm = parameters.estimate_Vcmax_constant_gm(inputs)

vcmax_jmax = parameters.estimate_Vcmax_Jmax(inputs)

inputs = {'Rd':Rd_Bn_LL['Rd'].values,'Jmax':Jmax_individual['Jmax'].values,\
          'Theta':Jmax_individual['theta'].values,\
          'k2LL':Rd_Bn_LL['Slope'].values*phi2LL_individual['Phi2LL'].values,'Sco':sco['Sco'].values}
vcmax_individual = parameters.estimate_individual_Vcmax(inputs)
Vcmaxs=Vcmax_tabel(vcmax_individual,species,treatment)
Vcmax_values=Vcmax_values.append(Vcmaxs)

# NRH procedure, input Rd,phi2ll,lump: estimates: vcmax, Jmax, Tp, sigma
gm = parameters.NRH_A_gm(Rd)

inputs = {'Rd':[Rd]*4,'Jmax':Jmax[0][0],'Theta':Jmax[0][1],'gms':[gm.loc['gm','estimate']]*4,
          'k2LL':[gm.loc['lump','estimate']*phi2LL[0][0]]*4,'Sco':3.259}

vcmax_jmax = parameters.estimate_Vcmax_Jmax(inputs)


# inputs = {'Rd':[Rd]*4,'Jmax':Jmax_individual['Jmax'].values,'Theta':Jmax_individual['theta'].values,\
#           'k2LL':[gm.loc['lump','estimate']*phi2LL[0][0]]*4}

# vcmax_Bush = parameters.estimate_Vcmax_Bush(inputs)
vcmax_Bush_XY = parameters.estimate_Vcmax_Bush_XY(inputs)


#parameters.compare_k2(k2_Hi_HL,k2_Hi_LL,'HL','LL')

Parameter_tabel(3,species,treatment,Rd,Rd_err,Jmax,sco,bH_bL,vcmax_var_gm)

boxplot = Rd_values.boxplot(column=['Rd'], by=['Treatment','Species'],figsize = (8,10),grid=False,layout=(2, 1))
boxplot[0].set_ylabel('R$_d$ (µmol $m^{-2}$ $s^{-1}$)')
boxplot.set_title('')

boxplot = Jmax_values.boxplot(column=['Jmax'], by=['Treatment','Species'],figsize = (8,10),grid=False,layout=(2, 1),fontsize=16)
boxplot[0].set_ylabel('J$_{max}$ (µmol $m^{-2}$ $s^{-1}$)',fontsize=16)

#Phi2LL_values.to_excel(PATH +'Parameters_Phi2LL.xlsx', index = False)
boxplot = Phi2LL_values.boxplot(column=['Phi2LL'], by=['Treatment','Species'],figsize = (8,10),grid=False,layout=(2, 1),fontsize=16)
boxplot[0].set_ylabel("\u03A6$_{PSII,LL}$ (-)",fontsize=16)

boxplot = Vcmax_values.boxplot(column=['Vcmax'], by=['Treatment','Species'],figsize = (8,10),grid=False,layout=(2, 1),fontsize=16)
boxplot[0].set_ylabel('V$_{cmax}$ (µmol $m^{-2}$ $s^{-1}$)',fontsize=16)

parameters.compare_k2(k2_Hi_HL,k2_Bn_HL,'H.Incana','B.Nigra')
parameters.compare_k2(k2_Hi_LL,k2_Bn_LL,'H.Incana','B.Nigra')

p_treatment = parameters.anova_test_treatments(Rd_values)
p_species = parameters.anova_test_species(Rd_values)

p_treatment = parameters.anova_test_treatments(Rd_values_O2)
p_species = parameters.anova_test_species(Rd_values_O2)

# Compare Rd between 2 % O2 and 21 % O2 
plants = Rd_values['Species'].unique()
treatments = Rd_values['Treatment'].unique()
replicates = Rd_values['Replicate'].unique()
cols = ['Species','Treatment','p']
compare_Rd_O2_level = pd.DataFrame([],columns = cols)
for plant in plants:
    x = Rd_values[Rd_values['Species']==plant]
    y = Rd_values_O2[Rd_values_O2['Species']==plant]    
    for treatment in treatments:
        df=pd.DataFrame([],columns=cols)
        xd = x[x['Treatment']==treatment]
        yd = y[y['Treatment']==treatment]            
        rd1 = xd['Rd'].values
        rd2 = yd['Rd'].values
        [t,p]= stats.ttest_ind(rd1,rd2, equal_var = False)
        df.loc[0,'Species']=plant
        df.loc[0,'Treatment']=treatment
        df.loc[0,'p']=p       
        compare_Rd_O2_level = compare_Rd_O2_level.append(df)
#Rd_values.to_excel(PATH + 'Parameters_Rd_300.xlsx', index = False)
#Rd_values_O2.to_excel(PATH + 'Parameters_Rd_300_21_O2.xlsx', index = False)

#file_path = (r'\\WURNET.NL\Homes\retta001\My Documents\Project\2021\GasExchange\Parameters_Rd_300.xlsx')
#df_old = pd.read_excel('Parameters_Rd_300.xlsx')
#df_old2 = pd.read_excel('Parameters_Rd_300_21_O2.xlsx')
#writer = pd.ExcelWriter(file_path, engine = 'xlsxwriter')
#df_old.to_excel(writer, sheet_name = 'Rd 2% O2')
#Rd_values_O2.to_excel(writer, sheet_name = 'Rd 21% O2')
#compare_Rd_O2_level.to_excel(writer, sheet_name = 'compare_2_21_O2-770')
#writer.save()
#writer.close()     
#Jmax_values.to_excel(PATH +'Parameters_Jmax_theta.xlsx', index = False)
#Vcmax_values.to_excel(PATH +'Parameters_Vcmax_variable_gm_.xlsx', index = False)
#df_params.to_excel(PATH +'Parameters_all.xlsx', index = False)
