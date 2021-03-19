# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 20:02:52 2021

@author: Moges Retta
"""
from Gas_exchange_measurement import Gas_exchange_measurement
from Estimate_FvCB_parameters import Estimate_FvCB_parameters

import pandas as pd
PATH = (r'\\WURNET.NL\Homes\retta001\My Documents\Project\2021\GasExchange\\')

columns = ['Species','Treatment','Response','Ci','A','Iinc','PhiPS2','Std.dev A','gs','Std.dev gs','Std.dev PhiPS2']
ave_gas_Exchange_data = pd.DataFrame([])
df = pd.DataFrame([],columns=columns )

species = ['B.Nigra','H.Incana']
treatments = ['HL','LL']
O = 0.21

# Make average data of ACI and AI
for plant in species:
        for treatment in treatments:
            gas_exch_measurement = Gas_exchange_measurement(O,plant,treatment)
            [I_ave_ci,Ci_ave_ci,A_ave_ci,phiPS2,gs_ave_ci,A_std,gs_std,phiPS2_std] = gas_exch_measurement.average_A_CI()
            df = pd.DataFrame([],columns=columns )            
            df['Ci']=Ci_ave_ci; df['A']=A_ave_ci; df['Iinc']=I_ave_ci; 
            df['Std.dev A']=A_std; df['gs']=gs_ave_ci;df['PhiPS2']=phiPS2;df['Std.dev PhiPS2']=phiPS2_std;
            df['Std.dev gs']=gs_std;df['Species']=plant; df['Treatment']=treatment; df['Response']='ACI'; 
            ave_gas_Exchange_data=ave_gas_Exchange_data.append(df)
            [I_ave_i,Ci_ave_i,A_ave_i,gs_ave_i,phiPS2,A_std,gs_std,phiPS2_std]  = gas_exch_measurement.average_A_I()            
            df = pd.DataFrame([],columns=columns )
            df['Ci']=Ci_ave_i; df['A']=A_ave_i; df['Iinc']=I_ave_i; 
            df['Std.dev A']=A_std; df['gs']=gs_ave_i;df['PhiPS2']=phiPS2;df['Std.dev PhiPS2']=phiPS2_std;
            df['Std.dev gs']=gs_std;df['Species']=plant; df['Treatment']=treatment; df['Response']='AI'; 
            ave_gas_Exchange_data=ave_gas_Exchange_data.append(df)
            
##ave_gas_Exchange_data.to_excel(PATH + 'Ave_Gas_Exchange_data.xlsx', index = False)

species = 'B.Nigra'
treatment = 'LL'
O = 0.21

gas_exch_measurement = Gas_exchange_measurement(O,species,treatment)

[I_ave_ci,Ci_ave_ci,A_ave_ci,phiPS2,gs_ave_ci,A_std,gs_std,phiPS2_std] = gas_exch_measurement.average_A_CI()
#gas_exch_measurement.plot_ave_A_CI(Ci_ave_ci,A_ave_ci,A_std)
#gas_exch_measurement.plot_ave_gs_CI(Ci_ave_ci,gs_ave_ci,gs_std)

[I_ave_i,Ci_ave_i,A_ave_i,gs_ave_i,phiPS2,A_std,gs_std,phiPS2_std] = gas_exch_measurement.average_A_I()
#gas_exch_measurement.plot_ave_A_I(I_ave_i,A_ave_i,A_std)
#gas_exch_measurement.plot_ave_gs_I(I_ave_i,gs_ave_i,gs_std)

species = 'B.Nigra'
treatment = 'HL'
O = 0.21
gas_exch_measurement = Gas_exchange_measurement(O,species,treatment)
[I_ave_ci,Ci_ave_ci,A_ave_ci,phiPS2,gs_ave_ci,A_std,gs_std,phiPS2_std]  = gas_exch_measurement.average_A_CI()
#gas_exch_measurement.plot_ave_A_CI(Ci_ave_ci,A_ave_ci,A_std)
#gas_exch_measurement.plot_ave_gs_CI(Ci_ave_ci,gs_ave_ci,gs_std)

[I_ave_i,Ci_ave_i,A_ave_i,gs_ave_i,phiPS2,A_std,gs_std,phiPS2_std]  = gas_exch_measurement.average_A_I()
#gas_exch_measurement.plot_ave_A_I(I_ave_i,A_ave_i,A_std)
#gas_exch_measurement.plot_ave_gs_I(I_ave_i,gs_ave_i,gs_std)

species = 'H.Incana'
treatment = 'HL'
O = 0.21
gas_exch_measurement = Gas_exchange_measurement(O,species,treatment)

[I_ave_ci,Ci_ave_ci,A_ave_ci,phiPS2,gs_ave_ci,A_std,gs_std,phiPS2_std]  = gas_exch_measurement.average_A_CI()
#gas_exch_measurement.plot_ave_A_CI(Ci_ave_ci,A_ave_ci,A_std)
#gas_exch_measurement.plot_ave_gs_CI(Ci_ave_ci,gs_ave_ci,gs_std)

[I_ave_i,Ci_ave_i,A_ave_i,gs_ave_i,phiPS2,A_std,gs_std,phiPS2_std]  = gas_exch_measurement.average_A_I()
#gas_exch_measurement.plot_ave_A_I(I_ave_i,A_ave_i,A_std)
#gas_exch_measurement.plot_ave_gs_I(I_ave_i,gs_ave_i,gs_std)

species = 'H.Incana'
treatment = 'LL'
O = 0.21

gas_exch_measurement = Gas_exchange_measurement(O,species,treatment)
[I_ave_ci,Ci_ave_ci,A_ave_ci,phiPS2,gs_ave_ci,A_std,gs_std,phiPS2_std]  = gas_exch_measurement.average_A_CI()
#gas_exch_measurement.plot_ave_A_CI(Ci_ave_ci,A_ave_ci,A_std)
#gas_exch_measurement.plot_ave_gs_CI(Ci_ave_ci,gs_ave_ci,gs_std)

[I_ave_i,Ci_ave_i,A_ave_i,gs_ave_i,phiPS2,A_std,gs_std,phiPS2_std] = gas_exch_measurement.average_A_I()
#gas_exch_measurement.plot_ave_A_I(I_ave_i,A_ave_i,A_std)
#gas_exch_measurement.plot_ave_gs_I(I_ave_i,gs_ave_i,gs_std)

gas_exch_measurement.compare_A(ave_gas_Exchange_data)
gas_exch_measurement.compare_gs(ave_gas_Exchange_data)
gas_exch_measurement.compare_PhiPSII(ave_gas_Exchange_data)

# Estimate Rd values
Rd_values = pd.DataFrame([], columns=['Species','Treatment','Replicate','Rd','Std.err'])

def Rd_tabel(Rds,species,treatment):
        df = pd.DataFrame([], columns=['Species','Treatment','Replicate','Rd','Std.err'])
        df['Species']=[species]*4
        df['Treatment']=[treatment]*4
        df['Rd']=Rds['Rd'].values
        df['Replicate']=Rds['Replicate'].values
        df['Std.err']=Rds['Std.err'].values
        return df

species = 'H.Incana'
treatment = 'LL'
O = 0.02
gas_exch_measurement = Gas_exchange_measurement(O,species,treatment)
parameters = Estimate_FvCB_parameters(gas_exch_measurement)
Rd_Hi_LL = parameters.estimate_Rd()
Rd = Rd_tabel(Rd_Hi_LL,species,treatment)
Rd_values=Rd_values.append(Rd)

species = 'H.Incana'
treatment = 'HL'
O = 0.02
gas_exch_measurement = Gas_exchange_measurement(O,species,treatment)
parameters = Estimate_FvCB_parameters(gas_exch_measurement)
Rd_Hi_HL = parameters.estimate_Rd()
Rd =Rd_tabel(Rd_Hi_HL,species,treatment)
Rd_values=Rd_values.append(Rd)


species = 'B.Nigra'
treatment = 'HL'
O = 0.02
gas_exch_measurement = Gas_exchange_measurement(O,species,treatment)
parameters = Estimate_FvCB_parameters(gas_exch_measurement)
Rd_Bn_HL = parameters.estimate_Rd()
Rd=Rd_tabel(Rd_Bn_HL,species,treatment)
Rd_values=Rd_values.append(Rd)


species = 'B.Nigra'
treatment = 'LL'
O = 0.02
gas_exch_measurement = Gas_exchange_measurement(O,species,treatment)
parameters = Estimate_FvCB_parameters(gas_exch_measurement)
Rd_Bn_LL = parameters.estimate_Rd()
Rd=Rd_tabel(Rd_Bn_LL,species,treatment)
Rd_values=Rd_values.append(Rd)
p_treatment = parameters.anova_test_treatments(Rd_values)
p_species = parameters.anova_test_species(Rd_values)

#Rd_values.to_excel(PATH + 'Parameters_Rd.xlsx', index = False)
boxplot = Rd_values.boxplot(column=['Rd'], by=['Treatment','Species'],figsize = (8,10),grid=False,layout=(2, 1))
boxplot[0].set_ylabel('R$_d$ (µmol $m^{-2}$ $s^{-1}$)')
boxplot.set_title('')