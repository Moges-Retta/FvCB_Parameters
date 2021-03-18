# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 20:02:52 2021

@author: Moges Retta
"""
from Gas_exchange_measurement import Gas_exchange_measurement
import pandas as pd
PATH = (r'\\WURNET.NL\Homes\retta001\My Documents\Project\2021\GasExchange\\')

columns = ['Species','Treatment','Response','Ci','A','Iinc','PhiPS2','Std.dev A','gs','Std.dev gs','Std.dev PhiPS2']
ave_gas_Exchange_data = pd.DataFrame([])
df = pd.DataFrame([],columns=columns )

species = 'B.Nigra'
treatment = 'LL'
O = 0.21
gas_exch_measurement = Gas_exchange_measurement(O,species,treatment)

[I_ave_ci,Ci_ave_ci,A_ave_ci,phiPS2,gs_ave_ci,A_std,gs_std,phiPS2_std] = gas_exch_measurement.average_A_CI()
#gas_exch_measurement.plot_ave_A_CI(Ci_ave_ci,A_ave_ci,A_std)
#gas_exch_measurement.plot_ave_gs_CI(Ci_ave_ci,gs_ave_ci,gs_std)

df = pd.DataFrame([],columns=columns )
df['Ci']=Ci_ave_ci; df['A']=A_ave_ci; df['Iinc']=I_ave_ci; 
df['Std.dev A']=A_std; df['gs']=gs_ave_ci;df['PhiPS2']=phiPS2;df['Std.dev PhiPS2']=phiPS2_std;
df['Std.dev gs']=gs_std;df['Species']='BN'; df['Treatment']='LL'; df['Response']='ACI'; 
ave_gas_Exchange_data=ave_gas_Exchange_data.append(df)

[I_ave_i,Ci_ave_i,A_ave_i,gs_ave_i,phiPS2,A_std,gs_std,phiPS2_std] = gas_exch_measurement.average_A_I()
#gas_exch_measurement.plot_ave_A_I(I_ave_i,A_ave_i,A_std)
#gas_exch_measurement.plot_ave_gs_I(I_ave_i,gs_ave_i,gs_std)


df = pd.DataFrame([],columns=columns )
df['Ci']=Ci_ave_i; df['A']=A_ave_i; df['Iinc']=I_ave_i; 
df['Std.dev A']=A_std; df['gs']=gs_ave_i;df['PhiPS2']=phiPS2;df['Std.dev PhiPS2']=phiPS2_std;
df['Std.dev gs']=gs_std;df['Species']='BN'; df['Treatment']='LL'; df['Response']='AI'; 
ave_gas_Exchange_data=ave_gas_Exchange_data.append(df)


species = 'B.Nigra'
treatment = 'HL'
O = 0.21
gas_exch_measurement = Gas_exchange_measurement(O,species,treatment)

[I_ave_ci,Ci_ave_ci,A_ave_ci,phiPS2,gs_ave_ci,A_std,gs_std,phiPS2_std]  = gas_exch_measurement.average_A_CI()
#gas_exch_measurement.plot_ave_A_CI(Ci_ave_ci,A_ave_ci,A_std)
#gas_exch_measurement.plot_ave_gs_CI(Ci_ave_ci,gs_ave_ci,gs_std)


df = pd.DataFrame([],columns=columns )
df['Ci']=Ci_ave_ci; df['A']=A_ave_ci; df['Iinc']=I_ave_ci; 
df['Std.dev A']=A_std; df['gs']=gs_ave_ci;df['PhiPS2']=phiPS2;df['Std.dev PhiPS2']=phiPS2_std;
df['Std.dev gs']=gs_std;df['Species']='BN'; df['Treatment']='HL'; df['Response']='ACI'; 
ave_gas_Exchange_data=ave_gas_Exchange_data.append(df)


[I_ave_i,Ci_ave_i,A_ave_i,gs_ave_i,phiPS2,A_std,gs_std,phiPS2_std]  = gas_exch_measurement.average_A_I()
#gas_exch_measurement.plot_ave_A_I(I_ave_i,A_ave_i,A_std)
#gas_exch_measurement.plot_ave_gs_I(I_ave_i,gs_ave_i,gs_std)

df = pd.DataFrame([],columns=columns )
df['Ci']=Ci_ave_i; df['A']=A_ave_i; df['Iinc']=I_ave_i; 
df['Std.dev A']=A_std; df['gs']=gs_ave_i;df['PhiPS2']=phiPS2;df['Std.dev PhiPS2']=phiPS2_std;
df['Std.dev gs']=gs_std;df['Species']='BN'; df['Treatment']='HL'; df['Response']='AI'; 
ave_gas_Exchange_data=ave_gas_Exchange_data.append(df)


species = 'H.Incana'
treatment = 'HL'
O = 0.21
gas_exch_measurement = Gas_exchange_measurement(O,species,treatment)

[I_ave_ci,Ci_ave_ci,A_ave_ci,phiPS2,gs_ave_ci,A_std,gs_std,phiPS2_std]  = gas_exch_measurement.average_A_CI()
#gas_exch_measurement.plot_ave_A_CI(Ci_ave_ci,A_ave_ci,A_std)
#gas_exch_measurement.plot_ave_gs_CI(Ci_ave_ci,gs_ave_ci,gs_std)

df = pd.DataFrame([],columns=columns )
df['Ci']=Ci_ave_ci; df['A']=A_ave_ci; df['Iinc']=I_ave_ci; 
df['Std.dev A']=A_std; df['gs']=gs_ave_ci;df['PhiPS2']=phiPS2;df['Std.dev PhiPS2']=phiPS2_std;
df['Std.dev gs']=gs_std;df['Species']='Hi'; df['Treatment']='HL'; df['Response']='ACI'; 
ave_gas_Exchange_data=ave_gas_Exchange_data.append(df)


[I_ave_i,Ci_ave_i,A_ave_i,gs_ave_i,phiPS2,A_std,gs_std,phiPS2_std]  = gas_exch_measurement.average_A_I()
#gas_exch_measurement.plot_ave_A_I(I_ave_i,A_ave_i,A_std)
#gas_exch_measurement.plot_ave_gs_I(I_ave_i,gs_ave_i,gs_std)

df = pd.DataFrame([],columns=columns )
df['Ci']=Ci_ave_i; df['A']=A_ave_i; df['Iinc']=I_ave_i; 
df['Std.dev A']=A_std; df['gs']=gs_ave_i;df['PhiPS2']=phiPS2;df['Std.dev PhiPS2']=phiPS2_std;
df['Std.dev gs']=gs_std;df['Species']='Hi'; df['Treatment']='HL'; df['Response']='AI'; 
ave_gas_Exchange_data=ave_gas_Exchange_data.append(df)


species = 'H.Incana'
treatment = 'LL'
O = 0.21

gas_exch_measurement = Gas_exchange_measurement(O,species,treatment)

[I_ave_ci,Ci_ave_ci,A_ave_ci,phiPS2,gs_ave_ci,A_std,gs_std,phiPS2_std]  = gas_exch_measurement.average_A_CI()
#gas_exch_measurement.plot_ave_A_CI(Ci_ave_ci,A_ave_ci,A_std)
#gas_exch_measurement.plot_ave_gs_CI(Ci_ave_ci,gs_ave_ci,gs_std)

df = pd.DataFrame([],columns=columns )
df['Ci']=Ci_ave_ci; df['A']=A_ave_ci; df['Iinc']=I_ave_ci; 
df['Std.dev A']=A_std; df['gs']=gs_ave_ci;df['PhiPS2']=phiPS2;df['Std.dev PhiPS2']=phiPS2_std;
df['Std.dev gs']=gs_std;df['Species']='Hi'; df['Treatment']='LL'; df['Response']='ACI'; 
ave_gas_Exchange_data=ave_gas_Exchange_data.append(df)

[I_ave_i,Ci_ave_i,A_ave_i,gs_ave_i,phiPS2,A_std,gs_std,phiPS2_std] = gas_exch_measurement.average_A_I()
#gas_exch_measurement.plot_ave_A_I(I_ave_i,A_ave_i,A_std)
#gas_exch_measurement.plot_ave_gs_I(I_ave_i,gs_ave_i,gs_std)

df = pd.DataFrame([],columns=columns )
df['Ci']=Ci_ave_i; df['A']=A_ave_i; df['Iinc']=I_ave_i; 
df['Std.dev A']=A_std; df['gs']=gs_ave_i;df['PhiPS2']=phiPS2;df['Std.dev PhiPS2']=phiPS2_std;
df['Std.dev gs']=gs_std;df['Species']='Hi'; df['Treatment']='LL'; df['Response']='AI'; 
ave_gas_Exchange_data=ave_gas_Exchange_data.append(df)

#ave_gas_Exchange_data.to_excel(PATH + 'Ave_Gas_Exchange_data.xlsx', index = False)

gas_exch_measurement.compare_A(ave_gas_Exchange_data)
gas_exch_measurement.compare_gs(ave_gas_Exchange_data)
gas_exch_measurement.compare_PhiPSII(ave_gas_Exchange_data)


species = 'B.Nigra'
treatment = 'LL'
O = 0.21
gas_exch_measurement = Gas_exchange_measurement(O,species,treatment)
stat_LL_I = gas_exch_measurement.t_test_A_I(treatment)
treatment = 'HL'
stat_HL_I = gas_exch_measurement.t_test_A_I(treatment)

treatment = 'LL'
stat_LL_CI = gas_exch_measurement.t_test_A_CI(treatment)
treatment = 'HL'
stat_HL_CI = gas_exch_measurement.t_test_A_CI(treatment)