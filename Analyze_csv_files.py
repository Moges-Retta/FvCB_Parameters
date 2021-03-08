# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:04:47 2021

@author: retta001
"""
import pandas as pd
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
        
# B.Nigra LL
measurement_days = [3,9,10]
A_I_BN_LL = make_data('AI',21,'Bn','LL',measurement_days)
A_CI_BN_LL = make_data('ACI',21,'Bn','LL',measurement_days)

# H.Icana LL
measurement_days = [4,6,12]
A_I_Hi_LL = make_data('AI',21,'Hi','LL',measurement_days)
A_CI_Hi_LL = make_data('ACI',21,'Hi','LL',measurement_days)

# B.Nigra HL
measurement_days = [8,13]
A_I_BN_HL = make_data('AI',21,'Bn','HL',measurement_days)
A_CI_BN_HL = make_data('ACI',21,'Bn','HL',measurement_days)

# H.Icana HL
measurement_days = [5,11]
A_I_Hi_HL = make_data('AI',21,'Hi','HL',measurement_days)
A_CI_Hi_HL = make_data('ACI',21,'Hi','HL',measurement_days)