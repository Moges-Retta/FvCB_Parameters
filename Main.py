# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 20:02:52 2021

@author: Moges Retta
"""
from Gas_exchange_measurement import Gas_exchange_measurement
species = 'B.Nigra'
treatment = 'LL'
O = 0.21
gas_exch_measurement = Gas_exchange_measurement(O,species,treatment)

[I_ave_ci,Ci_ave_ci,A_ave_ci,gs_ave_ci,A_std,gs_std] = gas_exch_measurement.average_A_CI()
gas_exch_measurement.plot_ave_A_CI(Ci_ave_ci,A_ave_ci,A_std)
gas_exch_measurement.plot_ave_gs_CI(Ci_ave_ci,gs_ave_ci,gs_std)

[I_ave_i,i_ave_i,A_ave_i,gs_ave_i,A_std,gs_std] = gas_exch_measurement.average_A_I()
gas_exch_measurement.plot_ave_A_I(I_ave_i,A_ave_i,A_std)
gas_exch_measurement.plot_ave_gs_I(I_ave_i,gs_ave_i,gs_std)