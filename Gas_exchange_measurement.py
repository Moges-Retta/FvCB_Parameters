# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 18:28:34 2021

@author: Moges
Measurement data of Ci, Iinc
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Gas_exchange_measurement:
    data = pd.read_excel ('Gas_exchange_data.xlsx') 
    FORMAT = ['Replicate','Species','Treatment','Measurement type','Oxygen level','Net CO2 assimilation rate', 'Intercellular CO2 concentration', 'PhiPS2','Irradiance','Stomatal conductance for CO2']
    df_selected = data[FORMAT]
#    print(df_selected)
    A_CI = df_selected[df_selected['Measurement type']=='A-CI curve']
    A_I= df_selected[df_selected['Measurement type']=='A-I curve']
    gs_CI= df_selected[df_selected['Measurement type']=='A-CI curve']
    gs_I= df_selected[df_selected['Measurement type']=='A-I curve']
#    print(A_CI)

    def __init__(self,O2,species,treatment):
            self.O2=O2
            self.species=species        
            self.treatment=treatment
            
                   
    def get_O2(self):
        return self.O2
      
        
    def get_A_Ci(self):
        return self.A_CI
    
    
    def get_A_I(self):
        return self.A_I


    def get_species(self):
        return self.species
    
    
    def get_treatment(self):
        return self.treatment


    def plot_A_CI(self):
        A_CI_d = self.A_CI[self.A_CI['Oxygen level']==self.get_O2()]
        A_CI_d = A_CI_d[A_CI_d['Species']==self.get_species()]
        A_CI_d = A_CI_d[A_CI_d['Treatment']==self.get_treatment()]
        replicates = A_CI_d['Replicate'].values
        replicates=np.unique(replicates)
        for replicate in replicates:
            A_CI_r= A_CI_d[A_CI_d['Replicate']==replicate]
            Ci = A_CI_r['Intercellular CO2 concentration'].values
            A = A_CI_r['Net CO2 assimilation rate'].values
            plt.plot(Ci,A,'o')
            plt.title("A-CI")
        plt.show()


    def plot_A_I(self):
        A_I_d  = self.A_I[self.A_I['Oxygen level']==self.get_O2()]
        A_I_d =  A_I_d[A_I_d['Species']==self.get_species()]
        A_I_d = A_I_d[A_I_d['Treatment']==self.get_treatment()]
        replicates = A_I_d['Replicate'].values
        replicates=np.unique(replicates)
        for replicate in replicates:
            A_I_r= A_I_d[A_I_d['Replicate']==replicate]
            I = A_I_r['Irradiance'].values
            A = A_I_r['Net CO2 assimilation rate'].values
            plt.plot(I,A,'o')
            plt.title("A-I")
        plt.show()


    def average_A_CI(self):
        A_CI_d = self.A_CI[self.A_CI['Oxygen level']==self.get_O2()]
        A_CI_d = A_CI_d[A_CI_d['Species']==self.get_species()]
        A_CI_d = A_CI_d[A_CI_d['Treatment']==self.get_treatment()]
        replicates = A_CI_d['Replicate'].values
        replicates=np.unique(replicates)
        Ci_ave = np.empty((len(A_CI_d[A_CI_d['Replicate']==1]), 0), int)
        A_ave = np.empty((len(A_CI_d[A_CI_d['Replicate']==1]), 0), int)
        I_ave = np.empty((len(A_CI_d[A_CI_d['Replicate']==1]), 0), int)
        gs_ave = np.empty((len(A_CI_d[A_CI_d['Replicate']==1]), 0), int)

        for replicate in replicates:
            A_CI_r= A_CI_d[A_CI_d['Replicate']==replicate]
            Ci = A_CI_r['Intercellular CO2 concentration'].values
            A = A_CI_r['Net CO2 assimilation rate'].values
            I = A_CI_r['Irradiance'].values
            gs = A_CI_r['Stomatal conductance for CO2'].values
            Ci_ave= np.append(Ci_ave, np.array([Ci]).transpose(),axis=1)
            A_ave=np.append(A_ave, np.array([A]).transpose(),axis=1)
            I_ave=np.append(I_ave, np.array([I]).transpose(),axis=1)
            gs_ave=np.append(gs_ave, np.array([gs]).transpose(),axis=1)
        return [I_ave.mean(axis=1),Ci_ave.mean(axis=1),A_ave.mean(axis=1),gs_ave.mean(axis=1),A_ave.std(axis=1),gs_ave.std(axis=1)]
    
    
    def average_A_I(self):
        A_I_d  = self.A_I[self.A_I['Oxygen level']==self.get_O2()]
        A_I_d =  A_I_d[A_I_d['Species']==self.get_species()]
        A_I_d = A_I_d[A_I_d['Treatment']==self.get_treatment()]
        replicates = A_I_d['Replicate'].values
        replicates=np.unique(replicates)
        I_ave = np.empty((len(A_I_d[A_I_d['Replicate']==1]), 0), int)
        Ci_ave = np.empty((len(A_I_d[A_I_d['Replicate']==1]), 0), int)
        A_ave = np.empty((len(A_I_d[A_I_d['Replicate']==1]), 0), int)
        A_ave = np.empty((len(A_I_d[A_I_d['Replicate']==1]), 0), int)
        gs_ave = np.empty((len(A_I_d[A_I_d['Replicate']==1]), 0), int)

        for replicate in replicates:
            A_I_r= A_I_d[A_I_d['Replicate']==replicate]
            I = A_I_r['Irradiance'].values
            A = A_I_r['Net CO2 assimilation rate'].values
            Ci = A_I_r['Intercellular CO2 concentration'].values
            gs = A_I_r['Stomatal conductance for CO2'].values
            I_ave=np.append(I_ave, np.array([I]).transpose(),axis=1)
            A_ave=np.append(A_ave, np.array([A]).transpose(),axis=1)
            Ci_ave=np.append(Ci_ave, np.array([Ci]).transpose(),axis=1)
            gs_ave=np.append(gs_ave, np.array([gs]).transpose(),axis=1)
        #self.plot_ave(I_ave,A_ave,"Gross photosynthesis (µmol m-2 s-1)","Irradiance (µmol m-2 s-1)")
        return [I_ave.mean(axis=1),Ci_ave.mean(axis=1),A_ave.mean(axis=1),gs_ave.mean(axis=1),A_ave.std(axis=1),gs_ave.std(axis=1)]
     
        
    def plot_ave_A_CI(self,x1,y1,yerr):
        plt.errorbar(x1,y1,yerr,fmt='o')
        plt.xlabel('Intercellular $CO_2$ (µmol $mol^{-1}$)')
        plt.ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)')
        plt.show()


    def plot_ave_A_I(self,x1,y1,yerr):
        plt.errorbar(x1,y1,yerr,fmt='o')
        plt.xlabel('Irradiance (µmol $m^{-2}$ $s^{-1}$)')
        plt.ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)')
        plt.show()


    def plot_ave_gs_I(self,x1,y1,yerr):
        plt.errorbar(x1,y1,yerr,fmt='o')
        plt.xlabel('Irradiance (µmol $m^{-2}$ $s^{-1}$)')
        plt.ylabel('Stomatal conductance (mol $m^{-2}$ $s^{-1}$)')
        plt.show()        


    def plot_ave_gs_CI(self,x1,y1,yerr):
        plt.errorbar(x1,y1,yerr,fmt='o')
        plt.xlabel('Intercellular $CO_2$ (µmol $mol^{-1}$)')
        plt.ylabel('Stomatal conductance (mol $m^{-2}$ $s^{-1}$)')
        plt.show()