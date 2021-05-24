# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 18:28:34 2021

@author: Moges
Measurement data of Ci, Iinc
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PATH = (r'\\WURNET.NL\Homes\retta001\My Documents\Project\2021\GasExchange\\')


class Gas_exchange_measurement:
    data = pd.read_excel ('Gas_Exchange_data.xlsx') 
#    FORMAT = ['Replicate','Species','Treatment','Measurement type','Oxygen level','Net CO2 assimilation rate', 'Intercellular CO2 concentration', 'PhiPS2','Irradiance','Stomatal conductance for CO2','CO2S','Trmmol','BLCond']
    FORMAT = ['Replicate','Species','Treatment','Measurement type','Oxygen level','Net CO2 assimilation rate', 'Intercellular CO2 concentration', 'PhiPS2','Irradiance','Stomatal conductance for CO2','CO2R']
    df_selected = data[FORMAT]
    A_CI = df_selected[df_selected['Measurement type']=='A-CI curve']
    A_I= df_selected[df_selected['Measurement type']=='A-I curve']
    gs_CI= df_selected[df_selected['Measurement type']=='A-CI curve']
    gs_I= df_selected[df_selected['Measurement type']=='A-I curve']

    def __init__(self,O2,species,treatment):
            self.O2=O2
            self.species=species        
            self.treatment=treatment
            
     
    def set_O2(self,O2):
        self.O2=O2


    def set_species(self,species):
        self.species=species
    
    
    def set_treatment(self,treatment):
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


    def get_AI_data(self):
        A_I_d  = self.A_I[self.A_I['Oxygen level']==self.get_O2()]
        A_I_d =  A_I_d[A_I_d['Species']==self.get_species()]
        A_I_d = A_I_d[A_I_d['Treatment']==self.get_treatment()]
        return A_I_d


    def get_ACI_data(self):
        A_CI = self.get_A_Ci()
        A_CI_d  = A_CI[A_CI['Oxygen level']==self.get_O2()]
        A_CI_d =  A_CI_d[A_CI_d['Species']==self.get_species()]
        A_CI_d = A_CI_d[A_CI_d['Treatment']==self.get_treatment()]
        return A_CI_d

            
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
        replicates = A_CI_d['Replicate'].unique()

        cols = ['Irradiance','Intercellular CO2 concentration','Net CO2 assimilation rate',\
                'PhiPS2','Stomatal conductance for CO2','Photo_err','gs_err','PhiPS2_err']
        df_ave = pd.DataFrame([],columns = cols)
        df_ci = pd.DataFrame([])
        df_A = pd.DataFrame([])
        df_I = pd.DataFrame([])
        df_gs = pd.DataFrame([])
        df_phi = pd.DataFrame([])
        count = 0

        for replicate in replicates:
            A_CI_r= A_CI_d[A_CI_d['Replicate']==replicate]
            Ci = A_CI_r['Intercellular CO2 concentration'].values
            A = A_CI_r['Net CO2 assimilation rate'].values
            I = A_CI_r['Irradiance'].values
            gs = A_CI_r['Stomatal conductance for CO2'].values
            PhiPS2 = A_CI_r['PhiPS2'].values
            
            df_ci.loc[:,count] = Ci
            df_A.loc[:,count] = A
            df_I.loc[:,count] = I
            df_gs.loc[:,count] = gs
            df_phi.loc[:,count] = PhiPS2

            count+=1
        
        df_ave.loc[:,'Irradiance'] = np.nanmean(df_I,axis=1)
        df_ave.loc[:,'Intercellular CO2 concentration'] = np.nanmean(df_ci,axis=1)
        df_ave.loc[:,'Net CO2 assimilation rate'] = np.nanmean(df_A,axis=1)
        df_ave.loc[:,'Stomatal conductance for CO2'] = np.nanmean(df_gs,axis=1)
        df_ave.loc[:,'PhiPS2'] = np.nanmean(df_phi,axis=1)
        df_ave.loc[:,'Photo_err'] = np.nanstd(df_A,axis=1)
        df_ave.loc[:,'gs_err'] = np.nanstd(df_gs,axis=1)
        df_ave.loc[:,'PhiPS2_err'] = np.nanstd(df_phi,axis=1)
        df_ave = df_ave.sort_values(by=['Intercellular CO2 concentration'])
        return df_ave
    
    def average_A_I(self):
        A_I_d  = self.A_I[self.A_I['Oxygen level']==self.get_O2()]
        A_I_d =  A_I_d[A_I_d['Species']==self.get_species()]
        A_I_d = A_I_d[A_I_d['Treatment']==self.get_treatment()]
        replicates = A_I_d['Replicate'].unique()

        df_ci = pd.DataFrame([])
        df_A = pd.DataFrame([])
        df_I = pd.DataFrame([])
        df_gs = pd.DataFrame([])
        df_phi = pd.DataFrame([])
        count = 0
        cols = ['Irradiance','Intercellular CO2 concentration','Net CO2 assimilation rate',\
                'PhiPS2','Stomatal conductance for CO2','Photo_err','gs_err','PhiPS2_err']
        df_ave = pd.DataFrame([],columns = cols)
        
        for replicate in replicates:
            A_I_r= A_I_d[A_I_d['Replicate']==replicate]
            I = A_I_r['Irradiance'].values
            A = A_I_r['Net CO2 assimilation rate'].values
            Ci = A_I_r['Intercellular CO2 concentration'].values
            gs = A_I_r['Stomatal conductance for CO2'].values
            PhiPS2 = A_I_r['PhiPS2'].values
            df_ci.loc[:,count] = Ci
            df_A.loc[:,count] = A
            df_I.loc[:,count] = I
            df_gs.loc[:,count] = gs
            df_phi.loc[:,count] = PhiPS2
            count+=1
            
        df_ave.loc[:,'Irradiance'] = np.nanmean(df_I,axis=1)
        df_ave.loc[:,'Intercellular CO2 concentration'] = np.nanmean(df_ci,axis=1)
        df_ave.loc[:,'Net CO2 assimilation rate'] = np.nanmean(df_A,axis=1)
        df_ave.loc[:,'Stomatal conductance for CO2'] = np.nanmean(df_gs,axis=1)
        df_ave.loc[:,'PhiPS2'] = np.nanmean(df_phi,axis=1)
        df_ave.loc[:,'Photo_err'] = np.nanstd(df_A,axis=1)
        df_ave.loc[:,'gs_err'] = np.nanstd(df_gs,axis=1)
        df_ave.loc[:,'PhiPS2_err'] = np.nanstd(df_phi,axis=1)        
        df_ave = df_ave.sort_values(by=['Irradiance'])            
        return df_ave
        
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
    
    def show_significant(self,p_values,A,x,axis):
        m=[]
        scale_factor=70 # to add p values to A       
        p_values = [element * scale_factor for element in p_values]        
        y = p_values+A;
        for stat in p_values:                
            if stat<0.05*scale_factor:
                m.append("*")
            else:
                m.append("")
        for i in range(len(x)):
                axis.plot(x[i]-60, y[i], color='red',marker=m[i])
                
    def show_significant_gs(self,p_values,gs,x,axis):
        m=[]
        scale_factor=0.5 # to add p values to A       
        p_values = [element * scale_factor for element in p_values]        
        y = p_values+gs;
        for stat in p_values:                
            if stat<0.05*scale_factor:
                m.append("*")
            else:
                m.append("")
        for i in range(len(x)):
                axis.plot(x[i]-60, y[i], color='red',marker=m[i])                
                    
    def compare_A(self,df):
        fig, ax = plt.subplots(2,2,constrained_layout=True)
        plt.rcParams["figure.figsize"] = (30,10)
        
        plants = ['H.Incana','B.Nigra']
        count = 0
        symbol=['ko','ks']
        for plant in plants:                
            A_CI_d = df[df['Species']==plant]
            A_CI_d = A_CI_d[A_CI_d['Treatment']=='HL']
            A_CI_d = A_CI_d[A_CI_d['Response']=='ACI']
            
            A_I_d = df[df['Species']==plant]
            A_I_d = A_I_d[A_I_d['Treatment']=='HL']
            A_I_d = A_I_d[A_I_d['Response']=='AI']

            AHL_CI = A_CI_d['A'].values
            Ci = A_CI_d['Ci'].values
            err = A_CI_d['Std.dev A'].values
            ax[0][0].errorbar(Ci,AHL_CI,err,fmt=symbol[count],mfc='white',mec='black',markersize=8)
            stat_HL_CI = self.t_test_A_CI('HL') 
            stat_HL_CI = stat_HL_CI['p_A'].values

            if plant == 'H.Incana':
                self.show_significant(stat_HL_CI,AHL_CI,Ci,ax[0][0])
                                
            AHL_I = A_I_d['A'].values
            Iinc = A_I_d['Iinc'].values
            err = A_I_d['Std.dev A'].values
            ax[1][0].errorbar(Iinc,AHL_I,err,fmt=symbol[count],mfc='white',mec='black',markersize=8)
            stat_HL_I = self.t_test_A_I('HL')
            stat_HL_I = stat_HL_I['p_A'].values
            
            if plant == 'H.Incana':
                self.show_significant(stat_HL_I,AHL_I,Iinc,ax[1][0])           
                 
            A_CI_d = df[df['Species']==plant]
            A_CI_d = A_CI_d[A_CI_d['Treatment']=='LL']
            A_CI_d = A_CI_d[A_CI_d['Response']=='ACI']
            A_I_d = A_CI_d[A_CI_d['Response']=='AI']
            
            ALL_CI = A_CI_d['A'].values
            Ci = A_CI_d['Ci'].values
            err = A_CI_d['Std.dev A'].values
            
            ax[0][1].errorbar(Ci,ALL_CI,err,fmt=symbol[count],mfc='white',mec='black',markersize=8)
            stat_LL_CI = self.t_test_A_CI('LL')
            stat_LL_CI = stat_LL_CI['p_A'].values

            if plant == 'H.Incana':
                self.show_significant(stat_LL_CI,ALL_CI,Ci,ax[0][1])           
            
            A_I_d = df[df['Species']==plant]
            A_I_d = A_I_d[A_I_d['Treatment']=='LL']
            A_I_d = A_I_d[A_I_d['Response']=='AI']
            
            ALL_I = A_I_d['A'].values
            Iinc = A_I_d['Iinc'].values
            err = A_I_d['Std.dev A'].values
            stat_LL_I = self.t_test_A_I('LL')
            stat_LL_I = stat_LL_I['p_A'].values
            if plant == 'H.Incana':
                self.show_significant(stat_LL_I,ALL_I,Iinc,ax[1][1]) 
                      
            ax[1][1].errorbar(Iinc, ALL_I,err,fmt=symbol[count],label=plant,mfc='white',mec='black',markersize=8)
            count+=1
            
        
        ax[0][0].set_ylabel("Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)",fontsize=16)
        ax[1][0].set_ylabel("Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)")
        ax[0][0].set_xlabel("Intercellular $CO_2$ (µmol $mol^{-1}$)")
        ax[1][0].set_xlabel("Irradiance (µmol $m^{-2}$ $s^{-1}$)")   
        ax[0][1].set_xlabel("Intercellular $CO_2$ (µmol $mol^{-1}$)")        
        ax[1][1].set_xlabel("Irradiance (µmol $m^{-2}$ $s^{-1}$)")
        ax[0][1].set_ylim(top=85)
        ax[0][0].set_ylim(top=85)
        
        ax[1][0].set_ylim(top=60)
        ax[1][1].set_ylim(top=60)
        
        ax[0][1].set_ylim(bottom=0)
        ax[1][0].set_ylim(bottom=-5)
        ax[1][1].set_ylim(bottom=-5)
        ax[0][0].set_ylim(bottom=0)
        
        ax[1][1].xaxis.set_ticks(np.arange(0, 2200, 400))
        ax[1][0].xaxis.set_ticks(np.arange(0, 2200, 400))
        ax[0][0].xaxis.set_ticks(np.arange(0, 2200, 400))
        ax[0][1].xaxis.set_ticks(np.arange(0, 2200, 400))
        
        ax[1][1].legend(loc='lower right', fontsize='x-large')     
#        fig.savefig('Compare_A_response.tiff', dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})

    def compare_gs(self,df):
        fig, ax = plt.subplots(2,2,constrained_layout=True)
        plt.rcParams["figure.figsize"] = (30,10)
        plants = ['H.Incana','B.Nigra']
        count = 0
        symbol=['ko','ks']
        for plant in plants:                
            A_CI_d = df[df['Species']==plant]
            A_CI_d = A_CI_d[A_CI_d['Treatment']=='HL']
            A_CI_d = A_CI_d[A_CI_d['Response']=='ACI']
            A_I_d = df[df['Species']==plant]
            A_I_d = A_I_d[A_I_d['Treatment']=='HL']
            A_I_d = A_I_d[A_I_d['Response']=='AI']
            
            AHL_CI = A_CI_d['gs'].values
            Ci = A_CI_d['Ci'].values
            err = A_CI_d['Std.dev gs'].values
            ax[0][0].errorbar(Ci,AHL_CI,err,fmt=symbol[count],mfc='white',mec='black',markersize=8)
            stat_HL_CI = self.t_test_A_CI('HL') 
            stat_HL_CI = stat_HL_CI['p_gs'].values

            if plant == 'H.Incana':
                self.show_significant_gs(stat_HL_CI,AHL_CI,Ci,ax[0][0])
                
            AHL_I = A_I_d['gs'].values
            Iinc = A_I_d['Iinc'].values
            err = A_I_d['Std.dev gs'].values
            ax[1][0].errorbar(Iinc,AHL_I,err,fmt=symbol[count],mfc='white',mec='black',markersize=8)
            stat_HL_I = self.t_test_A_I('HL')
            stat_HL_I = stat_HL_I['p_gs'].values
            
            if plant == 'H.Incana':
                self.show_significant_gs(stat_HL_I,AHL_I,Iinc,ax[1][0])               
                 
            A_CI_d = df[df['Species']==plant]
            A_CI_d = A_CI_d[A_CI_d['Treatment']=='LL']
            A_CI_d = A_CI_d[A_CI_d['Response']=='ACI']
            A_I_d = A_CI_d[A_CI_d['Response']=='AI']
            
            ALL_CI = A_CI_d['gs'].values
            Ci = A_CI_d['Ci'].values
            err = A_CI_d['Std.dev gs'].values
            
            ax[0][1].errorbar(Ci,ALL_CI,err,fmt=symbol[count],mfc='white',mec='black',markersize=8)
            stat_LL_CI = self.t_test_A_CI('LL')
            stat_LL_CI = stat_LL_CI['p_gs'].values

            if plant == 'H.Incana':
                self.show_significant_gs(stat_LL_CI,ALL_CI,Ci,ax[0][1])    
                
            A_I_d = df[df['Species']==plant]
            A_I_d = A_I_d[A_I_d['Treatment']=='LL']
            A_I_d = A_I_d[A_I_d['Response']=='AI']
            ALL_I = A_I_d['gs'].values
            Iinc = A_I_d['Iinc'].values
            err = A_I_d['Std.dev gs'].values
            
            ax[1][1].errorbar(Iinc, ALL_I,err,fmt=symbol[count],label=plant,mfc='white',mec='black',markersize=8)
            stat_LL_I = self.t_test_A_I('LL')
            stat_LL_I = stat_LL_I['p_gs'].values
            if plant == 'H.Incana':
                self.show_significant_gs(stat_LL_I,ALL_I,Iinc,ax[1][1]) 
            
            count+=1
        
        ax[0][0].set_ylabel("Stomatal conductance (mol $m^{-2}$ $s^{-1}$)")
        ax[1][0].set_ylabel("Stomatal conductance (mol $m^{-2}$ $s^{-1}$)")
        ax[0][0].set_xlabel("Intercellular $CO_2$ (µmol $mol^{-1}$)")
        ax[1][0].set_xlabel("Irradiance (µmol $m^{-2}$ $s^{-1}$)")   
        ax[0][1].set_xlabel("Intercellular $CO_2$ (µmol $mol^{-1}$)")        
        ax[1][1].set_xlabel("Irradiance (µmol $m^{-2}$ $s^{-1}$)")
        ax[0][1].set_ylim(top=1.4)
        ax[1][1].set_ylim(top=1.2)
        ax[0][0].set_ylim(top=1.4)
        ax[1][0].set_ylim(top=1.2)
        ax[0][1].set_ylim(bottom=0.2)
        ax[1][1].set_ylim(bottom=0.1)
        ax[0][0].set_ylim(bottom=0.2)
        ax[1][0].set_ylim(bottom=0.1)
        ax[1][1].xaxis.set_ticks(np.arange(0, 2200, 400))
        ax[1][0].xaxis.set_ticks(np.arange(0, 2200, 400))
        ax[0][0].xaxis.set_ticks(np.arange(0, 2200, 400))
        ax[0][1].xaxis.set_ticks(np.arange(0, 2200, 400))
        
        ax[1][1].legend(loc='lower right', fontsize='x-large')            
        

    def compare_PhiPSII(self,df):
        fig, ax = plt.subplots(2,2,constrained_layout=True)
        plt.rcParams["figure.figsize"] = (10,10)
        plants = ['H.Incana','B.Nigra']
        count = 0
        symbol=['ko','ks']
        for plant in plants:                
            A_CI_d = df[df['Species']==plant]
            A_CI_d = A_CI_d[A_CI_d['Treatment']=='HL']
            A_CI_d = A_CI_d[A_CI_d['Response']=='ACI']
            A_I_d = df[df['Species']==plant]
            A_I_d = A_I_d[A_I_d['Treatment']=='HL']
            A_I_d = A_I_d[A_I_d['Response']=='AI']
            
            AHL_CI = A_CI_d['PhiPS2'].values
            Ci = A_CI_d['Ci'].values
            err = A_CI_d['Std.dev PhiPS2'].values
            ax[0][0].errorbar(Ci,AHL_CI,err,fmt=symbol[count],mfc='white',mec='black',markersize=8)
            stat_HL_CI = self.t_test_A_CI('HL') 
            stat_HL_CI = stat_HL_CI['p_phi'].values

            if plant == 'H.Incana':
                self.show_significant_gs(stat_HL_CI,AHL_CI,Ci,ax[0][0])
                
            AHL_I = A_I_d['PhiPS2'].values
            Iinc = A_I_d['Iinc'].values
            err = A_I_d['Std.dev PhiPS2'].values
            ax[1][0].errorbar(Iinc,AHL_I,err,fmt=symbol[count],mfc='white',mec='black',markersize=8)
            stat_HL_I = self.t_test_A_I('HL')
            stat_HL_I = stat_HL_I['p_phi'].values
            
            if plant == 'H.Incana':
                self.show_significant_gs(stat_HL_I,AHL_I,Iinc,ax[1][0])             
                 
            A_CI_d = df[df['Species']==plant]
            A_CI_d = A_CI_d[A_CI_d['Treatment']=='LL']
            A_CI_d = A_CI_d[A_CI_d['Response']=='ACI']
            A_I_d = A_CI_d[A_CI_d['Response']=='AI']
            
            ALL_CI = A_CI_d['PhiPS2'].values
            Ci = A_CI_d['Ci'].values
            err = A_CI_d['Std.dev PhiPS2'].values
            
            ax[0][1].errorbar(Ci,ALL_CI,err,fmt=symbol[count],mfc='white',mec='black',markersize=8)
            stat_LL_CI = self.t_test_A_CI('LL')
            stat_LL_CI = stat_LL_CI['p_phi'].values
            if plant == 'H.Incana':
                self.show_significant_gs(stat_LL_CI,ALL_CI,Ci,ax[0][1])   
                
            A_I_d = df[df['Species']==plant]
            A_I_d = A_I_d[A_I_d['Treatment']=='LL']
            A_I_d = A_I_d[A_I_d['Response']=='AI']
            ALL_I = A_I_d['PhiPS2'].values
            Iinc = A_I_d['Iinc'].values
            err = A_I_d['Std.dev PhiPS2'].values
            
            ax[1][1].errorbar(Iinc, ALL_I,err,fmt=symbol[count],mfc='white',mec='black',label=plant,markersize=8)
            stat_LL_I = self.t_test_A_I('LL')
            stat_LL_I = stat_LL_I['p_phi'].values
            if plant == 'H.Incana':
                self.show_significant_gs(stat_LL_I,ALL_I,Iinc,ax[1][1])             
            
            count+=1
        
        ax[0][0].set_ylabel("\u03A6$_{PSII}$ (-)")
        ax[1][0].set_ylabel("\u03A6$_{PSII}$ (-)")
        ax[0][0].set_xlabel("Intercellular $CO_2$ (µmol $mol^{-1}$)")
        ax[1][0].set_xlabel("Irradiance (µmol $m^{-2}$ $s^{-1}$)")   
        ax[0][1].set_xlabel("Intercellular $CO_2$ (µmol $mol^{-1}$)")        
        ax[1][1].set_xlabel("Irradiance (µmol $m^{-2}$ $s^{-1}$)")
        ax[0][0].set_ylim(top=0.50)        
        ax[0][1].set_ylim(top=0.50)
        ax[1][1].set_ylim(top=1)
        ax[1][0].set_ylim(top=1)
        ax[0][0].set_ylim(bottom=0.10)        
        ax[0][1].set_ylim(bottom=0.10)
        ax[1][1].set_ylim(bottom=0.2)
        ax[1][0].set_ylim(bottom=0.2)
        ax[1][1].xaxis.set_ticks(np.arange(0, 2200, 400))
        ax[1][0].xaxis.set_ticks(np.arange(0, 2200, 400))
        ax[0][0].xaxis.set_ticks(np.arange(0, 2200, 400))
        ax[0][1].xaxis.set_ticks(np.arange(0, 2200, 400))
        
        ax[1][1].legend(loc='upper right', fontsize='x-large')                   
        
        
    def t_test_A_I(self,treatment):
        A_I_d_BN  = self.A_I[self.A_I['Species']=='B.Nigra']
        A_I_d_BN  = A_I_d_BN[A_I_d_BN['Oxygen level']==self.get_O2()]
        A_I_d_BN = A_I_d_BN[A_I_d_BN['Treatment']==treatment]
        
        A_I_d_Hi  = self.A_I[self.A_I['Species']=='H.Incana']        
        A_I_d_Hi  = A_I_d_Hi[A_I_d_Hi['Oxygen level']==self.get_O2()]
        A_I_d_Hi = A_I_d_Hi[A_I_d_Hi['Treatment']==treatment]
        p_values = []
        par_values = [100,120,150,180,200,250,300,400,550,800,1100,1500,1800,2200]*4
        A_I_d_BN['Irradiance']=par_values
        A_I_d_Hi['Irradiance']=par_values
        PARi = A_I_d_BN['Irradiance'].values
        PARi=np.unique(PARi)
        
        p_values = pd.DataFrame([],columns = ['p_A','p_gs','p_phi'])
        count = 0
        for Par in PARi:
            A_I_r_BN= A_I_d_BN[A_I_d_BN['Irradiance']==Par]
            A_I_r_Hi= A_I_d_Hi[A_I_d_Hi['Irradiance']==Par]
            AHi = A_I_r_Hi['Net CO2 assimilation rate'].values
            ABN = A_I_r_BN['Net CO2 assimilation rate'].values
            [t,p]= stats.ttest_ind(AHi,ABN, equal_var = False)
            p_values.loc[count,'p_A']=p
           
            gsHi = A_I_r_Hi['Stomatal conductance for CO2'].values
            gsBN = A_I_r_BN['Stomatal conductance for CO2'].values
            [t,p]= stats.ttest_ind(gsHi,gsBN, equal_var = False)
            p_values.loc[count,'p_gs']=p            
            
            phi_Hi = A_I_r_Hi['PhiPS2'].values
            phi_ABN = A_I_r_BN['PhiPS2'].values
            [t,p]= stats.ttest_ind(phi_Hi,phi_ABN, equal_var = False)
            p_values.loc[count,'p_phi']=p            
            count+=1
        return p_values
             
    def t_test_A_CI(self,treatment):
        A_CI_d_BN  = self.A_CI[self.A_CI['Species']=='B.Nigra']
        A_CI_d_BN  = A_CI_d_BN[A_CI_d_BN['Oxygen level']==self.get_O2()]
        A_CI_d_BN = A_CI_d_BN[A_CI_d_BN['Treatment']==treatment]
        
        A_CI_d_Hi  = self.A_CI[self.A_CI['Species']=='H.Incana']        
        A_CI_d_Hi  = A_CI_d_Hi[A_CI_d_Hi['Oxygen level']==self.get_O2()]
        A_CI_d_Hi = A_CI_d_Hi[A_CI_d_Hi['Treatment']==treatment]

        par_values = [400,300,250,200,150,100,500,600,700,850,1000,1200,1500,1800,2200]*4
        A_CI_d_BN['Irradiance']=par_values
        A_CI_d_Hi['Irradiance']=par_values
        PARi = A_CI_d_BN['Irradiance'].values
        PARi=np.unique(PARi)

        p_values = pd.DataFrame([],columns = ['p_A','p_gs','p_phi'])
        count = 0
        for Par in PARi:
            A_CI_r_BN= A_CI_d_BN[A_CI_d_BN['Irradiance']==Par]
            A_CI_r_Hi= A_CI_d_Hi[A_CI_d_Hi['Irradiance']==Par]
            AHi = A_CI_r_Hi['Net CO2 assimilation rate'].values
            ABN = A_CI_r_BN['Net CO2 assimilation rate'].values
            [t,p]= stats.ttest_ind(AHi,ABN, equal_var = False)
            p_values.loc[count,'p_A']=p
            gsHi = A_CI_r_Hi['Stomatal conductance for CO2'].values
            gsBN = A_CI_r_BN['Stomatal conductance for CO2'].values
            [t,p]= stats.ttest_ind(gsHi,gsBN, equal_var = False)
            p_values.loc[count,'p_gs']=p            
            phi_Hi = A_CI_r_Hi['PhiPS2'].values
            phi_BN = A_CI_r_BN['PhiPS2'].values
            [t,p]= stats.ttest_ind(phi_Hi,phi_BN, equal_var = False)
            p_values.loc[count,'p_phi']=p            
            count+=1
        return p_values
             
    
    def correct_leak(self):
        """
        Correct for leakage
        1. correction for CO2s based on linear relationship between CO2R-CO2S and
        CO2R
        2. correct CO2s for main measurement using the above model
        3. recalculate gbc and Ci and photo  
        """
        columns = ['Replicate','Species','Treatment','Measurement type',\
                   'Oxygen level','Net CO2 assimilation rate',\
                   'Intercellular CO2 concentration','PhiPS2','Irradiance',\
                   'Stomatal conductance for CO2','CO2R','CO2S','H2OR','H2OS',\
                   'Flow','Area','Trmmol','BLCond']
        Gas_Exchange_data_corr = pd.DataFrame([],columns=columns )
        ACI_corr = self.A_CI;
        AI_corr = self.A_I;
        cond = ACI_corr['Stomatal conductance for CO2'].values
        blcond = ACI_corr['BLCond'].values
        trmmol = ACI_corr['Trmmol'].values        
#        photo = ACI_corr['Net CO2 assimilation rate'].values
        CO2S = ACI_corr['CO2S'].values
        CO2R = ACI_corr['CO2R'].values
        H2OR = ACI_corr['H2OR'].values
        H2OS = ACI_corr['H2OS'].values
        flow = ACI_corr['Flow'].values
        area = ACI_corr['Area'].values

        model_delta_co2 = 0.0006*CO2R - 0.4661 # correct for CO2s
        co2s_corrected = CO2S+model_delta_co2
        a = (1000-H2OR)/(1000-H2OS)        
        photo_corr = flow*(CO2R-co2s_corrected*a/(100*area))           
        gbl_corr = 1/(1/cond+1.37/blcond)
        ci_corr = ((gbl_corr-trmmol/1000/2)*co2s_corrected-photo_corr)/(gbl_corr+trmmol/1000/2)
        
        ACI_corr.loc[:,['Net CO2 assimilation rate']]=photo_corr
        ACI_corr.loc[:,['Intercellular CO2 concentration']]=ci_corr        
        
        Gas_Exchange_data_corr= Gas_Exchange_data_corr.append(ACI_corr)
        
        cond = AI_corr['Stomatal conductance for CO2'].values
        blcond = AI_corr['BLCond'].values
        trmmol = AI_corr['Trmmol'].values        
        CO2S = AI_corr['CO2S'].values
        CO2R = AI_corr['CO2R'].values
        H2OR = AI_corr['H2OR'].values
        H2OS = AI_corr['H2OS'].values
        flow = AI_corr['Flow'].values
        area = AI_corr['Area'].values

        model_delta_co2 = 0.0006*CO2R - 0.4661 # correct for CO2s
        co2s_corrected = CO2S+model_delta_co2
        a = (1000-H2OR)/(1000-H2OS)        
        photo_corr = flow*(CO2R-co2s_corrected*a/(100*area))           
        gbl_corr = 1/(1/cond+1.37/blcond)
        ci_corr = ((gbl_corr-trmmol/1000/2)*co2s_corrected-photo_corr)/(gbl_corr+trmmol/1000/2)
        
        AI_corr.loc[:,['Net CO2 assimilation rate']]=photo_corr
        AI_corr.loc[:,['Intercellular CO2 concentration']]=ci_corr     
        
        Gas_Exchange_data_corr=Gas_Exchange_data_corr.append(AI_corr)
        
#        FORMAT = ['Replicate','Species','Treatment','Measurement type','Oxygen level','Net CO2 assimilation rate', 'Intercellular CO2 concentration', 'PhiPS2','Irradiance','Stomatal conductance for CO2']
#        Gas_Exchange_data_corr = Gas_Exchange_data_corr[FORMAT]
#        
#        Gas_Exchange_data_corr.to_excel(PATH + 'Gas_Exchange_data_leak_corr.xlsx', index = False)
        return Gas_Exchange_data_corr
        

        
        
        
        
        
        
        