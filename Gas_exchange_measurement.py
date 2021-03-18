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
#    plt.rcParams["figure.figsize"] = (5,5)

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
        PhiPSII_ave = np.empty((len(A_CI_d[A_CI_d['Replicate']==1]), 0), int)

        for replicate in replicates:
            A_CI_r= A_CI_d[A_CI_d['Replicate']==replicate]
            Ci = A_CI_r['Intercellular CO2 concentration'].values
            A = A_CI_r['Net CO2 assimilation rate'].values
            I = A_CI_r['Irradiance'].values
            gs = A_CI_r['Stomatal conductance for CO2'].values
            PhiPS2 = A_CI_r['PhiPS2'].values
           
            Ci_ave= np.append(Ci_ave, np.array([Ci]).transpose(),axis=1)
            A_ave=np.append(A_ave, np.array([A]).transpose(),axis=1)
            I_ave=np.append(I_ave, np.array([I]).transpose(),axis=1)
            gs_ave=np.append(gs_ave, np.array([gs]).transpose(),axis=1)
            phiPS2_ave=np.append(PhiPSII_ave, np.array([PhiPS2]).transpose(),axis=1)

        return [np.nanmean(I_ave,axis=1),np.nanmean(Ci_ave,axis=1),np.nanmean(A_ave,axis=1),np.nanmean(phiPS2_ave,axis=1),np.nanmean(gs_ave,axis=1),np.nanstd(A_ave,axis=1),np.nanstd(gs_ave,axis=1),np.nanstd(phiPS2_ave)]
    
    
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
        PhiPSII_ave = np.empty((len(A_I_d[A_I_d['Replicate']==1]), 0), int)

        for replicate in replicates:
            A_I_r= A_I_d[A_I_d['Replicate']==replicate]
            I = A_I_r['Irradiance'].values
            A = A_I_r['Net CO2 assimilation rate'].values
            Ci = A_I_r['Intercellular CO2 concentration'].values
            gs = A_I_r['Stomatal conductance for CO2'].values
            PhiPS2 = A_I_r['PhiPS2'].values
            I_ave=np.append(I_ave, np.array([I]).transpose(),axis=1)
            A_ave=np.append(A_ave, np.array([A]).transpose(),axis=1)
            Ci_ave=np.append(Ci_ave, np.array([Ci]).transpose(),axis=1)
            gs_ave=np.append(gs_ave, np.array([gs]).transpose(),axis=1)
            phiPS2_ave=np.append(PhiPSII_ave, np.array([PhiPS2]).transpose(),axis=1)
            
        #self.plot_ave(I_ave,A_ave,"Gross photosynthesis (µmol m-2 s-1)","Irradiance (µmol m-2 s-1)")
        return [np.nanmean(I_ave,axis=1),np.nanmean(Ci_ave,axis=1),np.nanmean(A_ave,axis=1),np.nanmean(gs_ave,axis=1),np.nanmean(phiPS2_ave,axis=1),np.nanstd(A_ave,axis=1),np.nanstd(gs_ave,axis=1),np.nanstd(phiPS2_ave)]
     
        
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
                    
    def compare_A(self,df):
        fig, ax = plt.subplots(2,2,constrained_layout=True)
        plt.rcParams["figure.figsize"] = (10,10)
        
        plants = ['Hi','BN']
        count = 0
        symbol=['ko','k<']
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

            if plant == 'Hi':
                self.show_significant(stat_HL_CI,AHL_CI,Ci,ax[0][0])
                                
            AHL_I = A_I_d['A'].values
            Iinc = A_I_d['Iinc'].values
            err = A_I_d['Std.dev A'].values
            ax[1][0].errorbar(Iinc,AHL_I,err,fmt=symbol[count],mfc='white',mec='black',markersize=8)
            stat_HL_I = self.t_test_A_I('HL')
            
            if plant == 'Hi':
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

            if plant == 'Hi':
                self.show_significant(stat_LL_CI,ALL_CI,Ci,ax[0][1])           
            
            A_I_d = df[df['Species']==plant]
            A_I_d = A_I_d[A_I_d['Treatment']=='LL']
            A_I_d = A_I_d[A_I_d['Response']=='AI']
            ALL_I = A_I_d['A'].values
            Iinc = A_I_d['Iinc'].values
            err = A_I_d['Std.dev A'].values
            stat_LL_I = self.t_test_A_I('LL')

            if plant == 'Hi':
                self.show_significant(stat_LL_I,ALL_I,Iinc,ax[1][1]) 
                
            if plant=='Hi':
                name="H. Incana"
            else:
                name="B. Nigra"
                      
            ax[1][1].errorbar(Iinc, ALL_I,err,fmt=symbol[count],label=name,mfc='white',mec='black',markersize=8)
            count+=1
            
        
        ax[0][0].set_ylabel("Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)",fontsize=16)
#        ax[0][1].set_ylabel("Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)")       
        ax[1][0].set_ylabel("Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)")
#        ax[1][1].set_ylabel("Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)")
        ax[0][0].set_xlabel("Intercellular $CO_2$ (µmol $mol^{-1}$)")
        ax[1][0].set_xlabel("Irradiance (µmol $mol^{-1}$)")   
        ax[0][1].set_xlabel("Intercellular $CO_2$ (µmol $mol^{-1}$)")        
        ax[1][1].set_xlabel("Irradiance (µmol $mol^{-1}$)")
        ax[0][1].set_ylim(top=80)
        ax[1][0].set_ylim(top=60)
        ax[1][1].set_ylim(top=60)
        
        ax[1][1].legend(loc='lower right', fontsize='x-large')     
        fig.savefig('Compare_A_response.tiff', dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})

    def compare_gs(self,df):
        fig, ax = plt.subplots(2,2,constrained_layout=True)
        plt.rcParams["figure.figsize"] = (10,10)
        plants = ['Hi','BN']
        count = 0
        symbol=['ko','k<']
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
            ax[0][0].errorbar(Ci,AHL_CI,err,fmt=symbol[count],markersize=8)
            
            AHL_I = A_I_d['gs'].values
            Iinc = A_I_d['Iinc'].values
            err = A_I_d['Std.dev gs'].values
            ax[1][0].errorbar(Iinc,AHL_I,err,fmt=symbol[count],markersize=8)
            
                 
            A_CI_d = df[df['Species']==plant]
            A_CI_d = A_CI_d[A_CI_d['Treatment']=='LL']
            A_CI_d = A_CI_d[A_CI_d['Response']=='ACI']
            A_I_d = A_CI_d[A_CI_d['Response']=='AI']
            
            ALL_CI = A_CI_d['gs'].values
            Ci = A_CI_d['Ci'].values
            err = A_CI_d['Std.dev gs'].values
            
            ax[0][1].errorbar(Ci,ALL_CI,err,fmt=symbol[count],markersize=8)
            
            A_I_d = df[df['Species']==plant]
            A_I_d = A_I_d[A_I_d['Treatment']=='LL']
            A_I_d = A_I_d[A_I_d['Response']=='AI']
            ALL_I = A_I_d['gs'].values
            Iinc = A_I_d['Iinc'].values
            err = A_I_d['Std.dev gs'].values
            
            ax[1][1].errorbar(Iinc, ALL_I,err,fmt=symbol[count],label=plant,markersize=8)
            count+=1
        
        ax[0][0].set_ylabel("Stomatal conductance (mol $m^{-2}$ $s^{-1}$)")
#        ax[0][1].set_ylabel("Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)")       
        ax[1][0].set_ylabel("Stomatal conductance (mol $m^{-2}$ $s^{-1}$)")
#        ax[1][1].set_ylabel("Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)")
        ax[0][0].set_xlabel("Intercellular $CO_2$ (µmol $mol^{-1}$)")
        ax[1][0].set_xlabel("Irradiance (µmol $mol^{-1}$)")   
        ax[0][1].set_xlabel("Intercellular $CO_2$ (µmol $mol^{-1}$)")        
        ax[1][1].set_xlabel("Irradiance (µmol $mol^{-1}$)")
        ax[0][1].set_ylim(top=1.1)
        ax[1][1].set_ylim(top=1.1)
        ax[1][1].legend(loc='lower right', fontsize='x-large')            
        

    def compare_PhiPSII(self,df):
        fig, ax = plt.subplots(2,2,constrained_layout=True)
        plt.rcParams["figure.figsize"] = (10,10)
        plants = ['Hi','BN']
        count = 0
        symbol=['ko','k<']
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
            ax[0][0].errorbar(Ci,AHL_CI,err,fmt=symbol[count],markersize=8)
            
            AHL_I = A_I_d['PhiPS2'].values
            Iinc = A_I_d['Iinc'].values
            err = A_I_d['Std.dev PhiPS2'].values
            ax[1][0].errorbar(Iinc,AHL_I,err,fmt=symbol[count],markersize=8)
            
                 
            A_CI_d = df[df['Species']==plant]
            A_CI_d = A_CI_d[A_CI_d['Treatment']=='LL']
            A_CI_d = A_CI_d[A_CI_d['Response']=='ACI']
            A_I_d = A_CI_d[A_CI_d['Response']=='AI']
            
            ALL_CI = A_CI_d['PhiPS2'].values
            Ci = A_CI_d['Ci'].values
            err = A_CI_d['Std.dev PhiPS2'].values
            
            ax[0][1].errorbar(Ci,ALL_CI,err,fmt=symbol[count],markersize=8)
            
            A_I_d = df[df['Species']==plant]
            A_I_d = A_I_d[A_I_d['Treatment']=='LL']
            A_I_d = A_I_d[A_I_d['Response']=='AI']
            ALL_I = A_I_d['PhiPS2'].values
            Iinc = A_I_d['Iinc'].values
            err = A_I_d['Std.dev PhiPS2'].values
            
            ax[1][1].errorbar(Iinc, ALL_I,err,fmt=symbol[count],label=plant,markersize=8)
            count+=1
        
        ax[0][0].set_ylabel("\u03A6$_{PSII}$ (-)")
        ax[1][0].set_ylabel("\u03A6$_{PSII}$ (-)")
        ax[0][0].set_xlabel("Intercellular $CO_2$ (µmol $mol^{-1}$)")
        ax[1][0].set_xlabel("Irradiance (µmol $mol^{-1}$)")   
        ax[0][1].set_xlabel("Intercellular $CO_2$ (µmol $mol^{-1}$)")        
        ax[1][1].set_xlabel("Irradiance (µmol $mol^{-1}$)")
        ax[0][1].set_ylim(top=0.50)
        ax[1][1].set_ylim(top=1)
        ax[1][0].set_ylim(top=1)
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
        
        for Par in PARi:
            A_I_r_BN= A_I_d_BN[A_I_d_BN['Irradiance']==Par]
            A_I_r_Hi= A_I_d_Hi[A_I_d_Hi['Irradiance']==Par]
            AHi = A_I_r_Hi['Net CO2 assimilation rate'].values
            ABN = A_I_r_BN['Net CO2 assimilation rate'].values
            [t,p]= stats.ttest_ind(AHi,ABN, equal_var = False)
            p_values.append(p)
        return p_values
             
    def t_test_A_CI(self,treatment):
        A_CI_d_BN  = self.A_CI[self.A_CI['Species']=='B.Nigra']
        A_CI_d_BN  = A_CI_d_BN[A_CI_d_BN['Oxygen level']==self.get_O2()]
        A_CI_d_BN = A_CI_d_BN[A_CI_d_BN['Treatment']==treatment]
        
        A_CI_d_Hi  = self.A_CI[self.A_CI['Species']=='H.Incana']        
        A_CI_d_Hi  = A_CI_d_Hi[A_CI_d_Hi['Oxygen level']==self.get_O2()]
        A_CI_d_Hi = A_CI_d_Hi[A_CI_d_Hi['Treatment']==treatment]
        p_values = []
        par_values = [400,300,250,200,150,100,500,600,700,850,1000,1200,1500,1800,2200]*4
        A_CI_d_BN['Irradiance']=par_values
        A_CI_d_Hi['Irradiance']=par_values
        PARi = A_CI_d_BN['Irradiance'].values
        PARi=np.unique(PARi)
        
        for Par in PARi:
            A_CI_r_BN= A_CI_d_BN[A_CI_d_BN['Irradiance']==Par]
            A_CI_r_Hi= A_CI_d_Hi[A_CI_d_Hi['Irradiance']==Par]
            AHi = A_CI_r_Hi['Net CO2 assimilation rate'].values
            ABN = A_CI_r_BN['Net CO2 assimilation rate'].values
            [t,p]= stats.ttest_ind(AHi,ABN, equal_var = False)
            p_values.append(p)
        return p_values
             