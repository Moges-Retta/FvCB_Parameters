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
    data = pd.read_excel ('Gas_Exchange_data_leak_corr.xlsx') 
    
    # FORMAT = ['Replicate','Species','Treatment','Measurement_type',\
    #           'Oxygen_level','Net_CO2_assimilation_rate', \
    #         'Intercellular_CO2_concentration', 'PhiPS2','Irradiance',\
    #     'Stomatal_conductance_for_CO2','CO2S','CO2R','BLCond',\
    #     'VpdL','H2OR','H2OS','Flow','Area','Trmmol']
    FORMAT = ['Replicate','Species','Treatment','Measurement_type','Oxygen_level',\
              'Net_CO2_assimilation_rate', 'Intercellular_CO2_concentration', \
                  'PhiPS2','Irradiance','Stomatal_conductance_for_CO2','CO2R']
    df_selected = data[FORMAT]
    A_CI = df_selected.query('Measurement_type=="A_CI_curve"')
    A_I = df_selected.query('Measurement_type=="A_I_curve"')

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

    def make_avareage_data(self):
        columns = ['Species','Treatment','Response','Ci','A','Iinc','PhiPS2','Std.dev A','gs','Std.dev gs','Std.dev PhiPS2']
        species = ['B.Nigra','H.Incana']
        treatments = ['HL','LL']
        ave_gas_Exchange_data = pd.DataFrame([])
        df = pd.DataFrame([],columns=columns )
        O = 0.21
        
        # Make average data of ACI and AI
        for plant in species:
                for treatment in treatments:
                    gas_exch_measurement = Gas_exchange_measurement(O,plant,treatment)
                    df_ave = gas_exch_measurement.average_A_CI()
                    
                    I_ave_ci = df_ave['Irradiance'].values
                    Ci_ave_ci = df_ave['Intercellular_CO2_concentration'].values
                    A_ave_ci = df_ave['Net_CO2_assimilation_rate'].values
                    gs_ave_ci = df_ave['Stomatal_conductance_for_CO2'].values
                    phiPS2 = df_ave['PhiPS2'].values
                    A_std = df_ave['Photo_err'].values
                    gs_std = df_ave['gs_err'].values
                    phiPS2_std = df_ave['PhiPS2_err'].values
                    
                    df = pd.DataFrame([],columns=columns )            
                    df['Ci']=Ci_ave_ci; df['A']=A_ave_ci; df['Iinc']=I_ave_ci; 
                    df['Std.dev A']=A_std; df['gs']=gs_ave_ci;df['PhiPS2']=phiPS2;df['Std.dev PhiPS2']=phiPS2_std;
                    df['Std.dev gs']=gs_std;df['Species']=plant; df['Treatment']=treatment; df['Response']='ACI'; 
                    ave_gas_Exchange_data=ave_gas_Exchange_data.append(df)
                    
                    df_ave  = gas_exch_measurement.average_A_I() 
                    I_ave_i = df_ave['Irradiance'].values
                    Ci_ave_i = df_ave['Intercellular_CO2_concentration'].values
                    A_ave_i = df_ave['Net_CO2_assimilation_rate'].values
                    gs_ave_i = df_ave['Stomatal_conductance_for_CO2'].values
                    phiPS2 = df_ave['PhiPS2'].values
                    A_std = df_ave['Photo_err'].values
                    gs_std = df_ave['gs_err'].values
                    phiPS2_std = df_ave['PhiPS2_err'].values
                    
                    
                    df = pd.DataFrame([],columns=columns )
                    df['Ci']=Ci_ave_i; df['A']=A_ave_i; df['Iinc']=I_ave_i; 
                    df['Std.dev A']=A_std; df['gs']=gs_ave_i;df['PhiPS2']=phiPS2;df['Std.dev PhiPS2']=phiPS2_std;
                    df['Std.dev gs']=gs_std;df['Species']=plant; df['Treatment']=treatment; df['Response']='AI'; 
                    ave_gas_Exchange_data=ave_gas_Exchange_data.append(df)
        return ave_gas_Exchange_data
                
    ##ave_gas_Exchange_data.to_excel(PATH + 'Ave_Gas_Exchange_data_corr.xlsx', index = False)


    def get_average_values(self,curve):
        if curve == 'ACI':
            df_ave = self.average_A_CI()    
        else :
            df_ave = self.average_A_I()    
        
        return df_ave


    def plot_A_CI(self):
        A_CI_d = self.A_CI[self.A_CI['Oxygen_level']==self.get_O2()]
        A_CI_d = A_CI_d[A_CI_d['Species']==self.get_species()]
        A_CI_d = A_CI_d[A_CI_d['Treatment']==self.get_treatment()]
        replicates = A_CI_d['Replicate'].values
        replicates=np.unique(replicates)
        for replicate in replicates:
            A_CI_r= A_CI_d[A_CI_d['Replicate']==replicate]
            Ci = A_CI_r['Intercellular_CO2_concentration'].values
            A = A_CI_r['Net_CO2_assimilation_rate'].values
            plt.plot(Ci,A,'o',markersize=8)
            plt.xlabel('Intercellular CO$_2$ (µmol mol$^{-1}$)',fontsize=24)
            plt.ylabel('Net photosynthesis (µmol m$^{-2}$ s$^{-1}$)',fontsize=24)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
        plt.show()


    def get_AI_data(self):
        A_I_d  = self.A_I[self.A_I['Oxygen_level']==self.get_O2()]
        A_I_d =  A_I_d[A_I_d['Species']==self.get_species()]
        A_I_d = A_I_d[A_I_d['Treatment']==self.get_treatment()]
        return A_I_d


    def get_ACI_data(self):
        A_CI = self.get_A_Ci()
        A_CI_d  = A_CI[A_CI['Oxygen_level']==self.get_O2()]
        A_CI_d =  A_CI_d[A_CI_d['Species']==self.get_species()]
        A_CI_d = A_CI_d[A_CI_d['Treatment']==self.get_treatment()]
        return A_CI_d

            
    def plot_A_I(self):
        A_I_d  = self.A_I[self.A_I['Oxygen_level']==self.get_O2()]
        A_I_d =  A_I_d[A_I_d['Species']==self.get_species()]
        A_I_d = A_I_d[A_I_d['Treatment']==self.get_treatment()]
        replicates = A_I_d['Replicate'].values
        replicates=np.unique(replicates)
        for replicate in replicates:
            A_I_r= A_I_d[A_I_d['Replicate']==replicate]
            I = A_I_r['Irradiance'].values
            A = A_I_r['Net_CO2_assimilation_rate'].values
            plt.plot(I,A,'o',markersize=8)
            plt.xlabel('Irradiance (µmol m$^{-2}$ s$^{-1}$)',fontsize=24)
            plt.ylabel('Net photosynthesis (µmol m$^{-2}$ s$^{-1}$)',fontsize=24)
            plt.title("A-I")
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
        plt.show()


    def average_A_CI(self):
        A_CI_d = self.A_CI[self.A_CI['Oxygen_level']==self.get_O2()]
        A_CI_d = A_CI_d[A_CI_d['Species']==self.get_species()]
        A_CI_d = A_CI_d[A_CI_d['Treatment']==self.get_treatment()]
        replicates = A_CI_d['Replicate'].unique()

        cols = ['Irradiance','Intercellular_CO2_concentration','Net_CO2_assimilation_rate',\
                'PhiPS2','Stomatal_conductance_for_CO2','Photo_err','gs_err','PhiPS2_err']
        df_ave = pd.DataFrame([],columns = cols)
        df_ci = pd.DataFrame([])
        df_A = pd.DataFrame([])
        df_I = pd.DataFrame([])
        df_gs = pd.DataFrame([])
        df_phi = pd.DataFrame([])
        count = 0

        for replicate in replicates:
            A_CI_r= A_CI_d[A_CI_d['Replicate']==replicate]
            Ci = A_CI_r['Intercellular_CO2_concentration'].values
            A = A_CI_r['Net_CO2_assimilation_rate'].values
            I = A_CI_r['Irradiance'].values
            gs = A_CI_r['Stomatal_conductance_for_CO2'].values
            PhiPS2 = A_CI_r['PhiPS2'].values
            
            df_ci.loc[:,count] = Ci
            df_A.loc[:,count] = A
            df_I.loc[:,count] = I
            df_gs.loc[:,count] = gs
            df_phi.loc[:,count] = PhiPS2

            count+=1
        
        df_ave.loc[:,'Irradiance'] = np.nanmean(df_I,axis=1)
        df_ave.loc[:,'Intercellular_CO2_concentration'] = np.nanmean(df_ci,axis=1)
        df_ave.loc[:,'Net_CO2_assimilation_rate'] = np.nanmean(df_A,axis=1)
        df_ave.loc[:,'Stomatal_conductance_for_CO2'] = np.nanmean(df_gs,axis=1)
        df_ave.loc[:,'PhiPS2'] = np.nanmean(df_phi,axis=1)
        df_ave.loc[:,'Photo_err'] = np.nanstd(df_A,axis=1)
        df_ave.loc[:,'gs_err'] = np.nanstd(df_gs,axis=1)
        df_ave.loc[:,'PhiPS2_err'] = np.nanstd(df_phi,axis=1)
        df_ave = df_ave.sort_values(by=['Intercellular_CO2_concentration'])
        return df_ave
    
    def average_A_I(self):
        A_I_d  = self.A_I[self.A_I['Oxygen_level']==self.get_O2()]
        A_I_d =  A_I_d[A_I_d['Species']==self.get_species()]
        A_I_d = A_I_d[A_I_d['Treatment']==self.get_treatment()]
        replicates = A_I_d['Replicate'].unique()

        df_ci = pd.DataFrame([])
        df_A = pd.DataFrame([])
        df_I = pd.DataFrame([])
        df_gs = pd.DataFrame([])
        df_phi = pd.DataFrame([])
        count = 0
        cols = ['Irradiance','Intercellular_CO2_concentration','Net_CO2_assimilation_rate',\
                'PhiPS2','Stomatal_conductance_for_CO2','Photo_err','gs_err','PhiPS2_err']
        df_ave = pd.DataFrame([],columns = cols)
        
        for replicate in replicates:
            A_I_r= A_I_d[A_I_d['Replicate']==replicate]
            I = A_I_r['Irradiance'].values
            A = A_I_r['Net_CO2_assimilation_rate'].values
            Ci = A_I_r['Intercellular_CO2_concentration'].values
            gs = A_I_r['Stomatal_conductance_for_CO2'].values
            PhiPS2 = A_I_r['PhiPS2'].values
            df_ci.loc[:,count] = Ci
            df_A.loc[:,count] = A
            df_I.loc[:,count] = I
            df_gs.loc[:,count] = gs
            df_phi.loc[:,count] = PhiPS2
            count+=1
            
        df_ave.loc[:,'Irradiance'] = np.nanmean(df_I,axis=1)
        df_ave.loc[:,'Intercellular_CO2_concentration'] = np.nanmean(df_ci,axis=1)
        df_ave.loc[:,'Net_CO2_assimilation_rate'] = np.nanmean(df_A,axis=1)
        df_ave.loc[:,'Stomatal_conductance_for_CO2'] = np.nanmean(df_gs,axis=1)
        df_ave.loc[:,'PhiPS2'] = np.nanmean(df_phi,axis=1)
        df_ave.loc[:,'Photo_err'] = np.nanstd(df_A,axis=1)
        df_ave.loc[:,'gs_err'] = np.nanstd(df_gs,axis=1)
        df_ave.loc[:,'PhiPS2_err'] = np.nanstd(df_phi,axis=1)        
        df_ave = df_ave.sort_values(by=['Irradiance'])            
        return df_ave
        
    def plot_ave_A_CI(self,df_ave):
        
        Ci_ave_ci = df_ave['Intercellular_CO2_concentration'].values
        A_ave_ci = df_ave['Net_CO2_assimilation_rate'].values
        A_std = df_ave['Photo_err'].values
        plt.errorbar(Ci_ave_ci,A_ave_ci,A_std,fmt='o',markersize=8)
        plt.xlabel('Intercellular CO$_2$ (µmol mol$^{-1}$)',fontsize=24)
        plt.ylabel('Net photosynthesis (µmol m$^{-2}$ s$^{-1}$)',fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)        
        plt.show()



    def plot_ave_A_I(self,df_ave):
        I_ave = df_ave['Irradiance'].values
        A_ave = df_ave['Net_CO2_assimilation_rate'].values
        A_std = df_ave['Photo_err'].values

        plt.errorbar(I_ave,A_ave,A_std,fmt='o')
        plt.xlabel('Irradiance (µmol $m^{-2}$ $s^{-1}$)')
        plt.ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)        
        plt.show()


    def plot_ave_gs_I(self,df_ave):
        gs_ave = df_ave['Stomatal_conductance_for_CO2'].values
        gs_std = df_ave['gs_err'].values
        I_ave = df_ave['Irradiance'].values

        plt.errorbar(I_ave,gs_ave,gs_std,fmt='o')
        plt.xlabel('Irradiance (µmol $m^{-2}$ $s^{-1}$)')
        plt.ylabel('Stomatal conductance (mol $m^{-2}$ $s^{-1}$)')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)        
        plt.show()        


    def plot_ave_gs_CI(self,df_ave):
        gs_ave = df_ave['Stomatal_conductance_for_CO2'].values
        gs_std = df_ave['gs_err'].values
        Ci_ave_ci = df_ave['Intercellular_CO2_concentration'].values

        plt.errorbar(Ci_ave_ci,gs_ave,gs_std,fmt='o')
        plt.xlabel('Intercellular $CO_2$ (µmol $mol^{-1}$)')
        plt.ylabel('Stomatal conductance (mol $m^{-2}$ $s^{-1}$)')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)        
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
        A_I_d_BN  = A_I_d_BN[A_I_d_BN['Oxygen_level']==self.get_O2()]
        A_I_d_BN = A_I_d_BN[A_I_d_BN['Treatment']==treatment]
        
        A_I_d_Hi  = self.A_I[self.A_I['Species']=='H.Incana']        
        A_I_d_Hi  = A_I_d_Hi[A_I_d_Hi['Oxygen_level']==self.get_O2()]
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
            AHi = A_I_r_Hi['Net_CO2_assimilation_rate'].values
            ABN = A_I_r_BN['Net_CO2_assimilation_rate'].values
            [t,p]= stats.ttest_ind(AHi,ABN, equal_var = False)
            p_values.loc[count,'p_A']=p
           
            gsHi = A_I_r_Hi['Stomatal_conductance_for_CO2'].values
            gsBN = A_I_r_BN['Stomatal_conductance_for_CO2'].values
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
        A_CI_d_BN  = A_CI_d_BN[A_CI_d_BN['Oxygen_level']==self.get_O2()]
        A_CI_d_BN = A_CI_d_BN[A_CI_d_BN['Treatment']==treatment]
        
        A_CI_d_Hi  = self.A_CI[self.A_CI['Species']=='H.Incana']        
        A_CI_d_Hi  = A_CI_d_Hi[A_CI_d_Hi['Oxygen_level']==self.get_O2()]
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
            AHi = A_CI_r_Hi['Net_CO2_assimilation_rate'].values
            ABN = A_CI_r_BN['Net_CO2_assimilation_rate'].values
            [t,p]= stats.ttest_ind(AHi,ABN, equal_var = False)
            p_values.loc[count,'p_A']=p
            gsHi = A_CI_r_Hi['Stomatal_conductance_for_CO2'].values
            gsBN = A_CI_r_BN['Stomatal_conductance_for_CO2'].values
            [t,p]= stats.ttest_ind(gsHi,gsBN, equal_var = False)
            p_values.loc[count,'p_gs']=p            
            phi_Hi = A_CI_r_Hi['PhiPS2'].values
            phi_BN = A_CI_r_BN['PhiPS2'].values
            [t,p]= stats.ttest_ind(phi_Hi,phi_BN, equal_var = False)
            p_values.loc[count,'p_phi']=p            
            count+=1
        return p_values


     
    def average_A_CI_df(self,df):
        A_CI_d = df[df['Oxygen_level']==self.get_O2()]
        A_CI_d = A_CI_d[A_CI_d['Species']==self.get_species()]
        A_CI_d = A_CI_d[A_CI_d['Treatment']==self.get_treatment()]
        A_CI_d = A_CI_d[A_CI_d['Measurement_type']=='A_CI_curve']
        
        replicates = A_CI_d['Replicate'].unique()

        cols = ['Irradiance','Intercellular_CO2_concentration','Net_CO2_assimilation_rate',\
                'PhiPS2','Stomatal_conductance_for_CO2','Photo_err','gs_err','PhiPS2_err']
        df_ave = pd.DataFrame([],columns = cols)
        df_ci = pd.DataFrame([])
        df_A = pd.DataFrame([])
        df_I = pd.DataFrame([])
        df_gs = pd.DataFrame([])
        df_phi = pd.DataFrame([])
        count = 0

        for replicate in replicates:
            A_CI_r= A_CI_d[A_CI_d['Replicate']==replicate]
            Ci = A_CI_r['Intercellular_CO2_concentration'].values
            A = A_CI_r['Net_CO2_assimilation_rate'].values
            I = A_CI_r['Irradiance'].values
            gs = A_CI_r['Stomatal_conductance_for_CO2'].values
            PhiPS2 = A_CI_r['PhiPS2'].values
            
            df_ci.loc[:,count] = Ci
            df_A.loc[:,count] = A
            df_I.loc[:,count] = I
            df_gs.loc[:,count] = gs
            df_phi.loc[:,count] = PhiPS2

            count+=1
        
        df_ave.loc[:,'Irradiance'] = np.nanmean(df_I,axis=1)
        df_ave.loc[:,'Intercellular_CO2_concentration'] = np.nanmean(df_ci,axis=1)
        df_ave.loc[:,'Net_CO2_assimilation_rate'] = np.nanmean(df_A,axis=1)
        df_ave.loc[:,'Stomatal_conductance_for_CO2'] = np.nanmean(df_gs,axis=1)
        df_ave.loc[:,'PhiPS2'] = np.nanmean(df_phi,axis=1)
        df_ave.loc[:,'Photo_err'] = np.nanstd(df_A,axis=1)
        df_ave.loc[:,'gs_err'] = np.nanstd(df_gs,axis=1)
        df_ave.loc[:,'PhiPS2_err'] = np.nanstd(df_phi,axis=1)
        df_ave = df_ave.sort_values(by=['Intercellular_CO2_concentration'])
        return df_ave
    
    def average_A_I_df(self,df):
        A_I_d  = df[df['Oxygen_level']==self.get_O2()]
        A_I_d =  A_I_d[A_I_d['Species']==self.get_species()]
        A_I_d = A_I_d[A_I_d['Treatment']==self.get_treatment()]
        A_I_d = A_I_d[A_I_d['Measurement_type']=='A_I_curve']

        replicates = A_I_d['Replicate'].unique()

        df_ci = pd.DataFrame([])
        df_A = pd.DataFrame([])
        df_I = pd.DataFrame([])
        df_gs = pd.DataFrame([])
        df_phi = pd.DataFrame([])
        count = 0
        cols = ['Irradiance','Intercellular_CO2_concentration','Net_CO2_assimilation_rate',\
                'PhiPS2','Stomatal_conductance_for_CO2','Photo_err','gs_err','PhiPS2_err']
        df_ave = pd.DataFrame([],columns = cols)
        
        for replicate in replicates:
            A_I_r= A_I_d[A_I_d['Replicate']==replicate]
            I = A_I_r['Irradiance'].values
            A = A_I_r['Net_CO2_assimilation_rate'].values
            Ci = A_I_r['Intercellular_CO2_concentration'].values
            gs = A_I_r['Stomatal_conductance_for_CO2'].values
            PhiPS2 = A_I_r['PhiPS2'].values
            df_ci.loc[:,count] = Ci
            df_A.loc[:,count] = A
            df_I.loc[:,count] = I
            df_gs.loc[:,count] = gs
            df_phi.loc[:,count] = PhiPS2
            count+=1
            
        df_ave.loc[:,'Irradiance'] = np.nanmean(df_I,axis=1)
        df_ave.loc[:,'Intercellular_CO2_concentration'] = np.nanmean(df_ci,axis=1)
        df_ave.loc[:,'Net_CO2_assimilation_rate'] = np.nanmean(df_A,axis=1)
        df_ave.loc[:,'Stomatal_conductance_for_CO2'] = np.nanmean(df_gs,axis=1)
        df_ave.loc[:,'PhiPS2'] = np.nanmean(df_phi,axis=1)
        df_ave.loc[:,'Photo_err'] = np.nanstd(df_A,axis=1)
        df_ave.loc[:,'gs_err'] = np.nanstd(df_gs,axis=1)
        df_ave.loc[:,'PhiPS2_err'] = np.nanstd(df_phi,axis=1)        
        df_ave = df_ave.sort_values(by=['Irradiance'])            
        return df_ave        
        
      

    def compare_leak_corr(self,Gas_Exchange_data_corr):

        CO2R = [100,150,200,250,300,400,500,600,700,850,1000,1200,1500,1800,2000]
        
        df_ave_ACI_corr=self.average_A_CI_df(Gas_Exchange_data_corr)
        df_ave_AI_corr=self.average_A_I_df(Gas_Exchange_data_corr)

        df_ave_ACI=self.average_A_CI()
        df_ave_AI=self.average_A_I()
        
        Ci_ave_ci_corr = df_ave_ACI_corr['Intercellular_CO2_concentration'].values
        A_ave_ci_corr = df_ave_ACI_corr['Net_CO2_assimilation_rate'].values
        
        Ci_ave_ci = df_ave_ACI['Intercellular_CO2_concentration'].values
        A_ave_ci = df_ave_ACI['Net_CO2_assimilation_rate'].values
        # A_std_ci = df_ave_ACI['Photo_err'].values
        
        i_ave_i_corr = df_ave_AI_corr['Irradiance'].values
        A_ave_i_corr = df_ave_AI_corr['Net_CO2_assimilation_rate'].values
        Ci_ave_i_corr = df_ave_AI_corr['Intercellular_CO2_concentration'].values

        i_ave_i = df_ave_AI['Irradiance'].values
        A_ave_i = df_ave_AI['Net_CO2_assimilation_rate'].values
        Ci_ave_i = df_ave_AI['Intercellular_CO2_concentration'].values
        # A_std_i = df_ave_AI['Photo_err'].values
        
        fig, ax = plt.subplots(2,2,constrained_layout=True)
        plt.rcParams["figure.figsize"] = (20,20)
                
        ax[0][0].plot(Ci_ave_ci,A_ave_ci,'r-',linewidth=2,label='Raw')
        ax[0][0].plot(Ci_ave_ci_corr,A_ave_ci_corr,'k-',linewidth=2,label='Corrected')
        ax[0][0].set_xlabel('Intercellular CO$_2$ (µmol mol$^{-1}$)',fontsize=24)
        ax[0][0].set_ylabel('Net photosynthesis(µmol $m^{-2}$ s$^{-1}$ )',fontsize=24)
        # ax[0][0].xaxis.set_ticks(np.arange(0, 2200, 400))        
        
        ax[0][1].plot(i_ave_i,A_ave_i,'r-',linewidth=2,label='Raw')
        ax[0][1].plot(i_ave_i_corr,A_ave_i_corr,'k-',linewidth=2,label='Corrected')
        # ax1.plot(i_inc_i,Ap_i,'g-',linewidth=2,label='Ap')
        
        ax[0][1].set_xlabel('Irradiance (µmol m$^{-2}$ s$^{-1}$)',fontsize=24)
        ax[0][1].set_ylabel('Net photosynthesis (µmol $m^{-2}$ s$^{-1}$ )',fontsize=24)   
        
        ax[1][0].plot(CO2R,Ci_ave_ci,'r-',linewidth=2,label='Raw')
        ax[1][0].plot(CO2R,Ci_ave_ci_corr,'k-',linewidth=2,label='Corrected')
        ax[1][0].set_ylabel('Intercellular CO$_2$ (µmol mol$^{-1}$)',fontsize=24)
        ax[1][0].set_xlabel(' CO2R (µmol mol$^{-1}$)',fontsize=24)
        # ax[0][0].xaxis.set_ticks(np.arange(0, 2200, 400))        
        
        ax[1][1].plot(i_ave_i,Ci_ave_i,'r-',linewidth=2,label='Raw')
        ax[1][1].plot(i_ave_i_corr,Ci_ave_i_corr,'k-',linewidth=2,label='Corrected')
        # ax1.plot(i_inc_i,Ap_i,'g-',linewidth=2,label='Ap')
        
        ax[1][1].set_xlabel('Irradiance (µmol m$^{-2}$ s$^{-1}$)',fontsize=24)
        ax[1][1].set_ylabel('Intercellular CO$_2$ (µmol mol$^{-1}$)',fontsize=24)    
        
        ax[1][1].legend(loc='upper right', fontsize=32)     
        # ax2.xaxis.set_ticks(np.arange(0, 2200, 400)) 


    def leak_model(self,species,treatment,CO2R):
        
        """ returns species and treatment specific model for the 
        relationship between CO2R-CO2s vs CO2s"""
        
        if species=='B.Nigra' and treatment=='HL':          
            model_delta_co2 = 0.0065*CO2R - 0.1425 # correct for CO2s, BnHL
        elif species=='B.Nigra' and treatment=='LL':  
            model_delta_co2 = 0.0065*CO2R - 0.1425 # correct for CO2s, BnHL   
        elif species=='H.Incana' and treatment=='HL': 
            model_delta_co2 = 0.0044*CO2R - 0.5706 # correct for CO2s, HiHL
        else:
            model_delta_co2 = 0.0044*CO2R - 0.5706 # correct for CO2s, HiHL
        return model_delta_co2 
                           
    
    def correct_leak(self,species,treatment):
        """
        Correct for leakage
        1. correction for CO2s based on linear relationship between CO2R-CO2S and
        CO2R
        2. correct CO2s for main measurement using the above model
        3. recalculate gbc and Ci and photo  
        """
        columns = ['Replicate','Species','Treatment','Measurement type',\
                   'Oxygen_level','Net_CO2_assimilation_rate',\
                   'Intercellular_CO2_concentration','PhiPS2','Irradiance',\
                   'Stomatal_conductance_for_CO2','CO2R','CO2S','H2OR','H2OS',\
                   'Flow','Area','Trmmol','BLCond']
        Gas_Exchange_data_corr = pd.DataFrame([],columns=columns )
        ACI_corr = self.A_CI;
        ACI_corr=ACI_corr[ACI_corr['Species']==species];
        ACI_corr=ACI_corr[ACI_corr['Treatment']==treatment];
        AI_corr = self.A_I;
        AI_corr=AI_corr[AI_corr['Species']==species];
        AI_corr=AI_corr[AI_corr['Treatment']==treatment];
        
        
        cond = ACI_corr['Stomatal_conductance_for_CO2'].values
        blcond = ACI_corr['BLCond'].values
        trmmol = ACI_corr['Trmmol'].values        
        CO2S = ACI_corr['CO2S'].values
        CO2R = ACI_corr['CO2R'].values
        H2OR = ACI_corr['H2OR'].values
        H2OS = ACI_corr['H2OS'].values
        flow = ACI_corr['Flow'].values
        area = ACI_corr['Area'].values

        model_delta_co2 = self.leak_model(species,treatment,CO2R) # correct for CO2s, HiHL
 
        co2s_corrected = CO2S+model_delta_co2
        a = (1000-H2OR)/(1000-H2OS)        
        photo_corr = flow*(CO2R-co2s_corrected*a)/(100*area)           
        gbl_corr = 1/(1.6/cond+1.37/blcond)
        ci_corr = ((gbl_corr-trmmol/2000)*co2s_corrected-photo_corr)/(gbl_corr+trmmol/2000)
        
        ACI_corr.loc[:,'Net_CO2_assimilation_rate']=photo_corr
        ACI_corr.loc[:,'Intercellular_CO2_concentration']=ci_corr        
        ACI_corr.loc[:,'Measurement_type']=ACI_corr['Measurement_type'].values        
        Gas_Exchange_data_corr= Gas_Exchange_data_corr.append(ACI_corr)
        
        cond = AI_corr['Stomatal_conductance_for_CO2'].values
        blcond = AI_corr['BLCond'].values
        trmmol = AI_corr['Trmmol'].values        
        CO2S = AI_corr['CO2S'].values
        CO2R = AI_corr['CO2R'].values
        H2OR = AI_corr['H2OR'].values
        H2OS = AI_corr['H2OS'].values
        flow = AI_corr['Flow'].values
        area = AI_corr['Area'].values
        
        # model_delta_co2 = 0.0065*CO2R - 0.1425 # correct for CO2s
        model_delta_co2 = self.leak_model(species,treatment,CO2R) # correct for CO2s, HiHL
        
        co2s_corrected = CO2S+model_delta_co2
        a = (1000-H2OR)/(1000-H2OS)        
        photo_corr = flow*(CO2R-co2s_corrected*a)/(100*area)           
        gbl_corr = 1/(1.6/cond+1.37/blcond)
        ci_corr = ((gbl_corr-trmmol/2000)*co2s_corrected-photo_corr)/(gbl_corr+trmmol/2000)
        
        AI_corr.loc[:,'Net_CO2_assimilation_rate']=photo_corr
        AI_corr.loc[:,'Intercellular_CO2_concentration']=ci_corr     
        AI_corr.loc[:,'Measurement_type']=AI_corr['Measurement_type'].values        
        
        Gas_Exchange_data_corr=Gas_Exchange_data_corr.append(AI_corr)
        
        FORMAT = ['Replicate','Species','Treatment','Measurement_type',\
                  'Oxygen_level','Net_CO2_assimilation_rate', \
                      'Intercellular_CO2_concentration', 'PhiPS2',\
                          'Irradiance','Stomatal_conductance_for_CO2','CO2R']
        Gas_Exchange_data_corr = Gas_Exchange_data_corr[FORMAT]
        
        self.compare_leak_corr(Gas_Exchange_data_corr)
        # Gas_Exchange_data_corr.to_excel(PATH + 'Gas_Exchange_data_leak_corr_'+species+'_'+treatment+'.xlsx', index = False)
        
        return Gas_Exchange_data_corr
        

   