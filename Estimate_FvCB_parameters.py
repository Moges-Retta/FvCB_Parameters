# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:03:59 2021

@author: Moges Retta
Estimate parameters of photosynthesis kinectics based on Yin and Struik 2009
Plant, cell and Environment, 2009,32(5) pp.448-64
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy import optimize

class Estimate_FvCB_parameters:
    def __init__(self,gas_exch_measurement):
        self.gas_exch_measurement=gas_exch_measurement
        

    def plot_Rd(self,Rds,points):
            fig, ax = plt.subplots(2,2,constrained_layout=True)
            replicates = Rds['Replicate'].values
            plt.rcParams["figure.figsize"] = (10,10)
            count = 0
            for replicate in replicates:
                Rd = Rds['Rd'].values
                slope = Rds['Slope'].values

                X = points['X'][count]
                Y = points['Y'][count]
                R2 = Rds['R2'].values
                ypredict = slope[count]*X+Rd[count]
                
                if Rd[count]>0:
                    rd_text = ' + ' + str(np.round(Rd[count],2))                    
                else:
                    rd_text = ' - ' + str(np.round(abs(Rd[count]),2))
                text = 'y = ' + str(np.round(slope[count],3)) + '*x '+\
                rd_text+' \n R2 = '+str(np.round(R2[count],3))
                
                if replicate == 1:
                    ax[0][0].plot(X, Y, 'ko')           
                    ax[0][0].plot(X, ypredict, color='black', linewidth=3) 
                    ax[0][0].set_ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)')
                    ax[0][0].text(min(X)+1,max(Y)-3,text)
                    
                elif replicate==2:
                    ax[0][1].plot(X, Y, 'ko')           
                    ax[0][1].plot(X, ypredict, color='black', linewidth=3)
                    ax[0][1].text(min(X)+1,max(Y)-3,text)
                     
                elif replicate==3:
                    ax[1][0].plot(X, Y, 'ko')         
                    ax[1][0].plot(X, ypredict, color='black', linewidth=3)
                    ax[1][0].set_xlabel('\u03A6$_{PSII}$ * $I_{inc}$/4 (µmol $m^{-2}$ $s^{-1}$)') 
                    ax[1][0].set_ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)')
                    ax[1][0].text(min(X)+1,max(Y)-3,text)
                    
                else:
                    ax[1][1].plot(X, Y, 'ko')           
                    ax[1][1].plot(X, ypredict, color='black', linewidth=3) 
                    ax[1][1].set_xlabel('\u03A6$_{PSII}$ * $I_{inc}$/4 (µmol $m^{-2}$ $s^{-1}$)')
                    ax[1][1].text(min(X)+1,max(Y)-3,text)
                count+=1           
            plt.show()
            
            
    def anova_test_treatments(self,df_params):
        species = df_params['Species'].values
        species=np.unique(species)
        treatments=df_params['Treatment'].values
        treatments=np.unique(treatments)
        p_values = pd.DataFrame([], columns=['Species','p value'])
        count = 0
        for plant in species:
            data_rd  = df_params[df_params['Species']==plant]
            data_rd_1  = data_rd[data_rd['Treatment']==treatments[0]]  
            data_rd_2  = data_rd[data_rd['Treatment']==treatments[1]]                       
            Rds_1 = data_rd_1['Rd'].values
            Rds_2 = data_rd_2['Rd'].values                
            [t,p]= stats.ttest_ind(Rds_1,Rds_2, equal_var = False)
            p_values.loc[count,'Species']=plant
            p_values.loc[count,'p value']=p  
            count+=1
        return p_values
            

    def anova_test_species(self,df_params):
        species = df_params['Species'].values
        species=np.unique(species)
        treatments=df_params['Treatment'].values
        treatments=np.unique(treatments)
        p_values = pd.DataFrame([], columns=['Treatment','p value'])
        count = 0
        for treatment in treatments:
            data_rd  = df_params[df_params['Treatment']==treatment]
            data_rd_1  = data_rd[data_rd['Species']==species[0]]  
            data_rd_2  = data_rd[data_rd['Species']==species[1]]                       
            Rds_1 = data_rd_1['Rd'].values
            Rds_2 = data_rd_2['Rd'].values                
            [t,p]= stats.ttest_ind(Rds_1,Rds_2, equal_var = False)
            count+=1
            p_values.loc[count,'Treatment']=treatment            
            p_values.loc[count,'p value']=p              
        return p_values
    
# Estimate Rd and calibration factor s
    def estimate_Rd(self): 
        
        """
        Estimate Rd as intercept of linear regression between 
        A and PhiPSII*Iinc/4
        """
        
        AI = self.gas_exch_measurement.get_AI_data()
        AI=AI[AI['Irradiance']<350]
        replicates = AI['Replicate'].values
        replicates=np.unique(replicates)
        df = pd.DataFrame([],columns=['Replicate','Rd','Std.err','R2','Slope']) 
        df2 = pd.DataFrame([],columns=['Replicate','X','Y'] ) 
        
        count = 0
        for replicate in replicates:
            A_I_r= AI[AI['Replicate']==replicate]
            I = A_I_r['Irradiance'].values            
            PhiPS2 = A_I_r['PhiPS2'].values
            A = A_I_r['Net CO2 assimilation rate'].values
            X = PhiPS2*I/4
            result = stats.linregress(X, A) #slope, intercept, r, p, se 
            df.loc[count,'Rd'] = result.intercept
#            df.loc[count,'Std.err Rd'] = result.intercept_stderr
            df.loc[count,'R2'] = result.rvalue**2
            df.loc[count,'Replicate'] = replicate
            df.loc[count,'Slope'] = result.slope
            df.loc[count,'Std.err'] = result.stderr            
            df2.loc[count,'X'] = X
            df2.loc[count,'Y'] = A
            df2.loc[count,'Replicate'] = replicate
            count+=1
        self.plot_Rd(df,df2)
        return df    
   
         
# Estimate Jmax
    def model_individual_Jmax(self,params,s,PHI2LL,Iinc,phiPS2): 
        
        """
        Estimate Jmax, curvature factor (theta) and quantum efficiency of PSII
        e flow at strictly limiting light level (Phi2LL)
        """
        Jmax,theta = params
        k2LL = s*PHI2LL
        J = s*Iinc*phiPS2
        SQ = ((k2LL*Iinc+Jmax)**2-4*theta*k2LL*Jmax*Iinc)**0.5
        J_mod = (k2LL*Iinc+Jmax-SQ)/(2*theta)            
        return J_mod-J 
    
    def plot_Jmax_fit_individual(self,params,s,PHI2LL,Iinc,phiPS2): 
        plt.rcParams["figure.figsize"] = (10,10)        
        Jmax,theta = params
        k2LL = s*PHI2LL
        J = s*Iinc*phiPS2
        SQ = ((k2LL*Iinc+Jmax)**2-4*theta*k2LL*Jmax*Iinc)**0.5
        J_mod = (k2LL*Iinc+Jmax-SQ)/(2*theta) 
        fig, ax = plt.subplots()
        ax.plot(Iinc,J,'ko',label='CF')         
        plt.plot(Iinc,J_mod,'k-',linewidth=2.0,label='Model')  
        ax.set_xlabel('Irradiance (µmol $m^{-2}$ $s^{-1}$)')
        ax.set_ylabel('ATP production (µmol $m^{-2}$ $s^{-1}$)')  
        ax.legend(loc='lower right', fontsize='x-large')  
        
        
# Estimate phi2LL
    def model_phi2LL_individual(self,params,Iinc,phiPS2): 
        
        """
        Estimate the quantum efficiency of PSII
        e flow at strictly limiting light level (Phi2LL)
        """
#        AI = self.gas_exch_measurement.get_AI_data()
        PHI2LL,theta2,J2max = params
        phi1LL = 1
        fcyc = 0
        absor = 0.90
        Iabs = Iinc*absor
        alpha2LL = (1-fcyc)*PHI2LL/((1-fcyc)+PHI2LL/phi1LL)
        SQ = ((alpha2LL*Iabs+J2max)**2-4*theta2*alpha2LL*J2max*Iabs)**0.5 
        phiPS2_model = PHI2LL*(alpha2LL*Iabs+J2max-SQ)/(2*theta2*alpha2LL*Iabs)       
        return phiPS2-phiPS2_model

    def estimate_individual_phi2LL(self):
        AI = self.gas_exch_measurement.get_AI_data()
        replicates = AI['Replicate'].unique()
        phi2LL_values = pd.DataFrame([],columns=['Replicate','Phi2LL','Phi2LL_err','thetha2','thetha2_err','J2max','J2max_err']) #PHI2LL,theta2,J2max
        count = 0
        for replicate in replicates:
            AI_r = AI[AI['Replicate']==replicate]
            Iinc = AI_r['Irradiance'].values   
            PhiPS2 = AI_r['PhiPS2'].values            
            bnds=((0,0,0),(1,1,700)) 
            x0 = np.array([0.78,0.7,150])       #phi2LL,thetha,J2max         
            result = optimize.least_squares(self.model_phi2LL_individual,x0,args=[Iinc,PhiPS2],method='trf',bounds=bnds)
            res = self.model_phi2LL_individual(result.x,Iinc,PhiPS2)
            J = np.array(result.jac)
            S = np.array(res).T.dot(np.array(res))
            H=2*J.T.dot(J);
            degfr=len(res)-3;
            G=np.linalg.inv(H);
            var_1=2*S*G[0,0]/degfr;
            var_2=2*S*G[1,1]/degfr;
            var_3=2*S*G[2,2]/degfr;
            var_1 = np.sqrt(var_1)
            var_2 = np.sqrt(var_2)
            var_3 = np.sqrt(var_3)
            if result.success:
                phi2LL_values.loc[count,'Replicate']=replicate
                phi2LL_values.loc[count,'Phi2LL']=result.x[0]
                phi2LL_values.loc[count,'Phi2LL_err']=var_1
                phi2LL_values.loc[count,'thetha2']=result.x[1]
                phi2LL_values.loc[count,'thetha2_err']=var_2
                phi2LL_values.loc[count,'J2max']=result.x[2]
                phi2LL_values.loc[count,'J2max_err']=var_3 
                count+=1
            else:
                raise ValueError(result.message)
                return []        
        return phi2LL_values  
        
    def estimate_individual_Jmax(self,inputs):
        AI = self.gas_exch_measurement.get_AI_data()
        replicates = AI['Replicate'].unique()
        jmax_values = pd.DataFrame([],columns=['Replicate','Jmax','Jmax_err','theta','theta_err'])
        count=0
        s_values = inputs.get('s')
        phi2_LL_values=inputs.get('phi2LL')       
        for replicate in replicates:
            AI_r = AI[AI['Replicate']==replicate]
            Iinc = AI_r['Irradiance'].values
            phiPS2 = AI_r['PhiPS2'].values               
            bnds=((0,0.5),(1000,1)) #lb,ub
            x0 = np.array([150,0.7])
            s = s_values[count]
            phi2LL= phi2_LL_values[count]
            result = optimize.least_squares(self.model_individual_Jmax,x0, \
                                args=[s,phi2LL,\
                                Iinc,phiPS2],method='trf',bounds=bnds)
            res = self.model_individual_Jmax(result.x,s,phi2LL,Iinc,phiPS2)
            J = np.array(result.jac)
            S = np.array(res).T.dot(np.array(res))
            H=2*J.T.dot(J);
            degfr=len(res)-2;
            G=np.linalg.inv(H);
            var_1=2*S*G[0,0]/degfr;
            var_2=2*S*G[1,1]/degfr;
            var_1 = np.sqrt(var_1)
            var_2 = np.sqrt(var_2)
            if result.success:
                self.plot_Jmax_fit_individual(result.x,s,phi2LL,Iinc,phiPS2)
                jmax_values.loc[count,'Replicate']=replicate
                jmax_values.loc[count,'Jmax']=result.x[0]
                jmax_values.loc[count,'Jmax_err']=var_1
                jmax_values.loc[count,'theta']=result.x[1]
                jmax_values.loc[count,'theta_err']=var_2
                count+=1
            else:
                raise ValueError(result.message)
                return []
        return jmax_values

# Estimate Jmax
    def model_Jmax(self,params,s,PHI2LL): 
        
        """
        Estimate Jmax, curvature factor (theta) and quantum efficiency of PSII
        e flow at strictly limiting light level (Phi2LL)
        """
        AI = self.gas_exch_measurement.get_AI_data()
        Jmax,theta = params
        k2LL = s*PHI2LL
        Iinc = AI['Irradiance'].values
        phiPS2 = AI['PhiPS2'].values
        J = s*Iinc*phiPS2
        SQ = ((k2LL*Iinc+Jmax)**2-4*theta*k2LL*Jmax*Iinc)**0.5
        J_mod = (k2LL*Iinc+Jmax-SQ)/(2*theta)            
        return J_mod-J 
    
    def plot_Jmax_fit(self,params,s,PHI2LL): 
        plt.rcParams["figure.figsize"] = (10,10)

        Iave,Ci,A,gs,phiPS2,std_A,std_gs,std_phiPS2 = self.gas_exch_measurement.average_A_I()
        Jmax,theta = params
        k2LL = s*PHI2LL
        J = s*Iave*phiPS2
        SQ = ((k2LL*Iave+Jmax)**2-4*theta*k2LL*Jmax*Iave)**0.5
        J_mod = (k2LL*Iave+Jmax-SQ)/(2*theta) 
        fig, ax = plt.subplots()
        ax.errorbar(Iave,J,std_phiPS2*J,fmt='ko',label='CF') 
        plt.plot(Iave,J_mod,'k-',linewidth=2.0,label='Model')  
        ax.set_xlabel('Irradiance (µmol $m^{-2}$ $s^{-1}$)')
        ax.set_ylabel('ATP production (µmol $m^{-2}$ $s^{-1}$)')  
        ax.legend(loc='lower right', fontsize='x-large')  
        
        
# Estimate phi2LL
    def model_phi2LL(self,params): 
        
        """
        Estimate the quantum efficiency of PSII
        e flow at strictly limiting light level (Phi2LL)
        """
        AI = self.gas_exch_measurement.get_AI_data()
        PHI2LL,theta2,J2max = params
        phi1LL = 1
        fcyc = 0
        absor = 0.90
        Iinc = AI['Irradiance'].values
        Iabs = Iinc*absor
        phiPS2 = AI['PhiPS2'].values
        alpha2LL = (1-fcyc)*PHI2LL/((1-fcyc)+PHI2LL/phi1LL)
        SQ = ((alpha2LL*Iabs+J2max)**2-4*theta2*alpha2LL*J2max*Iabs)**0.5 
        phiPS2_model = PHI2LL*(alpha2LL*Iabs+J2max-SQ)/(2*theta2*alpha2LL*Iabs)       
        return phiPS2-phiPS2_model


    def estimate_phi2LL(self):
        bnds=((0,0,0),(1,1,700)) 
        x0 = np.array([0.78,0.7,150])       #phi2LL,thetha,J2max         
        result = optimize.least_squares(self.model_phi2LL,x0,method='trf',bounds=bnds)
        res = self.model_phi2LL(result.x)
        J = np.array(result.jac)
        S = np.array(res).T.dot(np.array(res))
        H=2*J.T.dot(J);
        degfr=len(res)-3;
        G=np.linalg.inv(H);
        var_1=2*S*G[0,0]/degfr;
        var_2=2*S*G[1,1]/degfr;
        var_3=2*S*G[2,2]/degfr;
        var_1 = np.sqrt(var_1)
        var_2 = np.sqrt(var_2)
        var_3 = np.sqrt(var_3)
   
        if result.success:
            return [result.x,var_1,var_2,var_3]
        else:
            raise ValueError(result.message)
            return []        
        
        
    def estimate_Jmax(self,inputs):
        bnds=((0,0.5),(1000,1)) #lb,ub
        x0 = np.array([150,0.7])    
        s = inputs.get('s')
        PHI2LL=inputs.get('PHI2LL')
        result = optimize.least_squares(self.model_Jmax,x0,args=[s,PHI2LL],method='trf',bounds=bnds)
        res = self.model_Jmax(result.x,s,PHI2LL)
        J = np.array(result.jac)
        S = np.array(res).T.dot(np.array(res))
        H=2*J.T.dot(J);
        degfr=len(res)-2;
        G=np.linalg.inv(H);
        var_1=2*S*G[0,0]/degfr;
        var_2=2*S*G[1,1]/degfr;
        var_1 = np.sqrt(var_1)
        var_2 = np.sqrt(var_2)
        if result.success:
            self.plot_Jmax_fit(result.x,s,PHI2LL)
            return [result.x,var_1,var_2]
        else:
            raise ValueError(result.message)
            return []

     
    def calculate_k2(self,s_values):
        AI = self.gas_exch_measurement.get_AI_data()
        replicates = AI['Replicate'].unique()
        cols = ['Replicate','Irradiance','k2']
        count=0
        k2_all=pd.DataFrame([],columns=cols)
        cols2=['r0','r1','r2','r3']
        replicates_k2=pd.DataFrame([],columns=cols2)
        for replicate in replicates:            
            df=pd.DataFrame([],columns=cols)
            A_I_r= AI[AI['Replicate']==replicate]
            I = A_I_r['Irradiance'].values            
            PhiPS2 = A_I_r['PhiPS2'].values
            k2 = s_values[count]*PhiPS2
            df.loc[:,'Replicate']=[replicate]*len(I)            
            df.loc[:,'Irradiance']=I
            df.loc[:,'k2']=k2            
            k2_all=k2_all.append(df)
            replicates_k2.loc[:,cols2[count]]=k2
            count+=1
        return replicates_k2
    
    
    def show_significant(self,p_values,k2_plant1_ave,k2_plant2_ave,k2_std1,k2_std2,axis):
        m=[]
        scale_factor=10 # to add p values to A       
        p_values = [element * scale_factor for element in p_values] 
        k2_values1 = [element  for element in k2_plant1_ave]   
        k2_values2 = [element  for element in k2_plant2_ave]                     
        par_values = [100,120,150,180,200,250,300,400,550,800,1100,1500,1800,2200]
        
        for stat in p_values:                
            if stat<0.05*scale_factor:
                m.append("*")
            else:
                m.append("")
        for i in range(len(par_values)):
            k1 = k2_values1[i]
            k2 = k2_values2[i]
            if k1>k2:               
                k2_values = k1
            else:
                k2_values = k2
            axis.plot(par_values[i]-60, k2_values, color='red',marker=m[i])
            axis.legend(loc='top right', fontsize='x-large')     
  
                  
    def compare_k2(self,k2_plant1,k2_plant2,label1,label2):
        k2_plant1_ave_d = []
        k2_plant2_ave_d = []
        k2_std1_d = []
        k2_std2_d =[]
        p_values = []
        plt.rcParams.update({'font.size': 18})
        for i in range(len(k2_plant1)):
            data1=k2_plant1.loc[i,:]
            data2=k2_plant2.loc[i,:]
            k2_plant1_ave = np.mean(k2_plant1.loc[i,:],axis=0)
            k2_plant2_ave = np.mean(k2_plant2.loc[i,:],axis=0)
            k2_std1 = np.std(k2_plant1.loc[i,:],axis=0)/2
            k2_std2 = np.std(k2_plant2.loc[i,:],axis=0)/2
            k2_plant1_ave_d.append(k2_plant1_ave) 
            k2_plant2_ave_d.append(k2_plant2_ave)       
            k2_std1_d.append(k2_std1)       
            k2_std2_d.append(k2_std2)                  
            t,p = stats.ttest_ind(data1,data2, equal_var = False)
            p_values.append(p) 
        fig, ax = plt.subplots()
        par_values = [100,120,150,180,200,250,300,400,550,800,1100,1500,1800,2200]
        ax.errorbar(par_values,k2_plant1_ave_d,k2_std1_d,fmt='ko',mfc='none',label=label1)
        ax.errorbar(par_values,k2_plant2_ave_d,k2_std2_d,fmt='k<',mfc='none',label=label2)
        ax.set_xlabel('Irradiance (µmol $m^{-2}$ $s^{-1}$)',fontsize=24)
        ax.set_ylabel('k${_2}$ (mol $e^{-}$ mol photon$^{-1}$)',fontsize=24)
        self.show_significant(p_values,k2_plant1_ave_d,k2_plant2_ave_d,k2_std1_d,k2_std2_d,ax)          
        return p_values