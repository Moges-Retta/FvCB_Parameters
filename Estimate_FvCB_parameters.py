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
from scipy import optimize, constants,stats

class Estimate_FvCB_parameters:
    params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 10),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)
    rd_cut_light = 350
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
                    ax[0][0].set_ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)',fontsize=16)
                    ax[0][0].text(min(X)+1,max(Y)-3,text)
                    
                elif replicate==2:
                    ax[0][1].plot(X, Y, 'ko')           
                    ax[0][1].plot(X, ypredict, color='black', linewidth=3)
                    ax[0][1].text(min(X)+1,max(Y)-3,text)
                     
                elif replicate==3:
                    ax[1][0].plot(X, Y, 'ko')         
                    ax[1][0].plot(X, ypredict, color='black', linewidth=3)
                    ax[1][0].set_xlabel('\u03A6$_{PSII}$ * $I_{inc}$/4 (µmol $m^{-2}$ $s^{-1}$)',fontsize=16) 
                    ax[1][0].set_ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)',fontsize=16)
                    ax[1][0].text(min(X)+1,max(Y)-3,text)
                    
                else:
                    ax[1][1].plot(X, Y, 'ko')           
                    ax[1][1].plot(X, ypredict, color='black', linewidth=3) 
                    ax[1][1].set_xlabel('\u03A6$_{PSII}$ * $I_{inc}$/4 (µmol $m^{-2}$ $s^{-1}$)',fontsize=16)
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
        AI=AI[AI['Irradiance']<self.rd_cut_light]
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
   

# Estimate Rd and calibration factor s
    def estimate_abs_adjusted_Rd(self): 
        
        """
        Estimate Rd as intercept of linear regression between 
        A and PhiPSII*Iabs/4 where Iabs = Iinc*abs, measured absorbance
        """
        
        AI = self.gas_exch_measurement.get_AI_data()
        AI=AI[AI['Irradiance']<self.rd_cut_light]
        replicates = AI['Replicate'].values
        replicates=np.unique(replicates)
        df = pd.DataFrame([],columns=['Replicate','Rd','Std.err','R2','Slope']) 
        df2 = pd.DataFrame([],columns=['Replicate','X','Y'] ) 
        absorbance = 0.90
        count = 0
        for replicate in replicates:
            A_I_r= AI[AI['Replicate']==replicate]
            I = A_I_r['Irradiance'].values            
            PhiPS2 = A_I_r['PhiPS2'].values
            A = A_I_r['Net CO2 assimilation rate'].values
            X = PhiPS2*I*absorbance/4
            result = stats.linregress(X, A) #slope, intercept, r, p, se 
            df.loc[count,'Rd'] = result.intercept
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
   

    def estimate_Rd_common(self): 
        
        """
        Estimate Rd as intercept of linear regression between 
        A and PhiPSII*Iinc/4 with data for 21 % O2 and 2 % O2 poopled together
        """
        self.gas_exch_measurement.set_O2(0.02)
        AI = self.gas_exch_measurement.get_AI_data()
        self.gas_exch_measurement.set_O2(0.21)
        AI_21O2 = self.gas_exch_measurement.get_AI_data()        
        AI=AI[AI['Irradiance']<self.rd_cut_light]
        AI_21O2=AI_21O2[AI_21O2['Irradiance']<self.rd_cut_light]
        
        replicates = AI['Replicate'].values
        replicates=np.unique(replicates)
        df = pd.DataFrame([],columns=['Replicate','Rd','Std.err','R2','Slope']) 
        df2 = pd.DataFrame([],columns=['Replicate','X','Y'] ) 
        count = 0
        for replicate in replicates:
            A_I_r= AI[AI['Replicate']==replicate]
            I = A_I_r['Irradiance'].values            
            PhiPS2 = A_I_r['PhiPS2'].values

            A_I_r_O2= AI_21O2[AI_21O2['Replicate']==replicate]
            I_O2 = A_I_r_O2['Irradiance'].values            
            PhiPS2_O2 = A_I_r_O2['PhiPS2'].values
            
            A = A_I_r['Net CO2 assimilation rate'].values
            X = PhiPS2*I/4
            
            A_O2 = A_I_r_O2['Net CO2 assimilation rate'].values
            X_O2 = PhiPS2_O2*I_O2/4
            
            x_data = np.concatenate((X,X_O2),axis=None)
            y_data = np.concatenate((A,A_O2),axis=None)
            result = stats.linregress(x_data, y_data) #slope, intercept, r, p, se    
            
            df.loc[count,'Rd'] = result.intercept
            df.loc[count,'R2'] = result.rvalue**2
            df.loc[count,'Replicate'] = replicate
            df.loc[count,'Slope'] = result.slope
            df.loc[count,'Std.err'] = result.stderr            
            df2.loc[count,'X'] = x_data
            df2.loc[count,'Y'] = y_data
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
        x0 = np.array([150,0.5])    
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
            axis.plot(par_values[i]-60, k2_values, color='red',marker=m[i],markersize=8)
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
        ax.errorbar(par_values,k2_plant1_ave_d,k2_std1_d,fmt='ko',mfc='none',label=label1,markersize=10)
        ax.errorbar(par_values,k2_plant2_ave_d,k2_std2_d,fmt='k<',mfc='none',label=label2,markersize=10)
        ax.set_xlabel('Irradiance (µmol $m^{-2}$ $s^{-1}$)',fontsize=24)
        ax.set_ylabel('k${_2}$ (mol $e^{-}$ mol photon$^{-1}$)',fontsize=24)
        self.show_significant(p_values,k2_plant1_ave_d,k2_plant2_ave_d,k2_std1_d,k2_std2_d,ax)          
        return p_values
    
    def get_Sco_data(self,Rd_values):  
        
        self.gas_exch_measurement.set_O2(0.21)
        ACIH = self.gas_exch_measurement.get_ACI_data()
        ACIH = ACIH[ACIH['CO2R'].between(50,210)]

        self.gas_exch_measurement.set_O2(0.02)
        ACIL = self.gas_exch_measurement.get_ACI_data()
        ACIL = ACIL[ACIL['CO2R'].between(50,210)]
        cols =  ['Replicate','CiH','AH','CiL','AL','Rd']        
        df = pd.DataFrame([],columns=cols)
        replicates = ACIH['Replicate'].unique()
        count = 0
        for replicate in replicates:  
            df2 = pd.DataFrame([],columns=cols)
            ACIH_r = ACIH[ACIH['Replicate']==replicate]
            AH = ACIH_r['Net CO2 assimilation rate'].values
            CiH = ACIH_r['Intercellular CO2 concentration'].values 
            ACIL_r = ACIL[ACIL['Replicate']==replicate]
            AL = ACIL_r['Net CO2 assimilation rate'].values
            CiL = ACIL_r['Intercellular CO2 concentration'].values  
            Rd_value = Rd_values[count]
            df2.loc[:,'AH'] = AH
            df2.loc[:,'CiH'] = CiH
            df2.loc[:,'Replicate'] = replicate
            df2.loc[:,'Rd'] = Rd_value            
            df2.loc[:,'AL'] = AL
            df2.loc[:,'CiL'] = CiL    
            df = df.append(df2,'sort=False')
            count+=1        
        return df
    
    def estimate_bH_bL(self,Rd_values):
        data = self.get_Sco_data(Rd_values)
        CiH = data['CiH'].values*constants.atm/10**5
        AH = data['AH'].values   
        CiL = data['CiL'].values*constants.atm/10**5
        AL = data['AL'].values   
        df = pd.DataFrame([],columns=['bH','Std.err_bH','R2_bH','intercept_bH',\
                          'bL','Std.err_bL','R2_bL','intercept_bL'])
        result = stats.linregress(CiH, AH) #slope, intercept, r, p, se 
        bH = result.slope; intercept_bH = result.intercept; 
        R2_bH = result.rvalue**2
        df.loc[0,'R2_bH'] = R2_bH
        df.loc[0,'bH'] = bH
        df.loc[0,'intercept_bH'] = intercept_bH
        df.loc[0,'Std.err_bH'] = result.stderr            
        
        result = stats.linregress(CiL, AL) #slope, intercept, r, p, se
        bL = result.slope; intercept_bL = result.intercept; 
        R2_bL = result.rvalue**2
        
        df.loc[0,'R2_bL'] = R2_bL
        df.loc[0,'bL'] = bL
        df.loc[0,'intercept_bL'] = intercept_bL
        df.loc[0,'Std.err_bL'] = result.stderr            
        
        
#        fig, (ax1, ax2) = plt.subplots(ncols=2)        
#        ax1.plot(CiH,AH,'ko')
#        ax1.plot(CiH,bH*CiH+intercept_bH)
#        ax2.plot(CiL,AL,'ko')
#        ax2.plot(CiL,bL*CiL+intercept_bL)        
#        
#        text = 'y = ' + str(np.round(bH,3)) + '*x + '+str(np.round(intercept_bH,3))\
#              + ' \n R2 = '+str(np.round(R2_bH,3))
#
#        text2 = 'y = ' + str(np.round(bL,3)) + '*x + '+str(np.round(intercept_bL,3))\
#              + ' \n R2 = '+str(np.round(R2_bL,3))
#              
#        ax1.text(min(CiH)+1,max(AH)-3,text)
#        ax2.text(min(CiL)+1,max(AL)-3,text2)
#        plt.show()   
        return df
        
        
    def model_Sco(self,Sco,Rd_values):
        df = self.estimate_bH_bL(Rd_values)
        data = self.get_Sco_data(Rd_values)
        CiH = data['CiH'].values
        AH = data['AH'].values   
        CiL = data['CiL'].values
        AL = data['AL'].values 
        
        AH = data['AH'].values
        Rd = data['Rd'].values      
        AL = data['AL'].values                
        bH = df['bH'].values
        bL = df['bL'].values
        CiH = data['CiH'].values*constants.atm/10**5
        CiL = data['CiL'].values*constants.atm/10**5   
        OH = 210 #mbar
        OL = 20
        AH_mod = (AL + Rd)*bH/bL - (0.5*(OH-OL)/Sco + CiL-CiH)*bH + Rd
        diff = (AH-AH_mod).astype(np.float64)
        return diff
        
    def estimate_Sco(self,Rd_values):
        bnds=((0.1),(10)) #lb,ub
        x0 = np.array([3.2])   
        Rd = Rd_values.astype(np.float64)
        result = optimize.least_squares(self.model_Sco,x0,args=[Rd],method='trf',bounds=bnds)
        res = self.model_Sco(result.x,Rd)
        J = np.array(result.jac)
        S = np.array(res).T.dot(np.array(res))
        H=2*J.T.dot(J);
        degfr=len(res)-1;
        G=np.linalg.inv(H);
        var_1=2*S*G[0,0]/degfr;
        var_1 = np.sqrt(var_1)
        Sco = pd.DataFrame([], columns = ['Sco','Std.err'])
        if result.success:
            Sco.loc[0,'Sco']=result.x
            Sco.loc[0,'Std.err']=var_1
            return Sco          
        else:
            raise ValueError(result.message)
            return []        

    def get_vcmax_data(self,RD,JMAX,THETA,K2LL):
        self.gas_exch_measurement.set_O2(0.21)
        ACIH = self.gas_exch_measurement.get_ACI_data()
        replicates = ACIH['Replicate'].unique()
        cols = ['Rd','Theta','Jmax','k2LL','Ci','Iinc','A']  # RD THETA JMAX K2LL CI IINC A;
        df_vcmax = pd.DataFrame([],columns = cols)
        count = 0 
        for replicate in replicates:
            df = pd.DataFrame([],columns = cols)
            ACIH_r = ACIH[ACIH['Replicate']==replicate]
            CI = ACIH_r['Intercellular CO2 concentration'].values
            IINC = ACIH_r['Irradiance'].values
            A = ACIH_r['Net CO2 assimilation rate'].values            
            Rd = RD[count]
            k2LL = K2LL[count]
            df.loc[:,'A'] = A                        
            df.loc[:,'Rd'] = Rd
            df.loc[:,'Theta'] = THETA
            df.loc[:,'Jmax'] = JMAX
            df.loc[:,'k2LL'] = k2LL
            df.loc[:,'Ci'] = CI
            df.loc[:,'Iinc'] = IINC
            df_vcmax = df_vcmax.append(df)
            count+=1
        return df_vcmax
        
    def model_Vcmax(self,xo,RDs,JMAXs,THETAs,K2LLs):
        VCMAX,TP,R = xo
        df_vcmax = self.get_vcmax_data(RDs,JMAXs,THETAs,K2LLs)
        A = df_vcmax['A'].values
        RD = df_vcmax['Rd'].values
        JMAX = df_vcmax['Jmax'].values
        THETA = df_vcmax['Theta'].values
        K2LL = df_vcmax['k2LL'].values
        CI = df_vcmax['Ci'].values
        IINC = df_vcmax['Iinc'].values
        
        O = 210 #mbar
        SCO = 3.259;
        GAMMAX = 0.5*O/SCO;
        GM0 = 0;
#        R = 1;
        
        #Rubisco-limited part;
        KMC = 267
        KMO = 164
        KMCMO = KMC*(1+O/KMO);
        X1R = VCMAX;
        X2R = KMCMO;
        PR = GM0*(X2R+GAMMAX)+(X1R-RD)*R;
        QR = (CI-GAMMAX)*X1R-(CI+X2R)*RD;
        AAR = (1.+R)*X2R + GAMMAX + R*CI;
        BBR = -((X2R+GAMMAX)*(X1R-RD)+PR*(CI+X2R)+R*QR);
        CCR = PR*QR;
        AR = (-BBR-(BBR**2-4.*AAR*CCR)**0.5)/(2.*AAR);
        #Electron transport limited part;
        BB = K2LL*IINC + JMAX;
        J = (BB-(BB**2-4*THETA*JMAX*K2LL*IINC)**0.5)/(2*THETA);
        X1J = J/4;
        X2J = 2*GAMMAX;
        PJ = GM0*(X2J+GAMMAX)+(X1J-RD)*R;
        QJ = (CI-GAMMAX)*X1J-(CI+X2J)*RD;
        AAJ = (1.+R)*X2J + GAMMAX + R*CI;
        BBJ = -((X2J+GAMMAX)*(X1J-RD)+PJ*(CI+X2J)+R*QJ);
        CCJ = PJ*QJ;
        AJ = (-BBJ-(BBJ**2-4.*AAJ*CCJ)**0.5)/(2.*AAJ);
        #TPU limited part;
        AP = 3*TP-RD;
        A1 = np.minimum(AR,AJ)
        A2 = np.minimum(A1,AP)  
#        print(K2LL)
        return A - A2
    
    
    def estimate_Vcmax(self,inputs):
        RD = inputs.get('Rd')*-1
        JMAX = inputs.get('Jmax')
        THETA = inputs.get('Theta')
        K2LL = inputs.get('k2LL')
        bnds=((0,0,0),(700,700,10)) 
        x0 = np.array([250,120,5])        #Vcmax, TP ,R     
        result = optimize.least_squares(self.model_Vcmax,x0,args=[RD,JMAX,THETA,K2LL],method='trf',bounds=bnds)
        res = self.model_Vcmax(result.x,RD,JMAX,THETA,K2LL)
        print(res)
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