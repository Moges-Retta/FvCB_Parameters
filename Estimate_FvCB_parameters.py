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
from scipy import optimize, constants,  stats

PATH = (r'\\WURNET.NL\Homes\retta001\My Documents\Project\2021\GasExchange\\')

class Estimate_FvCB_parameters:
    """   
            Model for FvCB model parameters 
            A :  net photosynthesis (µmol m-2 s-1)
            
            The method is explained in Yin et al. 2009, PCE,10.1111/j.1365-3040.2009.01934.x
            
            
            Jmax            # Maximum rate of electron transport through Photosystem II 
                              at saturating light, 
                              µmol e− m−2 leaf s−1
                              
            teta            # Convexity factor of the response of J to Iinc                  
            
                              
            alpha2LL        # Quantum yield of electron transport through Photosystem II
                              under strictly electron-transport-limiting conditions on the
                              basis of light absorbed by both Photosystem I and Photosystem
                              II,
                              mol e− mol−1 photon
            
            phi2            # Quantum yield of electron transport through Photosystem II, 
                              mol e− mol−1 photon
                              
            k2LL            # Conversion factor of incident irradiance into electron 
                              transport under electron-transport-limited conditions, 
                              mol e− mol−1 photon
            
            s               # Slope of the assumed linear relationship between AN and 1
                            # 1/4*Iinc*Phi2 under strictly electron-transport-limited conditions
                            
            KmC             # Michaelis–Menten constant of Rubisco for CO2, 
                              µbar CO2
            
            KmO             # Michaelis–Menten constant of Rubisco for O, 
                              mbar
            Tp              # Rate of triose phosphate utilization, 
                              µmol phosphate m−2 leaf s−1
            
                            
            Sco             # Relative CO2/O2 specificity factor of Rubisco, 
                              mbar O2 µbar−1 CO2    
                              
            gamma_star      # CO2 compensation point, 
                              µbar CO2
                        
            rsc             The fraction of mitochondria in the inner cytosol
            
            k2               Light conversion efficiency

            k2LL             Light conversion efficiency at limiting light
            
            Vcmax            Maximum rate of Rubisco carboxylation,µmol m−2 leaf s−1
    """
    
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
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)                
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
            A = A_I_r['Net_CO2_assimilation_rate'].values
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
   

    def compare_df(self,df1,df2):
        [t,p]= stats.ttest_ind(df1,df2, equal_var = False)
        return p
     
        
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
            A = A_I_r['Net_CO2_assimilation_rate'].values
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
            
            A = A_I_r['Net_CO2_assimilation_rate'].values
            X = PhiPS2*I/4
            
            A_O2 = A_I_r_O2['Net_CO2_assimilation_rate'].values
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
        Estimate Jmax, convexity factor (theta) and quantum efficiency of PSII
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
            bnds=((0,0,0),(1,1,1000)) 
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
        Estimate Jmax and curvature factor (theta) using data for quantum efficiency of PSII
        e flow (PhiPS2)
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

        AI = self.gas_exch_measurement.get_AI_data()
        Iave = AI['Irradiance'].values
        phiPS2 = AI['PhiPS2'].values
        
        
        Jmax,theta = params
        k2LL = s*PHI2LL
        J = s*Iave*phiPS2
        SQ = ((k2LL*Iave+Jmax)**2-4*theta*k2LL*Jmax*Iave)**0.5
        J_mod = (k2LL*Iave+Jmax-SQ)/(2*theta) 
        
        JCF = np.array([J]).transpose()
        JCF = np.reshape(J,(4,14))
        Iinc = np.array([Iave]).transpose()
        Iinc = np.reshape(Iinc,(4,14))
        J_mod = np.array([J_mod]).transpose()
        J_mod = np.reshape(J_mod,(4,14))
        
        JCF_ave = stats.gmean(JCF,axis=0)
        stderr = stats.tstd(JCF,axis=0)
        Iinc_ave = stats.gmean(Iinc,axis=0)
        J_mod_ave = stats.gmean(J_mod,axis=0)

        fig, ax = plt.subplots()
        ax.errorbar(Iinc_ave,JCF_ave,stderr,fmt='ko',label='CF') 
        ax.plot(Iinc_ave,J_mod_ave,'k-',linewidth=2.0,label='Model')  
        ax.set_xlabel('Irradiance (µmol $m^{-2}$ $s^{-1}$)')
        ax.set_ylabel('J (µmol $m^{-2}$ $s^{-1}$)')  
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
        bnds=((0,0,0),(1,1,1000)) 
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
        bnds=((0,0),(1000,1)) #lb,ub
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
    
    def get_Sco_data(self,Rd_values,bH,bL):  
        
        self.gas_exch_measurement.set_O2(0.21)
        ACIH = self.gas_exch_measurement.get_ACI_data()
        ACIH = ACIH[ACIH['CO2R'].between(40,210)]

        self.gas_exch_measurement.set_O2(0.02)
        ACIL = self.gas_exch_measurement.get_ACI_data()
        ACIL = ACIL[ACIL['CO2R'].between(40,210)]
        cols =  ['Replicate','CiH','AH','CiL','AL','Rd','bH','bL']        
        df = pd.DataFrame([],columns=cols)
        replicates = ACIH['Replicate'].unique()
        for replicate in replicates:  
            df2 = pd.DataFrame([],columns=cols)
            ACIH_r = ACIH[ACIH['Replicate']==replicate]
            AH = ACIH_r['Net_CO2_assimilation_rate'].values
            CiH = ACIH_r['Intercellular_CO2_concentration'].values 
            ACIL_r = ACIL[ACIL['Replicate']==replicate]
            AL = ACIL_r['Net_CO2_assimilation_rate'].values
            CiL = ACIL_r['Intercellular_CO2_concentration'].values  
            Rd_value = Rd_values[replicate-1]*-1
            bL_value = bL[replicate-1]
            bH_value = bH[replicate-1]
            
            df2.loc[:,'AH'] = AH
            df2.loc[:,'CiH'] = CiH
            df2.loc[:,'Replicate'] = replicate
            df2.loc[:,'Rd'] = Rd_value 
            df2.loc[:,'bL'] = bL_value            
            df2.loc[:,'bH'] = bH_value                        
            df2.loc[:,'AL'] = AL
            df2.loc[:,'CiL'] = CiL    
            df = df.append(df2,'sort=False')
            # df.to_excel(PATH + 'SCO_data_'+self.gas_exch_measurement.get_species()+'_'+ self.gas_exch_measurement.get_treatment()+'.xlsx', index = False)
        
        return df
    
    def estimate_bH_bL(self,Rd_values):
        
        """
        Estimate initial carboxylation efficiency at 21 % O2 (bH) 
        and 2 % O2 (bL) as slope of initial part of A-CI (50<=CO2<=150).
        
        """
        
        data = self.get_Sco_data(Rd_values,Rd_values,Rd_values)
        replicates = data['Replicate'].unique()
        df = pd.DataFrame([],columns=['Replicate','bH','Std.err_bH','R2_bH',\
                             'bL','Std.err_bL','R2_bL'])
                        
        for replicate in replicates:  
            data_r=data[data['Replicate']==replicate]
            CiH = data_r['CiH'].values*constants.atm/10**5
            AH = data_r['AH'].values   
            CiL = data_r['CiL'].values*constants.atm/10**5
            AL = data_r['AL'].values   
            result = stats.linregress(CiH, AH) #slope, intercept, r, p, se 
            bH = result.slope; 
            R2_bH = result.rvalue**2
            df.loc[replicate-1,'R2_bH'] = R2_bH
            df.loc[replicate-1,'bH'] = bH
            df.loc[replicate-1,'Std.err_bH'] = result.stderr            
            df.loc[replicate-1,'Replicate'] = replicate            
            
            result = stats.linregress(CiL, AL) #slope, intercept, r, p, se
            bL = result.slope; 
            R2_bL = result.rvalue**2
            
            df.loc[replicate-1,'R2_bL'] = R2_bL
            df.loc[replicate-1,'bL'] = bL
            df.loc[replicate-1,'Std.err_bL'] = result.stderr            
#        
        # plt.rcParams["figure.figsize"] = (10,5)
        # fig, (ax1, ax2) = plt.subplots(ncols=2,constrained_layout=True)        
        # ax1.plot(CiH,AH,'ko')
        # ax1.plot(CiH,bH*CiH+intercept_bH)
        # ax1.set_ylabel('AH (µmol $m^{-2}$ $s^{-1}$)')
        # ax1.set_xlabel('CiH (µbar)')
        
        # ax2.plot(CiL,AL,'ko')
        # ax2.plot(CiL,bL*CiL+intercept_bL)        
        # ax2.set_ylabel('AL (µmol $m^{-2}$ $s^{-1}$)')
        # ax2.set_xlabel('CiL (µbar)')
       
        # text = 'y = ' + str(np.round(bH,3)) + '*x + '+str(np.round(intercept_bH,3))\
        #       + ' \n R2 = '+str(np.round(R2_bH,3))

        # text2 = 'y = ' + str(np.round(bL,3)) + '*x + '+str(np.round(intercept_bL,3))\
        #       + ' \n R2 = '+str(np.round(R2_bL,3))
              
        # ax1.text(min(CiH)+1,max(AH)-3,text)
        # ax2.text(min(CiL)+1,max(AL)-3,text2)
        # plt.show()   
        return df
        
        
    def model_Sco(self,Sco,data):
        AH = data['AH'].values   
        AL = data['AL'].values 
        
        Rd = data['Rd'].values      
        bH = data['bH'].values
        bL = data['bL'].values
        CiH = data['CiH'].values*constants.atm/10**5
        CiL = data['CiL'].values*constants.atm/10**5   
        OH = 210 #mbar
        OL = 20
        AH_mod = (AL + Rd)*bH/bL - (0.5*(OH-OL)/Sco + CiL-CiH)*bH - Rd
        diff = (AH-AH_mod).astype(np.float64)
        return diff
        
    
    def estimate_Sco(self,Rd_values,bH,bL):
        data = self.get_Sco_data(Rd_values,bH,bL)
        bnds=((0.1),(10)) #lb,ub
        x0 = np.array([3.2])   
        result = optimize.least_squares(self.model_Sco,x0,args=[data],method='trf',bounds=bnds)
        res = self.model_Sco(result.x,data)
        J = np.array(result.jac)
        S = np.array(res).T.dot(np.array(res))
        H=2*J.T.dot(J);
        degfr=len(res)-1;
        G=np.linalg.inv(H);
        var_1=2*S*G[0,0]/degfr;
        var_1 = np.sqrt(var_1)
        Sco = pd.DataFrame([], columns = ['Sco','Std.err'])
        if result.success:
            Sco.loc[0,'Sco']=result.x[0]
            Sco.loc[0,'Std.err']=var_1
            return Sco          
        else:
            raise ValueError(result.message)
            return []        

    def get_vcmax_data(self,Rd,Jmax,Theta,K2LL):
        cols = ['Replicate','Rd','Theta','Jmax','k2LL','Ci','Iinc','A','O','curve']  # RD THETA JMAX K2LL CI IINC A;
        vcmax_data = pd.DataFrame([],columns = cols)
        O2 = [210,20]
        for O in O2:
            self.gas_exch_measurement.set_O2(O/1000)
            ACIH = self.gas_exch_measurement.get_ACI_data()
            replicates = ACIH['Replicate'].unique()
            df_vcmax = pd.DataFrame([],columns = cols)
            count = 0 
            for replicate in replicates:
                df = pd.DataFrame([],columns = cols)
                ACIH_r = ACIH[ACIH['Replicate']==replicate]
                ci = ACIH_r['Intercellular_CO2_concentration'].values*constants.atm/10**5  
                i_inc = ACIH_r['Irradiance'].values
                A = ACIH_r['Net_CO2_assimilation_rate'].values  

                rd = Rd[replicate-1]
                theta = Theta
                jmax = Jmax               
                k2LL = K2LL[replicate-1]
                df.loc[:,'A'] = A                        
                df.loc[:,'Rd'] = rd*-1
                df.loc[:,'Theta'] = theta
                df.loc[:,'Jmax'] = jmax
                df.loc[:,'k2LL'] = k2LL
                df.loc[:,'Ci'] = ci
                df.loc[:,'Iinc'] = i_inc
                df.loc[:,'O'] = O    
                df.loc[:,'curve'] = 'ACI'    
                df.loc[:,'Replicate']=replicate    
                       
                df_vcmax = df_vcmax.append(df)
                count+=1
            vcmax_data = vcmax_data.append(df_vcmax)   
            
            AIH = self.gas_exch_measurement.get_AI_data()            
            count = 0 
            df_vcmax = pd.DataFrame([],columns = cols)
            
            for replicate in replicates:
                df = pd.DataFrame([],columns = cols)
                ACIH_r = AIH[AIH['Replicate']==replicate]
                ci = ACIH_r['Intercellular_CO2_concentration'].values*constants.atm/10**5  
                i_inc = ACIH_r['Irradiance'].values
                A = ACIH_r['Net_CO2_assimilation_rate'].values   

                rd = Rd[replicate-1]
                k2LL = K2LL[replicate-1]
                theta = Theta
                jmax = Jmax 
                df.loc[:,'A'] = A                        
                df.loc[:,'Rd'] = rd*-1
                df.loc[:,'Theta'] = theta
                df.loc[:,'Jmax'] = jmax
                df.loc[:,'k2LL'] = k2LL
                df.loc[:,'Ci'] = ci
                df.loc[:,'Iinc'] = i_inc
                df.loc[:,'O'] = O  
                df.loc[:,'curve'] = 'AI'   
                df.loc[:,'Replicate']=replicate                 
                                      
                df_vcmax = df_vcmax.append(df)
                count+=1 
            vcmax_data = vcmax_data.append(df_vcmax) 
        # species =  self.gas_exch_measurement.get_species()    
        # treatment = self.gas_exch_measurement.get_treatment()
        # vcmax_data.to_excel(PATH +'vcmax_data_leakCorr_'+str(species)+str(treatment)+'.xlsx', index = False)    
        return vcmax_data
       
    def calculate_A(self,xo,df_vcmax,sco):
        vcmax,TP,R,gm0 = xo
        
        rd = df_vcmax['Rd'].values.astype(float)
        jmax = df_vcmax['Jmax'].values.astype(float)
        theta = df_vcmax['Theta'].values.astype(float)
        k2LL = df_vcmax['k2LL'].values.astype(float)
        ci = df_vcmax['Ci'].values.astype(float) #µbar
        i_inc = df_vcmax['Iinc'].values.astype(float)
        O = df_vcmax['O'].values.astype(float)
        
        O = 210 #mbar
        sco = sco.astype(float)        
        gamma_x = 0.5*O/sco
#        gm0 = 0
        #Rubisco-limited part;
        kmc = 267
        kmo = 164
        kmm = kmc*(1+O/kmo)
        x1R = vcmax
        x2R = kmm
        PR = gm0*(x2R+gamma_x)+(x1R-rd)*R
        QR = (ci-gamma_x)*x1R-(ci+x2R)*rd
        AAR = (1.+R)*x2R + gamma_x + R*ci
        BBR = -((x2R+gamma_x)*(x1R-rd)+PR*(ci+x2R)+R*QR)
        CCR = PR*QR
        AR = (-BBR-(BBR**2-4*AAR*CCR)**0.5)/(2*AAR)
        #Electron transport limited part;
        BB = k2LL*i_inc + jmax;
        J = (BB-(BB**2-4*theta*jmax*k2LL*i_inc)**0.5)/(2*theta);
        x1J = J/4;
        x2J = 2*gamma_x;
        PJ = gm0*(x2J+gamma_x)+(x1J-rd)*R;
        QJ = (ci-gamma_x)*x1J-(ci+x2J)*rd;
        AAJ = (1+R)*x2J + gamma_x + R*ci;
        BBJ = -((x2J+gamma_x)*(x1J-rd)+PJ*(ci+x2J)+R*QJ);
        CCJ = PJ*QJ;
        AJ = (-BBJ-(BBJ**2-4*AAJ*CCJ)**0.5)/(2*AAJ);
        #TPU limited part;
        AP = 3*TP-rd;
        A1 = np.minimum(AR,AJ)
        A_mod = np.minimum(A1,AP) 
        
        return A_mod
 

    def calculate_A_mod_ave(self,xo,rds,jmaxs,thetas,k2LLs,sco,ci_ave,i_inc,curve):
        vcmax,TP,R,gm0 = xo
        
        df_vcmax = self.get_vcmax_data(rds,jmaxs,thetas,k2LLs)
        if curve:
                df_vcmax = df_vcmax[df_vcmax['curve']==curve]
                df_vcmax = df_vcmax[df_vcmax['O']==210]

        ci = df_vcmax['Ci'].values.astype(float)        
        i_inc = df_vcmax['Iinc'].values.astype(float)        
        
        rd = df_vcmax['Rd'].values.astype(float)
        rd = np.nanmean(rd)
        jmax = df_vcmax['Jmax'].values.astype(float)
        jmax = np.nanmean(jmax)
        
        theta = df_vcmax['Theta'].values.astype(float)
        theta = np.nanmean(theta)
        
        k2LL = df_vcmax['k2LL'].values.astype(float)
        k2LL = np.nanmean(k2LLs)
#        
#        O = df_vcmax['O'].values.astype(float)
        O = 210 #mbar
        sco = sco.astype(float)
        sco = np.nanmean(sco)
        
        gamma_x = 0.5*O/sco
#        gm0 = 0
        #Rubisco-limited part;
        kmc = 267
        kmo = 164
        kmm = kmc*(1+O/kmo)
        x1R = vcmax
        x2R = kmm
        PR = gm0*(x2R+gamma_x)+(x1R-rd)*R
        QR = (ci-gamma_x)*x1R-(ci+x2R)*rd
        AAR = (1+R)*x2R + gamma_x + R*ci
        BBR = -((x2R+gamma_x)*(x1R-rd)+PR*(ci+x2R)+R*QR)
        CCR = PR*QR
        AR = (-BBR-(BBR**2-4.*AAR*CCR)**0.5)/(2*AAR)
        #Electron transport limited part;
        BB = k2LL*i_inc + jmax;
        J = (BB-(BB**2-4*theta*jmax*k2LL*i_inc)**0.5)/(2*theta);
        x1J = J/4;
        x2J = 2*gamma_x;
        PJ = gm0*(x2J+gamma_x)+(x1J-rd)*R;
        QJ = (ci-gamma_x)*x1J-(ci+x2J)*rd;
        AAJ = (1.+R)*x2J + gamma_x + R*ci;
        BBJ = -((x2J+gamma_x)*(x1J-rd)+PJ*(ci+x2J)+R*QJ);
        CCJ = PJ*QJ;
        AJ = (-BBJ-(BBJ**2-4.*AAJ*CCJ)**0.5)/(2.*AAJ);
        #TPU limited part;
        AP = 3*TP-rd;
        A1 = np.minimum(AR,AJ)
        A_mod = np.minimum(A1,AP) 
        
        AA = A_mod + R*(A_mod+rd)+gm0*(ci-gamma_x) 
        gm_mod = (AA + (AA**2-4*(ci-gamma_x)*A_mod*gm0)**0.5)/(2*(ci-gamma_x))
        
        
        # A_mod = np.array([A_mod]).transpose()
        # A_mod = np.reshape(A_mod,(4,len(ci_ave)))
        # J = np.array([J]).transpose()
        # J = np.reshape(J,(4,len(ci_ave)))
        # gm_mod = np.array([gm_mod]).transpose()
        # gm_mod = np.reshape(gm_mod,(4,len(ci_ave)))
        
        
        return [A_mod,gm_mod,J]


    def plot_gm(self,xo,rds,jmaxs,thetas,k2LLs,sco):
        plt.rcParams["figure.figsize"] = (15,8) 
        plt.rcParams.update({'font.size': 12})

        self.gas_exch_measurement.set_O2(0.21)      
        ACI = self.gas_exch_measurement.average_A_CI()
        A = ACI['Net_CO2_assimilation_rate'].values 
        ci = ACI['Intercellular_CO2_concentration'].values*constants.atm/10**5
        len_ci_pts = len(ci)
        i_inc = ACI['Irradiance'].values   
        gamma_x = 0.5*210/sco
        [A_mod_ci,gm_ci_cf,J] = self.calculate_A_mod_ave(xo,rds,jmaxs,thetas,k2LLs,sco,ci,i_inc,'')
        A_mod_ci = np.array([A_mod_ci]).transpose()
        A_mod_ci = np.reshape(A_mod_ci,(4,len_ci_pts))
        J = np.array([J]).transpose()
        J = np.reshape(J,(4,len_ci_pts))

        
        Rd = np.mean(rds)
        AA = gamma_x*(J + 8*(A + Rd))
        gm_ci =  A/(ci-AA/(J-4*(A+Rd)));
        gm_ci = np.array([gm_ci]).transpose()
        gm_ci = np.reshape(gm_ci,(4,len_ci_pts))

        self.gas_exch_measurement.set_O2(0.21)      
        AI = self.gas_exch_measurement.average_A_I()
        A_i = AI['Net_CO2_assimilation_rate'].values 
        ci_i = AI['Intercellular_CO2_concentration'].values*constants.atm/10**5
        i_inc_i = AI['Irradiance'].values       
        [A_mod_i,gm_i_cf,J] = self.calculate_A_mod_ave(xo,rds,jmaxs,thetas,k2LLs,sco,ci_i,i_inc_i,'')
        AA = gamma_x*(J + 8*(A_i + Rd))
        gm_i =  A_i/(ci_i-AA/(J-4*(A_i+Rd)));
        len_i_pts=len(i_inc_i)
        A_mod_i = np.array([A_mod_i]).transpose()
        A_mod_i = np.reshape(A_mod_i,(4,len_i_pts))
        J = np.array([J]).transpose()
        J = np.reshape(J,(4,len_i_pts))
        gm_i = np.array([gm_i]).transpose()
        gm_i = np.reshape(gm_i,(4,len_i_pts))
        
        
        fig,(ax1,ax2) = plt.subplots(ncols=2,constrained_layout=True)        
        ax1.plot(ci,gm_ci,'ko',label='Variable J',mfc='white',mec='black',markersize=8)
        ax1.plot(ci,gm_ci_cf,'k-',label='Model')
        ax1.set_xlabel('Intercellular $CO_2$ (µmol $mol^{-1}$)',fontsize=20)
        ax1.set_ylabel('$g_m$ (mol $m^{-2}$ $s^{-1}$ $bar^{-1}$)',fontsize=20)
        ax1.xaxis.set_ticks(np.arange(0, 2050, 250))        
        ax2.plot(i_inc_i,gm_i,'ko',label='Variable J',mfc='white',mec='black',markersize=8)
        ax2.plot(i_inc_i,gm_i_cf,'k-',label='Model')
        ax2.set_xlabel('Irradiance (µmol $m^{-2}$ $s^{-1}$)',fontsize=20)
        ax2.set_ylabel('$g_m$ (mol $m^{-2}$ $s^{-1}$ $bar^{-1}$)',fontsize=20)    
        ax2.legend(loc='lower right', fontsize='x-large')     
        ax2.xaxis.set_ticks(np.arange(0, 2200, 300))   
            
        plt.show()  
        
        
    def plot_A(self,xo,rds,jmaxs,thetas,k2LLs,sco):
        len_ci_pts = 15
        len_i_pts = 14
        plt.rcParams["figure.figsize"] = (15,8) 
        plt.rcParams.update({'font.size': 12})

        self.gas_exch_measurement.set_O2(0.21)      
        ACI = self.gas_exch_measurement.get_ACI_data()
        ACI = ACI[ACI['Oxygen_level']==0.21]

        A = ACI['Net_CO2_assimilation_rate'].values 
        # A_err = ACI['Photo_err'].values/2 
        ci = ACI['Intercellular_CO2_concentration'].values*constants.atm/10**5

        i_inc = ACI['Irradiance'].values       
        [A_mod_ci,gm_ci,J] = self.calculate_A_mod_ave(xo,rds,jmaxs,thetas,k2LLs,sco,ci,i_inc,'ACI')

        A_mod_ci = np.array([A_mod_ci]).transpose()
        A_mod_ci = np.reshape(A_mod_ci,(4,len_ci_pts))
        J = np.array([J]).transpose()
        J = np.reshape(J,(4,len_ci_pts))
        gm_ci = np.array([gm_ci]).transpose()
        gm_ci = np.reshape(gm_ci,(4,len_ci_pts))

        A = np.array([A]).transpose()
        A = np.reshape(A,(4,len_ci_pts))
        ci = np.array([ci]).transpose()
        ci = np.reshape(ci,(4,len_ci_pts)) 
        Aave = np.nanmean(A,axis=0)
        A_err_ci = np.std(A,axis=0)/2

        ci = np.nanmean(ci,axis=0)
        A_mod_ci = np.nanmean(A_mod_ci,axis=0)
        df = pd.DataFrame([],columns = ['A','Ci','A_err'])        
        df.loc[:,'A']=Aave
        df.loc[:,'Ci']=ci
        df.loc[:,'A_err']=A_err_ci
        df = df.sort_values(by=['Ci'])

        self.gas_exch_measurement.set_O2(0.21)      
        AI = self.gas_exch_measurement.get_AI_data()
        A_i = AI['Net_CO2_assimilation_rate'].values 
        # A_err_i = AI['Photo_err'].values/2 
        ci_i = AI['Intercellular_CO2_concentration'].values*constants.atm/10**5
        i_inc_i = AI['Irradiance'].values       
        [A_mod_i,gm_i,J] = self.calculate_A_mod_ave(xo,rds,jmaxs,thetas,k2LLs,sco,ci_i,i_inc_i,'AI')
        
        A_mod_i = np.array([A_mod_i]).transpose()
        A_mod_i = np.reshape(A_mod_i,(4,len_i_pts))
        J = np.array([J]).transpose()
        J = np.reshape(J,(4,len_i_pts))
        gm_i = np.array([gm_i]).transpose()
        gm_i = np.reshape(gm_i,(4,len_i_pts))
        A_i = np.array([A_i]).transpose()
        A_i = np.reshape(A_i,(4,len_i_pts))
        ci_i = np.array([ci_i]).transpose()
        ci_i = np.reshape(ci_i,(4,len_i_pts))           
        i_inc_i = np.reshape(i_inc_i,(4,len_i_pts))           
        
        A_i_ave = np.nanmean(A_i,axis=0)
        A_err_i = np.std(A_i,axis=0)/2
        
        ci_i = np.nanmean(ci_i,axis=0)
        A_mod_i = np.nanmean(A_mod_i,axis=0)
        i_inc_i = np.nanmean(i_inc_i,axis=0)

        df2 = pd.DataFrame([],columns = ['A','Iinc','A_err'])        
        df2.loc[:,'A']=A_i_ave
        df2.loc[:,'Iinc']=i_inc_i
        df2.loc[:,'A_err']=A_err_i
        # df2 = df2.sort_values(by=['Iinc'])
        
        ci = df['Ci'].values
        Aave = df['A'].values
        A_err_ci = df['A_err'].values
        i_inc_i = df2['Iinc'].values
        A_i_ave = df2['A'].values
        A_err_i = df2['A_err'].values      
        
        A_mod_ci = np.sort(A_mod_ci)
        
        fig,(ax1,ax2) = plt.subplots(ncols=2,constrained_layout=True)        
        ax1.errorbar(ci,Aave,A_err_ci,fmt='ko',label='Expt.',mfc='white',mec='black',markersize=8)
        ax1.plot(ci,A_mod_ci,'k-',label='Model')
        ax1.set_xlabel('Intercellular $CO_2$ (µmol $mol^{-1}$)',fontsize=20)
        ax1.set_ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)',fontsize=20)
        ax1.xaxis.set_ticks(np.arange(0, 2050, 250))        
        ax2.errorbar(i_inc_i,A_i_ave,A_err_i,fmt='ko',label='Expt.',mfc='white',mec='black',markersize=8)
        ax2.plot(i_inc_i,A_mod_i,'k-',label='Model')
        ax2.set_xlabel('Irradiance (µmol $m^{-2}$ $s^{-1}$)',fontsize=20)
        ax2.set_ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)',fontsize=20)    
        ax2.legend(loc='lower right', fontsize='x-large')     
        ax2.xaxis.set_ticks(np.arange(0, 2200, 300))
        
        plt.show()  

    
    def model_Vcmax(self,xo,df_vcmax,sco):
        A_mod = self.calculate_A(xo,df_vcmax,sco)
        return df_vcmax['A'].values - A_mod
    
    
    def estimate_Vcmax(self,inputs):
        
        rd = inputs.get('Rd')*-1
        jmax = inputs.get('Jmax')
        theta = inputs.get('Theta')
        k2LL = inputs.get('k2LL')
        sco = inputs.get('Sco')
        
        df_vcmax = self.get_vcmax_data(rd,jmax,theta,k2LL)

        
        bnds=((0,0,0,0),(1000,70,20,20)) 
        x0 = np.array([200,12,0.5,1])        #Vcmax,TP,sigma,Jmax        
        result = optimize.least_squares(self.model_Vcmax,x0,args=[df_vcmax,sco],method='trf',bounds=bnds)
        # result = optimize.least_squares(self.model_Vcmax,x0,args=[df_vcmax,sco],method='lm')
        
        res = self.model_Vcmax(result.x,df_vcmax,sco)
        J = np.array(result.jac)
        S = np.array(res).T.dot(np.array(res))
        H=2*J.T.dot(J);
        degfr=len(res)-4;
        G = np.linalg.pinv(H)
        var_1=2*S*G[0,0]/degfr;
        var_2=2*S*G[1,1]/degfr;
        var_3=2*S*G[2,2]/degfr;
        var_4=2*S*G[3,3]/degfr;
        
        var_1 = np.sqrt(var_1)
        var_2 = np.sqrt(var_2)
        var_3 = np.sqrt(var_3)
        var_4 = np.sqrt(var_4)

        if result.success:
            self.plot_A(result.x,rd,jmax,theta,k2LL,sco)
            # self.plot_gm(result.x,rd,jmax,theta,k2LL,sco)
            cols = ['Vcmax','Vcmax_err','Tp','Tp_err','Sigma_gm','Sigma_gm_err','gm0','gm0_err']
            
            df=pd.DataFrame([],columns=cols)
            df.loc[0,'Vcmax'] = result.x[0]
            df.loc[0,'Tp'] = result.x[1]
            df.loc[0,'Sigma_gm'] = result.x[2]
            df.loc[0,'gm0'] = result.x[3]
            
            df.loc[0,'Vcmax_err'] = var_1
            df.loc[0,'Tp_err'] = var_2
            df.loc[0,'Sigma_gm_err'] = var_3
            df.loc[0,'gm0_err'] = var_4
            
            return df
        else:
            raise ValueError(result.message)
            return []        


    def get_vcmax_data_individual_plot(self,Rd,jmax,theta,K2LL,replicate):
        self.gas_exch_measurement.set_O2(0.21)
        ACIH = self.gas_exch_measurement.get_ACI_data()
        cols = ['Rd','Theta','Jmax','k2LL','Ci','Iinc','A','O']  # RD THETA JMAX K2LL CI IINC A;
        vcmax_data = pd.DataFrame([],columns = cols)
        df_vcmax = pd.DataFrame([],columns = cols)
        df = pd.DataFrame([],columns = cols)
        ACIH_r = ACIH[ACIH['Replicate']==replicate]
        ci = ACIH_r['Intercellular_CO2_concentration'].values*constants.atm/10**5  
        i_inc = ACIH_r['Irradiance'].values
        A = ACIH_r['Net_CO2_assimilation_rate'].values            
        df.loc[:,'A'] = A                        
        df.loc[:,'Rd'] = Rd*-1
        df.loc[:,'Theta'] = theta
        df.loc[:,'Jmax'] = jmax
        df.loc[:,'k2LL'] = K2LL
        df.loc[:,'Ci'] = ci
        df.loc[:,'Iinc'] = i_inc
        df_vcmax = df_vcmax.append(df)
        vcmax_data = vcmax_data.append(df_vcmax)   
            
        AIH = self.gas_exch_measurement.get_AI_data()            
        df = pd.DataFrame([],columns = cols)
        AIH_r = AIH[AIH['Replicate']==replicate]
        ci = AIH_r['Intercellular_CO2_concentration'].values*constants.atm/10**5  
        i_inc = AIH_r['Irradiance'].values
        A = AIH_r['Net_CO2_assimilation_rate'].values            
        df.loc[:,'A'] = A                        
        df.loc[:,'Rd'] = Rd*-1
        df.loc[:,'Theta'] = theta
        df.loc[:,'Jmax'] = jmax
        df.loc[:,'k2LL'] = K2LL
        df.loc[:,'Ci'] = ci
        df.loc[:,'Iinc'] = i_inc
        df_vcmax = df_vcmax.append(df)
        vcmax_data = vcmax_data.append(df_vcmax)          
        return df_vcmax
    

    def get_vcmax_data_individual(self,Rd,jmax,theta,K2LL,replicate):
        cols = ['Rd','Theta','Jmax','k2LL','Ci','Iinc','A','O']  # RD THETA JMAX K2LL CI IINC A;
        vcmax_data = pd.DataFrame([],columns = cols)
        O2 = [210]
        for O in O2:
            self.gas_exch_measurement.set_O2(O/1000)
            ACIH = self.gas_exch_measurement.get_ACI_data()
            df = pd.DataFrame([],columns = cols)
            ACIH_r = ACIH[ACIH['Replicate']==replicate]
            ci = ACIH_r['Intercellular_CO2_concentration'].values*constants.atm/10**5  
            i_inc = ACIH_r['Irradiance'].values
            A = ACIH_r['Net_CO2_assimilation_rate'].values            
            df.loc[:,'A'] = A                        
            df.loc[:,'Rd'] = Rd*-1
            df.loc[:,'Theta'] = theta
            df.loc[:,'Jmax'] = jmax
            df.loc[:,'k2LL'] = K2LL
            df.loc[:,'Ci'] = ci
            df.loc[:,'Iinc'] = i_inc
            df.loc[:,'O'] = O                
            vcmax_data = vcmax_data.append(df)   
            
            AIH = self.gas_exch_measurement.get_AI_data()            
            df = pd.DataFrame([],columns = cols)
            AIH_r = AIH[AIH['Replicate']==replicate]
            ci = AIH_r['Intercellular_CO2_concentration'].values*constants.atm/10**5  
            i_inc = AIH_r['Irradiance'].values
            A = AIH_r['Net_CO2_assimilation_rate'].values            
            df.loc[:,'A'] = A                        
            df.loc[:,'Rd'] = Rd*-1
            df.loc[:,'Theta'] = theta
            df.loc[:,'Jmax'] = jmax
            df.loc[:,'k2LL'] = K2LL
            df.loc[:,'Ci'] = ci
            df.loc[:,'Iinc'] = i_inc
            df.loc[:,'O'] = O                                
            vcmax_data = vcmax_data.append(df)          
        return vcmax_data


    def calculate_A_individual_plot(self,xo,df_vcmax,replicate,sco,ci,i_inc):
        vcmax,TP,R = xo
        rd = df_vcmax['Rd'].values.astype(float)
        rd = np.nanmean(rd)
        jmax = df_vcmax['Jmax'].values.astype(float)
        jmax = np.nanmean(jmax)        
        theta = df_vcmax['Theta'].values.astype(float)
        theta = np.nanmean(theta)        
        k2LL = df_vcmax['k2LL'].values.astype(float)
        k2LL = np.nanmean(k2LL)
        
        O = 210 #mbar
#        sco = 3.259
        sco = sco.astype(float)        
        gamma_x = 0.5*O/sco
        gm0 = 0
        #Rubisco-limited part;
        kmc = 267
        kmo = 164
        kmm = kmc*(1+O/kmo)
        x1R = vcmax
        x2R = kmm
        PR = gm0*(x2R+gamma_x)+(x1R-rd)*R
        QR = (ci-gamma_x)*x1R-(ci+x2R)*rd
        AAR = (1.+R)*x2R + gamma_x + R*ci
        BBR = -((x2R+gamma_x)*(x1R-rd)+PR*(ci+x2R)+R*QR)
        CCR = PR*QR
        AR = (-BBR-(BBR**2-4.*AAR*CCR)**0.5)/(2.*AAR)
        #Electron transport limited part;
        BB = k2LL*i_inc + jmax;
        J = (BB-(BB**2-4*theta*jmax*k2LL*i_inc)**0.5)/(2*theta);
        x1J = J/4;
        x2J = 2*gamma_x;
        PJ = gm0*(x2J+gamma_x)+(x1J-rd)*R;
        QJ = (ci-gamma_x)*x1J-(ci+x2J)*rd;
        AAJ = (1.+R)*x2J + gamma_x + R*ci;
        BBJ = -((x2J+gamma_x)*(x1J-rd)+PJ*(ci+x2J)+R*QJ);
        CCJ = PJ*QJ;
        AJ = (-BBJ-(BBJ**2-4.*AAJ*CCJ)**0.5)/(2.*AAJ);
        #TPU limited part;
        AP = 3*TP-rd;
        A1 = np.minimum(AR,AJ)
        A_mod = np.minimum(A1,AP) 
        return A_mod
    
       
    def calculate_A_individual(self,x0,df_vcmax,sco,replicate):
        vcmax,TP,R = x0
        rd = df_vcmax['Rd'].values.astype(float)
        jmax = df_vcmax['Jmax'].values.astype(float)
        theta = df_vcmax['Theta'].values.astype(float)
        k2LL = df_vcmax['k2LL'].values.astype(float)*constants.atm/10**5
        ci = df_vcmax['Ci'].values.astype(float)
        i_inc = df_vcmax['Iinc'].values.astype(float)
        O = df_vcmax['O'].values.astype(float)
        
#        O = 210 #mbar
#        sco = 3.259
        sco = sco.astype(float)        
#        O = 210 #mbar
        gamma_x = 0.5*O/sco
        gm0 = 0
        #Rubisco-limited part;
        kmc = 267
        kmo = 164
        kmm = kmc*(1+O/kmo)
        x1R = vcmax
        x2R = kmm
        PR = gm0*(x2R+gamma_x)+(x1R-rd)*R
        QR = (ci-gamma_x)*x1R-(ci+x2R)*rd
        AAR = (1.+R)*x2R + gamma_x + R*ci
        BBR = -((x2R+gamma_x)*(x1R-rd)+PR*(ci+x2R)+R*QR)
        CCR = PR*QR
        AR = (-BBR-(BBR**2-4.*AAR*CCR)**0.5)/(2.*AAR)
        #Electron transport limited part;
        BB = k2LL*i_inc + jmax;
        J = (BB-(BB**2-4*theta*jmax*k2LL*i_inc)**0.5)/(2*theta);
        x1J = J/4;
        x2J = 2*gamma_x;
        PJ = gm0*(x2J+gamma_x)+(x1J-rd)*R;
        QJ = (ci-gamma_x)*x1J-(ci+x2J)*rd;
        AAJ = (1.+R)*x2J + gamma_x + R*ci;
        BBJ = -((x2J+gamma_x)*(x1J-rd)+PJ*(ci+x2J)+R*QJ);
        CCJ = PJ*QJ;
        AJ = (-BBJ-(BBJ**2-4.*AAJ*CCJ)**0.5)/(2.*AAJ);
        #TPU limited part;
        AP = 3*TP-rd;
        A1 = np.minimum(AR,AJ)
        A_mod = np.minimum(A1,AP) 
        return A_mod

 
    def model_Vcmax_individual(self,xo,df_vcmax,sco,replicate):
        A_mod = self.calculate_A_individual(xo,df_vcmax,sco,replicate)
        return df_vcmax['A'].values - A_mod

    
    def plot_A_individual(self,xo,df_vcmax,replicate,sco):
       
        self.gas_exch_measurement.set_O2(0.21)      
        ACI = self.gas_exch_measurement.get_ACI_data()
        ACI = ACI[ACI['Replicate']==replicate]
        A = ACI['Net_CO2_assimilation_rate'].values 
        ci = ACI['Intercellular_CO2_concentration'].values
        i_inc = ACI['Irradiance'].values       
        A_mod_ci = self.calculate_A_individual_plot(xo,df_vcmax,replicate,sco,ci,i_inc)

        self.gas_exch_measurement.set_O2(0.21)      
        AI = self.gas_exch_measurement.get_AI_data()
        AI = AI[AI['Replicate']==replicate]
        A_i = AI['Net_CO2_assimilation_rate'].values 
        ci_i = AI['Intercellular_CO2_concentration'].values
        i_inc_i = AI['Irradiance'].values       
        A_mod_i = self.calculate_A_individual_plot(xo,df_vcmax,replicate,sco,ci_i,i_inc_i)        

        fig,(ax1,ax2) = plt.subplots(ncols=2,constrained_layout=True)        
        ax1.plot(ci,A,'ko',label='Expt.',markersize=8)
        ax1.plot(ci,A_mod_ci,'k-',label='Model')
        ax1.set_xlabel('Intercellular $CO_2$ (µmol $mol^{-1}$)',fontsize=20)
        ax1.set_ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)',fontsize=20)
        ax1.xaxis.set_ticks(np.arange(0, 2050, 250))        
        ax2.plot(i_inc_i,A_i,'ko',label='Expt.',markersize=8)
        ax2.plot(i_inc_i,A_mod_i,'k-',label='Model')
        ax2.set_xlabel('Irradiance (µmol $m^{-2}$ $s^{-1}$)',fontsize=20)
        ax2.set_ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)',fontsize=20)    
        ax2.legend(loc='lower right', fontsize='x-large')     
        ax2.xaxis.set_ticks(np.arange(0, 2200, 300))          
        
       
    def estimate_individual_Vcmax(self,inputs):
        AI = self.gas_exch_measurement.get_ACI_data()
        replicates = AI['Replicate'].unique()
        cols = ['Replicate','Vcmax','Vcmax_err','Tp','Tp_err','Sigma_gm','Sigma_gm_err']
        vcmax_values = pd.DataFrame([],columns=cols)
        count=0     
        for replicate in replicates:
            rds = inputs.get('Rd')
            rd = rds[count]*-1
            jmaxs = inputs.get('Jmax')
            jmax = jmaxs[count]
            thetas = inputs.get('Theta')
            theta = thetas[count]            
            k2LLs = inputs.get('k2LL')
            sco = inputs.get('Sco')            
            k2LL = k2LLs[count]    
            df_vcmax = self.get_vcmax_data_individual(rd,jmax,theta,k2LL,replicate)

#            bnds=((0,0,0),(1000,100,10)) 
            x0 = np.array([250,12,0.5])        #Vcmax, TP ,R     
#            result = optimize.least_squares(self.model_Vcmax_individual,x0,args=[rd,jmax,theta,k2LL,replicate,sco],method='trf',bounds=bnds)
            result = optimize.least_squares(self.model_Vcmax_individual,x0,args=[df_vcmax,sco,replicate],method='lm')            
            res = self.model_Vcmax_individual(result.x,df_vcmax,sco,replicate)
            J = np.array(result.jac)
            S = np.array(res).T.dot(np.array(res))
            H=2*J.T.dot(J);
            degfr=len(res)-3;
#            G=np.linalg.inv(H);
            G = np.linalg.pinv(H)
            var_1=2*S*G[0,0]/degfr;
            var_2=2*S*G[1,1]/degfr;
            var_3=2*S*G[2,2]/degfr;
            var_1 = np.sqrt(var_1)
            var_2 = np.sqrt(var_2)
            var_3 = np.sqrt(var_3)
            if result.success:
                self.plot_A_individual(result.x,df_vcmax,replicate,sco)
                df=pd.DataFrame([],columns=cols)
                df.loc[count,'Replicate'] = count                
                df.loc[count,'Vcmax'] = result.x[0]
                df.loc[count,'Tp'] = result.x[1]
                df.loc[count,'Sigma_gm'] = result.x[2]
                df.loc[count,'Vcmax_err'] = var_1
                df.loc[count,'Tp_err'] = var_2
                df.loc[count,'Sigma_gm_err'] = var_3
                vcmax_values = vcmax_values.append(df)
                count+=1
            else:
                raise ValueError(result.message)
                return []
        return vcmax_values   

    
    def calculate_A_mod_ave_vcmax_jmax(self,xo,rds,thetas,gms,k2LLs,sco,ci,i_inc):
        vcmax,TP,R,jmax = xo
        gm0=np.mean(gms)
        df_vcmax = self.get_vcmax_data(rds,0,thetas,k2LLs)
        rd = df_vcmax['Rd'].values.astype(float)
        rd = np.nanmean(rd)
        
        theta = df_vcmax['Theta'].values.astype(float)
        theta = np.nanmean(theta)
        
        k2LL = df_vcmax['k2LL'].values.astype(float)
        k2LL = np.nanmean(k2LLs)
#        
#        O = df_vcmax['O'].values.astype(float)
        O = 210 #mbar
        #sco = sco.astype(float)
        sco = np.nanmean(sco)
        
        gamma_x = 0.5*O/sco
#        gm0 = 0
#        R = 1;        
        #Rubisco-limited part;
        kmc = 267
        kmo = 164
        kmm = kmc*(1+O/kmo)
        x1R = vcmax
        x2R = kmm
        PR = gm0*(x2R+gamma_x)+(x1R-rd)*R
        QR = (ci-gamma_x)*x1R-(ci+x2R)*rd
        AAR = (1+R)*x2R + gamma_x + R*ci
        BBR = -((x2R+gamma_x)*(x1R-rd)+PR*(ci+x2R)+R*QR)
        CCR = PR*QR
        AR = (-BBR-(BBR**2-4.*AAR*CCR)**0.5)/(2*AAR)
        #Electron transport limited part;
        BB = k2LL*i_inc + jmax;
        J = (BB-(BB**2-4*theta*jmax*k2LL*i_inc)**0.5)/(2*theta);
        x1J = J/4;
        x2J = 2*gamma_x;
        PJ = gm0*(x2J+gamma_x)+(x1J-rd)*R;
        QJ = (ci-gamma_x)*x1J-(ci+x2J)*rd;
        AAJ = (1.+R)*x2J + gamma_x + R*ci;
        BBJ = -((x2J+gamma_x)*(x1J-rd)+PJ*(ci+x2J)+R*QJ);
        CCJ = PJ*QJ;
        AJ = (-BBJ-(BBJ**2-4.*AAJ*CCJ)**0.5)/(2.*AAJ);
        #TPU limited part;
        AP = 3*TP-rd;
        A1 = np.minimum(AR,AJ)
        A_mod = np.minimum(A1,AP) 
        return A_mod
    
    
    
    def plot_A_vcmax_jmax(self,xo,rds,thetas,gms,k2LLs,sco):
        plt.rcParams["figure.figsize"] = (15,8) 
        plt.rcParams.update({'font.size': 12})

        self.gas_exch_measurement.set_O2(0.21)      
        ACI = self.gas_exch_measurement.average_A_CI()
        A = ACI['Net_CO2_assimilation_rate'].values 
        A_err = ACI['Photo_err'].values/2 
        ci = ACI['Intercellular_CO2_concentration'].values
        i_inc = ACI['Irradiance'].values       
        A_mod_ci = self.calculate_A_mod_ave_vcmax_jmax(xo,rds,thetas,gms,k2LLs,sco,ci,i_inc)

        self.gas_exch_measurement.set_O2(0.21)      
        AI = self.gas_exch_measurement.average_A_I()
        A_i = AI['Net_CO2_assimilation_rate'].values 
        A_err_i = AI['Photo_err'].values/2 
        ci_i = AI['Intercellular_CO2_concentration'].values
        i_inc_i = AI['Irradiance'].values       
        A_mod_i = self.calculate_A_mod_ave_vcmax_jmax(xo,rds,thetas,gms,k2LLs,sco,ci_i,i_inc_i)
        
        fig,(ax1,ax2) = plt.subplots(ncols=2,constrained_layout=True)        
        ax1.errorbar(ci,A,A_err,fmt='ko',label='Expt.',mfc='white',mec='black',markersize=8)
        ax1.plot(ci,A_mod_ci,'k-',label='Model')
        ax1.set_xlabel('Intercellular $CO_2$ (µmol $mol^{-1}$)',fontsize=20)
        ax1.set_ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)',fontsize=20)
        ax1.xaxis.set_ticks(np.arange(0, 2050, 250))        
        ax2.errorbar(i_inc_i,A_i,A_err_i,fmt='ko',label='Expt.',mfc='white',mec='black',markersize=8)
        ax2.plot(i_inc_i,A_mod_i,'k-',label='Model')
        ax2.set_xlabel('Irradiance (µmol $m^{-2}$ $s^{-1}$)',fontsize=20)
        ax2.set_ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)',fontsize=20)    
        ax2.legend(loc='lower right', fontsize='x-large')     
        ax2.xaxis.set_ticks(np.arange(0, 2200, 300))            
        plt.show()  



    def calculate_A_vcmax_jmax(self,xo,rds,thetas,gms,k2LLs,sco):
        vcmax,TP,R,jmax = xo
        
        df_vcmax = self.get_vcmax_data(rds,0,thetas,k2LLs) #jmax = 0
       
        kmc = 267
        kmo = 164

        replicates = df_vcmax['Replicate'].unique()
        A_mod_arr = np.array([])
        for replicate in replicates:
            gm0=gms[replicate-1]
            df_vcmax_r=df_vcmax[df_vcmax['Replicate']==replicate];
            ci = df_vcmax_r['Ci'].values.astype(float)
            i_inc = df_vcmax_r['Iinc'].values.astype(float) 
            theta = df_vcmax_r['Theta'].values.astype(float)  
            O = df_vcmax_r['O'].values.astype(float)
            rds = df_vcmax_r['Rd'].values.astype(float)
            theta = df_vcmax_r['Theta'].values.astype(float)
            k2LLs = df_vcmax_r['k2LL'].values.astype(float)
        
            gamma_x = 0.5*O/sco
            
            kmm = kmc*(1+O/kmo)
            x1R = vcmax
            x2R = kmm
            
            rd = rds[replicate-1]
            if rd<0:
                rd=-1*rd
            k2LL = k2LLs[replicate-1]            
          
            #Rubisco-limited part;

            PR = gm0*(x2R+gamma_x)+(x1R-rd)*R
            QR = (ci-gamma_x)*x1R-(ci+x2R)*rd
            AAR = (1.+R)*x2R + gamma_x + R*ci
            BBR = -((x2R+gamma_x)*(x1R-rd)+PR*(ci+x2R)+R*QR)
            CCR = PR*QR
            AR = (-BBR-(BBR**2-4*AAR*CCR)**0.5)/(2*AAR)
            #Electron transport limited part;
            BB = k2LL*i_inc + jmax;
            J = (BB-(BB**2-4*theta*jmax*k2LL*i_inc)**0.5)/(2*theta);
            x1J = J/4;
            x2J = 2*gamma_x;
            PJ = gm0*(x2J+gamma_x)+(x1J-rd)*R;
            QJ = (ci-gamma_x)*x1J-(ci+x2J)*rd;
            AAJ = (1+R)*x2J + gamma_x + R*ci;
            BBJ = -((x2J+gamma_x)*(x1J-rd)+PJ*(ci+x2J)+R*QJ);
            CCJ = PJ*QJ;
            AJ = (-BBJ-(BBJ**2-4*AAJ*CCJ)**0.5)/(2*AAJ);
            #TPU limited part;
            AP = 3*TP-rd;
            A1 = np.minimum(AR,AJ)
            A_mod = np.minimum(A1,AP) 
            A_mod_arr = np.append(A_mod_arr,A_mod)
            
        return A_mod_arr

    
    def model_Vcmax_jmax(self,xo,df_vcmax,gms,sco):
        vcmax1,vcmax2,vcmax3,vcmax4,TP1,TP2,TP3,TP4,R,jmax1,jmax2,jmax3,jmax4 = xo
        
        vcmaxs =  [vcmax1,vcmax2,vcmax3,vcmax4]
        jmaxs =  [jmax1,jmax2,jmax3,jmax4]
        tps =  [TP1,TP2,TP3,TP4]
        
        kmc = 267
        kmo = 164

        replicates = df_vcmax['Replicate'].unique()
        A_mod_arr = np.array([])
        for replicate in replicates:
            gm0=gms[replicate-1]
            df_vcmax_r=df_vcmax[df_vcmax['Replicate']==replicate];
            ci = df_vcmax_r['Ci'].values.astype(float)
            i_inc = df_vcmax_r['Iinc'].values.astype(float) 
            theta = df_vcmax_r['Theta'].values.astype(float)  
            O = df_vcmax_r['O'].values.astype(float)
            rds = df_vcmax_r['Rd'].values.astype(float)
            k2LLs = df_vcmax_r['k2LL'].values.astype(float)
            vcmax =  vcmaxs[replicate-1]
            jmax =  jmaxs[replicate-1]
            TP =  tps[replicate-1]

            gamma_x = 0.5*O/sco
            
            kmm = kmc*(1+O/kmo)
            x1R = vcmax
            x2R = kmm
            
            rd = rds[replicate-1]
            if rd<0:
                rd=-1*rd
            k2LL = k2LLs[replicate-1]            
          
            #Rubisco-limited part;

            PR = gm0*(x2R+gamma_x)+(x1R-rd)*R
            QR = (ci-gamma_x)*x1R-(ci+x2R)*rd
            AAR = (1+R)*x2R + gamma_x + R*ci
            BBR = -((x2R+gamma_x)*(x1R-rd)+PR*(ci+x2R)+R*QR)
            CCR = PR*QR
            AR = (-BBR-(BBR**2-4*AAR*CCR)**0.5)/(2*AAR)
            
            #Electron transport limited part;
            BB = k2LL*i_inc + jmax;
            J = (BB-(BB**2-4*theta*jmax*k2LL*i_inc)**0.5)/(2*theta);
            x1J = J/4;
            x2J = 2*gamma_x;
            PJ = gm0*(x2J+gamma_x)+(x1J-rd)*R;
            QJ = (ci-gamma_x)*x1J-(ci+x2J)*rd;
            AAJ = (1+R)*x2J + gamma_x + R*ci;
            BBJ = -((x2J+gamma_x)*(x1J-rd)+PJ*(ci+x2J)+R*QJ);
            CCJ = PJ*QJ;
            AJ = (-BBJ-(BBJ**2-4*AAJ*CCJ)**0.5)/(2*AAJ);
            
            #TPU limited part;
            AP = 3*TP-rd;
            A1 = np.minimum(AR,AJ)
            A_mod = np.minimum(A1,AP) 
            A_mod_arr = np.append(A_mod_arr,A_mod)

        A = df_vcmax['A'].values    
        
        return A - A_mod_arr

    def estimate_Vcmax_Jmax(self,inputs):
        rd = inputs.get('Rd')
        rd=[-1*r for r in rd]   
        
        theta = inputs.get('Theta')
        k2LL = inputs.get('k2LL')
        sco = inputs.get('Sco')
        gms = inputs.get('gms')
        k2LL=[r for r in k2LL] 
        gms=[r for r in gms] 
        
        df_vcmax = self.get_vcmax_data(rd,0,theta,k2LL) #jmax = 0
        
        bnds=((50,50,50,50,8,8,8,8,0,100,100,100,100),(400,400,400,400,30,30,30,30,1,600,600,600,600)) 
        x0 = np.array([140,190,170,140,12,16,15,12,0.5,220,274,277,222])        # 4 values of Vcmax,TP,sigma,Jmax
    
        result = optimize.least_squares(self.model_Vcmax_jmax,x0,args=[df_vcmax,gms,sco],method='trf',bounds=bnds)
        # result = optimize.least_squares(self.model_Vcmax_jmax,x0,args=[df_vcmax,gms,sco],method='lm')
        
        res = self.model_Vcmax_jmax(result.x,df_vcmax,gms,sco)
        J = np.array(result.jac)
        S = np.array(res).T.dot(np.array(res))
        H=2*J.T.dot(J);
        degfr=len(res)-13;
        G = np.linalg.pinv(H)
        var_1=2*S*G[0,0]/degfr;
        var_2=2*S*G[1,1]/degfr;
        var_3=2*S*G[2,2]/degfr;
        var_4=2*S*G[3,3]/degfr;
        var_5=2*S*G[4,4]/degfr;
        var_6=2*S*G[5,5]/degfr;
        var_7=2*S*G[6,6]/degfr;
        var_8=2*S*G[7,7]/degfr;
        var_9=2*S*G[8,8]/degfr;
        var_10=2*S*G[9,9]/degfr;
        var_11=2*S*G[10,10]/degfr;
        var_12=2*S*G[11,11]/degfr;
        var_13=2*S*G[12,12]/degfr;
        
        var_1 = np.sqrt(var_1)
        var_2 = np.sqrt(var_2)
        var_3 = np.sqrt(var_3)
        var_4 = np.sqrt(var_4)
        var_5 = np.sqrt(var_5)
        var_6 = np.sqrt(var_6)
        var_7 = np.sqrt(var_7)
        var_8 = np.sqrt(var_8)
        var_9 = np.sqrt(var_9)
        var_10 = np.sqrt(var_10)
        var_11 = np.sqrt(var_11)
        var_12 = np.sqrt(var_12)
        var_13 = np.sqrt(var_13)

        if result.success:
            # self.plot_A_vcmax_jmax(result.x,rd,theta,gms,k2LL,sco)
            ind = ['vcmax1','vcmax2','vcmax3','vcmax4','Tp1','Tp2','Tp3','Tp4','R','Jmax1','Jmax2','Jmax3','Jmax4']
            df=pd.DataFrame([],columns=['estimate','err'],index=ind)
            df.loc['vcmax1','estimate'] = result.x[0]
            df.loc['vcmax2','estimate'] = result.x[1]
            df.loc['vcmax3','estimate'] = result.x[2]
            df.loc['vcmax4','estimate'] = result.x[3]
            
            df.loc['Tp1','estimate'] = result.x[4]
            df.loc['Tp2','estimate'] = result.x[5]
            df.loc['Tp3','estimate'] = result.x[6]
            df.loc['Tp4','estimate'] = result.x[7]

            df.loc['R','estimate'] = result.x[8]
            
            df.loc['Jmax1','estimate'] = result.x[9]
            df.loc['Jmax2','estimate'] = result.x[10]
            df.loc['Jmax3','estimate'] = result.x[11]
            df.loc['Jmax4','estimate'] = result.x[12]
            
            df.loc['vcmax1','err'] = var_1
            df.loc['vcmax2','err'] = var_2
            df.loc['vcmax3','err'] = var_3
            df.loc['vcmax4','err'] = var_4
            
            df.loc['Tp1','err'] = var_5
            df.loc['Tp2','err'] = var_6
            df.loc['Tp3','err'] = var_7
            df.loc['Tp4','err'] = var_8

            df.loc['R','err'] = var_9
            
            df.loc['Jmax1','err'] = var_10
            df.loc['Jmax2','err'] = var_11
            df.loc['Jmax3','err'] = var_12
            df.loc['Jmax4','err'] = var_13
            
            return df
        else:
            raise ValueError(result.message)
            return []  




    def calculate_A_const_gm(self,xo,df_vcmax,sco):
        vcmax,TP,gm0 = xo
        
        rd = df_vcmax['Rd'].values.astype(float)
        jmax = df_vcmax['Jmax'].values.astype(float)
        theta = df_vcmax['Theta'].values.astype(float)
        k2LL = df_vcmax['k2LL'].values.astype(float)
        ci = df_vcmax['Ci'].values.astype(float)
        i_inc = df_vcmax['Iinc'].values.astype(float)
        O = df_vcmax['O'].values.astype(float)
        
#        O = 210 #mbar
        sco = sco.astype(float)
        gamma_x = 0.5*O/sco
        R = 0
        #Rubisco-limited part;
        kmc = 267
        kmo = 164
        kmm = kmc*(1+O/kmo)
        x1R = vcmax
        x2R = kmm
        PR = gm0*(x2R+gamma_x)+(x1R-rd)*R
        QR = (ci-gamma_x)*x1R-(ci+x2R)*rd
        AAR = (1.+R)*x2R + gamma_x + R*ci
        BBR = -((x2R+gamma_x)*(x1R-rd)+PR*(ci+x2R)+R*QR)
        CCR = PR*QR
        AR = (-BBR-(BBR**2-4*AAR*CCR)**0.5)/(2*AAR)
        #Electron transport limited part;
        BB = k2LL*i_inc + jmax;
        J = (BB-(BB**2-4*theta*jmax*k2LL*i_inc)**0.5)/(2*theta);
        x1J = J/4;
        x2J = 2*gamma_x;
        PJ = gm0*(x2J+gamma_x)+(x1J-rd)*R;
        QJ = (ci-gamma_x)*x1J-(ci+x2J)*rd;
        AAJ = (1+R)*x2J + gamma_x + R*ci;
        BBJ = -((x2J+gamma_x)*(x1J-rd)+PJ*(ci+x2J)+R*QJ);
        CCJ = PJ*QJ;
        AJ = (-BBJ-(BBJ**2-4*AAJ*CCJ)**0.5)/(2*AAJ);
        #TPU limited part;
        AP = 3*TP-rd;
        A1 = np.minimum(AR,AJ)
        A_mod = np.minimum(A1,AP) 
        
        return A_mod
 

    def calculate_A_mod_ave_const_gm(self,xo,rds,jmaxs,thetas,k2LLs,sco,ci,i_inc):
        vcmax,TP,gm0 = xo
        
        df_vcmax = self.get_vcmax_data(rds,jmaxs,thetas,k2LLs)
        rd = df_vcmax['Rd'].values.astype(float)
        rd = np.nanmean(rd)
        jmax = df_vcmax['Jmax'].values.astype(float)
        jmax = np.nanmean(jmax)
        
        theta = df_vcmax['Theta'].values.astype(float)
        theta = np.nanmean(theta)
        
        k2LL = df_vcmax['k2LL'].values.astype(float)
        k2LL = np.nanmean(k2LLs)
#        
#        O = df_vcmax['O'].values.astype(float)
        O = 210 #mbar
        sco = sco.astype(float)
        sco = np.nanmean(sco)
        
        gamma_x = 0.5*O/sco
        R = 0
        #Rubisco-limited part;
        kmc = 267
        kmo = 164
        kmm = kmc*(1+O/kmo)
        x1R = vcmax
        x2R = kmm
        PR = gm0*(x2R+gamma_x)+(x1R-rd)*R
        QR = (ci-gamma_x)*x1R-(ci+x2R)*rd
        AAR = (1+R)*x2R + gamma_x + R*ci
        BBR = -((x2R+gamma_x)*(x1R-rd)+PR*(ci+x2R)+R*QR)
        CCR = PR*QR
        AR = (-BBR-(BBR**2-4.*AAR*CCR)**0.5)/(2*AAR)
        #Electron transport limited part;
        BB = k2LL*i_inc + jmax;
        J = (BB-(BB**2-4*theta*jmax*k2LL*i_inc)**0.5)/(2*theta);
        x1J = J/4;
        x2J = 2*gamma_x;
        PJ = gm0*(x2J+gamma_x)+(x1J-rd)*R;
        QJ = (ci-gamma_x)*x1J-(ci+x2J)*rd;
        AAJ = (1.+R)*x2J + gamma_x + R*ci;
        BBJ = -((x2J+gamma_x)*(x1J-rd)+PJ*(ci+x2J)+R*QJ);
        CCJ = PJ*QJ;
        AJ = (-BBJ-(BBJ**2-4.*AAJ*CCJ)**0.5)/(2.*AAJ);
        #TPU limited part;
        AP = 3*TP-rd;
        A1 = np.minimum(AR,AJ)
        A_mod = np.minimum(A1,AP) 
        
        gm_mod = ( A_mod + R*(A_mod+rd))/(ci-gamma_x) 
        # A_mod = np.array([A_mod]).transpose()
        # A_mod = np.reshape(A_mod,(4,len(ci)))
        # J = np.array([J]).transpose()
        # J = np.reshape(J,(4,len(ci)))
        gm_mod = np.array([gm_mod]).transpose()
        # gm_mod = np.reshape(gm_mod,(4,len(ci)))
        return [A_mod,gm_mod,J]


    def plot_const_gm(self,xo,rds,jmaxs,thetas,k2LLs,sco):
        plt.rcParams["figure.figsize"] = (15,8) 
        plt.rcParams.update({'font.size': 12})

        self.gas_exch_measurement.set_O2(0.21)      
        ACI = self.gas_exch_measurement.average_A_CI()
        A = ACI['Net_CO2_assimilation_rate'].values 
        ci = ACI['Intercellular_CO2_concentration'].values
        i_inc = ACI['Irradiance'].values   
        gamma_x = 0.5*210/sco
        [A_mod_ci,gm_ci_cf,J] = self.calculate_A_mod_ave_const_gm(xo,rds,jmaxs,thetas,k2LLs,sco,ci,i_inc)

        Rd = np.mean(rds)
        AA = gamma_x*(J + 8*(A + Rd))
        gm_ci =  A/(ci-AA/(J-4*(A+Rd)));

        self.gas_exch_measurement.set_O2(0.21)      
        AI = self.gas_exch_measurement.average_A_I()
        A_i = AI['Net_CO2_assimilation_rate'].values 
        ci_i = AI['Intercellular_CO2_concentration'].values
        i_inc_i = AI['Irradiance'].values       
        [A_mod_i,gm_i_cf,J] = self.calculate_A_mod_ave_const_gm(xo,rds,jmaxs,thetas,k2LLs,sco,ci_i,i_inc_i)
        AA = gamma_x*(J + 8*(A_i + Rd))
        gm_i =  A_i/(ci_i-AA/(J-4*(A_i+Rd)));
        
        fig,(ax1,ax2) = plt.subplots(ncols=2,constrained_layout=True)        
        ax1.plot(ci,gm_ci_cf,'ko',label='CF.',mfc='white',mec='black',markersize=8)
        ax1.plot(ci,gm_ci,'k-',label='Model')
        ax1.set_xlabel('Intercellular $CO_2$ (µmol $mol^{-1}$)',fontsize=20)
        ax1.set_ylabel('$g_m$ (mol $m^{-2}$ $s^{-1}$ $bar^{-1}$)',fontsize=20)
        ax1.xaxis.set_ticks(np.arange(0, 2050, 250))        
        ax2.plot(i_inc_i,gm_i_cf,'ko',label='CF.',mfc='white',mec='black',markersize=8)
        ax2.plot(i_inc_i,gm_i,'k-',label='Model')
        ax2.set_xlabel('Irradiance (µmol $m^{-2}$ $s^{-1}$)',fontsize=20)
        ax2.set_ylabel('$g_m$ (mol $m^{-2}$ $s^{-1}$ $bar^{-1}$)',fontsize=20)    
        ax2.legend(loc='lower right', fontsize='x-large')     
        ax2.xaxis.set_ticks(np.arange(0, 2200, 300))   
            
        plt.show()  
        
        
    def plot_A_const_gm(self,xo,rds,jmaxs,thetas,k2LLs,sco):
        plt.rcParams["figure.figsize"] = (15,8) 
        plt.rcParams.update({'font.size': 12})

        self.gas_exch_measurement.set_O2(0.21)      
        ACI = self.gas_exch_measurement.average_A_CI()
        A = ACI['Net_CO2_assimilation_rate'].values 
        A_err = ACI['Photo_err'].values/2 
        ci = ACI['Intercellular_CO2_concentration'].values
        i_inc = ACI['Irradiance'].values       
        [A_mod_ci,gm_ci,J] = self.calculate_A_mod_ave_const_gm(xo,rds,jmaxs,thetas,k2LLs,sco,ci,i_inc)

        self.gas_exch_measurement.set_O2(0.21)      
        AI = self.gas_exch_measurement.average_A_I()
        A_i = AI['Net_CO2_assimilation_rate'].values 
        A_err_i = AI['Photo_err'].values/2 
        ci_i = AI['Intercellular_CO2_concentration'].values
        i_inc_i = AI['Irradiance'].values       
        [A_mod_i,gm_i,J] = self.calculate_A_mod_ave_const_gm(xo,rds,jmaxs,thetas,k2LLs,sco,ci_i,i_inc_i)
        
        fig,(ax1,ax2) = plt.subplots(ncols=2,constrained_layout=True)        
        ax1.errorbar(ci,A,A_err,fmt='ko',label='Expt.',mfc='white',mec='black',markersize=8)
        ax1.plot(ci,A_mod_ci,'k-',label='Model')
        ax1.set_xlabel('Intercellular $CO_2$ (µmol $mol^{-1}$)',fontsize=20)
        ax1.set_ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)',fontsize=20)
        ax1.xaxis.set_ticks(np.arange(0, 2050, 250))        
        ax2.errorbar(i_inc_i,A_i,A_err_i,fmt='ko',label='Expt.',mfc='white',mec='black',markersize=8)
        ax2.plot(i_inc_i,A_mod_i,'k-',label='Model')
        ax2.set_xlabel('Irradiance (µmol $m^{-2}$ $s^{-1}$)',fontsize=20)
        ax2.set_ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)',fontsize=20)    
        ax2.legend(loc='lower right', fontsize='x-large')     
        ax2.xaxis.set_ticks(np.arange(0, 2200, 300))
        plt.show()  

    
    def model_Vcmax_const_gm(self,xo,df_vcmax,sco):
        A_mod = self.calculate_A_const_gm(xo,df_vcmax,sco)
        return df_vcmax['A'].values - A_mod
    
    
    def estimate_Vcmax_constant_gm(self,inputs):
        rd = inputs.get('Rd')*-1
        jmax = inputs.get('Jmax')
        theta = inputs.get('Theta')
        k2LL = inputs.get('k2LL')
        sco = inputs.get('Sco')
        
        df_vcmax = self.get_vcmax_data(rd,jmax,theta,k2LL)
        
        bnds=((0,0,0.1),(1000,70,20)) 
        x0 = np.array([200,23,1.5])        #Vcmax,TP,sigma,Jmax        
        result = optimize.least_squares(self.model_Vcmax_const_gm,x0,args=[df_vcmax,sco],method='trf',bounds=bnds)
#        result = optimize.least_squares(self.model_Vcmax_const_gm,x0,args=[df_vcmax,sco],method='lm')
        
        res = self.model_Vcmax_const_gm(result.x,df_vcmax,sco)
        J = np.array(result.jac)
        S = np.array(res).T.dot(np.array(res))
        H=2*J.T.dot(J);
        degfr=len(res)-3;
        G = np.linalg.pinv(H)
        var_1=2*S*G[0,0]/degfr;
        var_2=2*S*G[1,1]/degfr;
        var_3=2*S*G[2,2]/degfr;
        
        var_1 = np.sqrt(var_1)
        var_2 = np.sqrt(var_2)
        var_3 = np.sqrt(var_3)

        if result.success:
            self.plot_A_const_gm(result.x,rd,jmax,theta,k2LL,sco)
            self.plot_const_gm(result.x,rd,jmax,theta,k2LL,sco)
            cols = ['Vcmax','Vcmax_err','Tp','Tp_err','gm0','gm0_err']
            
            df=pd.DataFrame([],columns=cols)
            df.loc[0,'Vcmax'] = result.x[0]
            df.loc[0,'Tp'] = result.x[1]
            df.loc[0,'gm0'] = result.x[2]
            
            df.loc[0,'Vcmax_err'] = var_1
            df.loc[0,'Tp_err'] = var_2
            df.loc[0,'gm0_err'] = var_3
            
            return df
        else:
            raise ValueError(result.message)
            return []        


    def calculate_A_var_gm(self,xo,df_vcmax,sco):
        vcmax,TP,R = xo
        
        rd = df_vcmax['Rd'].values.astype(float)
        jmax = df_vcmax['Jmax'].values.astype(float)
        theta = df_vcmax['Theta'].values.astype(float)
        k2LL = df_vcmax['k2LL'].values.astype(float)
        ci = df_vcmax['Ci'].values.astype(float)
        i_inc = df_vcmax['Iinc'].values.astype(float)
        O = df_vcmax['O'].values.astype(float)
#        O = 210 #mbar
        sco = sco.astype(float)
        gamma_x = 0.5*O/sco
        gm0 = 0
        #Rubisco-limited part;
        kmc = 267
        kmo = 164
        kmm = kmc*(1+O/kmo)
        x1R = vcmax
        x2R = kmm
        PR = gm0*(x2R+gamma_x)+(x1R-rd)*R
        QR = (ci-gamma_x)*x1R-(ci+x2R)*rd
        AAR = (1.+R)*x2R + gamma_x + R*ci
        BBR = -((x2R+gamma_x)*(x1R-rd)+PR*(ci+x2R)+R*QR)
        CCR = PR*QR
        AR = (-BBR-(BBR**2-4*AAR*CCR)**0.5)/(2*AAR)
        #Electron transport limited part;
        BB = k2LL*i_inc + jmax;
        J = (BB-(BB**2-4*theta*jmax*k2LL*i_inc)**0.5)/(2*theta);
        x1J = J/4;
        x2J = 2*gamma_x;
        PJ = gm0*(x2J+gamma_x)+(x1J-rd)*R;
        QJ = (ci-gamma_x)*x1J-(ci+x2J)*rd;
        AAJ = (1+R)*x2J + gamma_x + R*ci;
        BBJ = -((x2J+gamma_x)*(x1J-rd)+PJ*(ci+x2J)+R*QJ);
        CCJ = PJ*QJ;
        AJ = (-BBJ-(BBJ**2-4*AAJ*CCJ)**0.5)/(2*AAJ);
        #TPU limited part;
        AP = 3*TP-rd;
        A1 = np.minimum(AR,AJ)
        A_mod = np.minimum(A1,AP) 
        
        return A_mod
 

    def calculate_A_mod_ave_var_gm(self,xo,rds,jmaxs,thetas,k2LLs,sco,ci,i_inc):
        vcmax,TP,R = xo
        
        df_vcmax = self.get_vcmax_data(rds,jmaxs,thetas,k2LLs)
        rd = df_vcmax['Rd'].values.astype(float)
        rd = np.nanmean(rd)
        jmax = df_vcmax['Jmax'].values.astype(float)
        jmax = np.nanmean(jmax)
        
        theta = df_vcmax['Theta'].values.astype(float)
        theta = np.nanmean(theta)
        
        k2LL = df_vcmax['k2LL'].values.astype(float)
        k2LL = np.nanmean(k2LLs)
#        
#        O = df_vcmax['O'].values.astype(float)
        O = 210 #mbar
        sco = sco.astype(float)
        sco = np.nanmean(sco)
        
        gamma_x = 0.5*O/sco
        gm0 = 0
        #Rubisco-limited part;
        kmc = 267
        kmo = 164
        kmm = kmc*(1+O/kmo)
        x1R = vcmax
        x2R = kmm
        PR = gm0*(x2R+gamma_x)+(x1R-rd)*R
        QR = (ci-gamma_x)*x1R-(ci+x2R)*rd
        AAR = (1+R)*x2R + gamma_x + R*ci
        BBR = -((x2R+gamma_x)*(x1R-rd)+PR*(ci+x2R)+R*QR)
        CCR = PR*QR
        AR = (-BBR-(BBR**2-4.*AAR*CCR)**0.5)/(2*AAR)
        #Electron transport limited part;
        BB = k2LL*i_inc + jmax;
        J = (BB-(BB**2-4*theta*jmax*k2LL*i_inc)**0.5)/(2*theta);
        x1J = J/4;
        x2J = 2*gamma_x;
        PJ = gm0*(x2J+gamma_x)+(x1J-rd)*R;
        QJ = (ci-gamma_x)*x1J-(ci+x2J)*rd;
        AAJ = (1.+R)*x2J + gamma_x + R*ci;
        BBJ = -((x2J+gamma_x)*(x1J-rd)+PJ*(ci+x2J)+R*QJ);
        CCJ = PJ*QJ;
        AJ = (-BBJ-(BBJ**2-4.*AAJ*CCJ)**0.5)/(2.*AAJ);
        #TPU limited part;
        AP = 3*TP-rd;
        A1 = np.minimum(AR,AJ)
        A_mod = np.minimum(A1,AP) 
        
        gm_mod = ( A_mod + R*(A_mod+rd))/(ci-gamma_x) 
        
        return [A_mod,gm_mod,J]


    def plot_var_gm(self,xo,rds,jmaxs,thetas,k2LLs,sco):
        plt.rcParams["figure.figsize"] = (15,8) 
        plt.rcParams.update({'font.size': 12})

        self.gas_exch_measurement.set_O2(0.21)      
        ACI = self.gas_exch_measurement.average_A_CI()
        A = ACI['Net_CO2_assimilation_rate'].values 
        ci = ACI['Intercellular_CO2_concentration'].values*constants.atm/10**5
        i_inc = ACI['Irradiance'].values   
        gamma_x = 0.5*210/sco
        [A_mod_ci,gm_ci_cf,J] = self.calculate_A_mod_ave_var_gm(xo,rds,jmaxs,thetas,k2LLs,sco,ci,i_inc)

        Rd = np.mean(rds)
        AA = gamma_x*(J + 8*(A + Rd))
        gm_ci =  A/(ci-AA/(J-4*(A+Rd)));

        self.gas_exch_measurement.set_O2(0.21)      
        AI = self.gas_exch_measurement.average_A_I()
        A_i = AI['Net_CO2_assimilation_rate'].values 
        ci_i = AI['Intercellular_CO2_concentration'].values*constants.atm/10**5
        i_inc_i = AI['Irradiance'].values       
        [A_mod_i,gm_i_cf,J] = self.calculate_A_mod_ave_var_gm(xo,rds,jmaxs,thetas,k2LLs,sco,ci_i,i_inc_i)
        AA = gamma_x*(J + 8*(A_i + Rd))
        gm_i =  A_i/(ci_i-AA/(J-4*(A_i+Rd)));
        
        fig,(ax1,ax2) = plt.subplots(ncols=2,constrained_layout=True)        
        ax1.plot(ci,gm_ci,'ko',label='Variable J',mfc='white',mec='black',markersize=8)
        ax1.plot(ci,gm_ci_cf,'k-',label='Model')
        ax1.set_xlabel('Intercellular $CO_2$ (µmol $mol^{-1}$)',fontsize=20)
        ax1.set_ylabel('$g_m$ (mol $m^{-2}$ $s^{-1}$ $bar^{-1}$)',fontsize=20)
        ax1.xaxis.set_ticks(np.arange(0, 2050, 250))        
        ax2.plot(i_inc_i,gm_i,'ko',label='Variable J',mfc='white',mec='black',markersize=8)
        ax2.plot(i_inc_i,gm_i_cf,'k-',label='Model')
        ax2.set_xlabel('Irradiance (µmol $m^{-2}$ $s^{-1}$)',fontsize=20)
        ax2.set_ylabel('$g_m$ (mol $m^{-2}$ $s^{-1}$ $bar^{-1}$)',fontsize=20)    
        ax2.legend(loc='lower right', fontsize='x-large')     
        ax2.xaxis.set_ticks(np.arange(0, 2200, 300))   
            
        plt.show()  
        
        
    def plot_A_var_gm(self,xo,rds,jmaxs,thetas,k2LLs,sco):
        plt.rcParams["figure.figsize"] = (15,8) 
        plt.rcParams.update({'font.size': 12})

        self.gas_exch_measurement.set_O2(0.21)      
        ACI = self.gas_exch_measurement.average_A_CI()
        A = ACI['Net_CO2_assimilation_rate'].values 
        A_err = ACI['Photo_err'].values/2 
        ci = ACI['Intercellular_CO2_concentration'].values*constants.atm/10**5
        i_inc = ACI['Irradiance'].values       
        [A_mod_ci,gm_ci,J] = self.calculate_A_mod_ave_var_gm(xo,rds,jmaxs,thetas,k2LLs,sco,ci,i_inc)

        self.gas_exch_measurement.set_O2(0.21)      
        AI = self.gas_exch_measurement.average_A_I()
        A_i = AI['Net_CO2_assimilation_rate'].values 
        A_err_i = AI['Photo_err'].values/2 
        ci_i = AI['Intercellular_CO2_concentration'].values*constants.atm/10**5
        i_inc_i = AI['Irradiance'].values       
        [A_mod_i,gm_i,J] = self.calculate_A_mod_ave_var_gm(xo,rds,jmaxs,thetas,k2LLs,sco,ci_i,i_inc_i)
        
        fig,(ax1,ax2) = plt.subplots(ncols=2,constrained_layout=True)        
        ax1.errorbar(ci,A,A_err,fmt='ko',label='Expt.',mfc='white',mec='black',markersize=8)
        ax1.plot(ci,A_mod_ci,'k-',label='Model')
        ax1.set_xlabel('Intercellular $CO_2$ (µmol $mol^{-1}$)',fontsize=20)
        ax1.set_ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)',fontsize=20)
        ax1.xaxis.set_ticks(np.arange(0, 2050, 250))        
        ax2.errorbar(i_inc_i,A_i,A_err_i,fmt='ko',label='Expt.',mfc='white',mec='black',markersize=8)
        ax2.plot(i_inc_i,A_mod_i,'k-',label='Model')
        ax2.set_xlabel('Irradiance (µmol $m^{-2}$ $s^{-1}$)',fontsize=20)
        ax2.set_ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)',fontsize=20)    
        ax2.legend(loc='lower right', fontsize='x-large')     
        ax2.xaxis.set_ticks(np.arange(0, 2200, 300))
        plt.show()  

    
    def model_Vcmax_var_gm(self,xo,df_vcmax,sco):
        A_mod = self.calculate_A_var_gm(xo,df_vcmax,sco)
        return df_vcmax['A'].values - A_mod
    
    
    def estimate_Vcmax_var_gm(self,inputs):
        rd = inputs.get('Rd')*-1
        jmax = inputs.get('Jmax')
        theta = inputs.get('Theta')
        k2LL = inputs.get('k2LL')
        sco = inputs.get('Sco')
        
        df_vcmax = self.get_vcmax_data(rd,jmax,theta,k2LL)

        
        bnds=((0,0,0),(1000,70,20)) 
        x0 = np.array([200,12,0.5])        #Vcmax,TP,sigma        
        result = optimize.least_squares(self.model_Vcmax_var_gm,x0,args=[df_vcmax,sco],method='trf',bounds=bnds)
        # result = optimize.least_squares(self.model_Vcmax_const_gm,x0,args=[rd,jmax,theta,k2LL,sco],method='lm')
        
        res = self.model_Vcmax_var_gm(result.x,df_vcmax,sco)
        J = np.array(result.jac)
        S = np.array(res).T.dot(np.array(res))
        H=2*J.T.dot(J);
        degfr=len(res)-3;
        G = np.linalg.pinv(H)
        var_1=2*S*G[0,0]/degfr;
        var_2=2*S*G[1,1]/degfr;
        var_3=2*S*G[2,2]/degfr;
        
        var_1 = np.sqrt(var_1)
        var_2 = np.sqrt(var_2)
        var_3 = np.sqrt(var_3)

        if result.success:
            self.plot_A_var_gm(result.x,rd,jmax,theta,k2LL,sco)
            self.plot_var_gm(result.x,rd,jmax,theta,k2LL,sco)
            cols = ['Vcmax','Vcmax_err','Tp','Tp_err','Sigma_gm','Sigma_gm_err']
            
            df=pd.DataFrame([],columns=cols)
            df.loc[0,'Vcmax'] = result.x[0]
            df.loc[0,'Tp'] = result.x[1]
            df.loc[0,'Sigma_gm'] = result.x[2]
            
            df.loc[0,'Vcmax_err'] = var_1
            df.loc[0,'Tp_err'] = var_2
            df.loc[0,'Sigma_gm_err'] = var_3
            
            return df
        else:
            raise ValueError(result.message)
            return []        
        
            
    def model_NRH_A(self,xo,rd):
        gm,lump=xo;
        self.gas_exch_measurement.set_O2(0.21) 
        ACI = self.gas_exch_measurement.get_ACI_data()
        ACI = ACI[ACI['CO2R']>400]
        A = ACI['Net_CO2_assimilation_rate'].values 

        ci = ACI['Intercellular_CO2_concentration'].values
        i_inc = ACI['Irradiance'].values
        phi2 = ACI['PhiPS2'].values
        O = 210;
        sco = 3.259;
        gammax=0.5*O/sco;
    
        X1 = lump*i_inc*phi2/4;
        Z = 2;
        BBJ = (X1-rd)/gm+Z*gammax+ci;
        SQJ = (BBJ**2-4/gm*((ci-gammax)*X1-(Z*gammax+ci)*rd))**0.5;
        A_mod = (BBJ-SQJ)/(2/gm);
            
        return A - A_mod
        
    def NRH_A_gm(self,rd):
        
        # bnds=((0,0,0,0,0),(10,10,10,10,1)) 
        # x0 = np.array([0.5,0.5,0.5,0.5,0.4])        #gm,s'
        bnds=((0,0),(10,1)) 
        x0 = np.array([0.5,0.4])                
        # result = optimize.least_squares(self.model_NRH_A,x0,args=[rd],method='trf',bounds=bnds)
        result = optimize.least_squares(self.model_NRH_A,x0,args=[-1*rd],method='lm')
        
        res = self.model_NRH_A(result.x,rd)
        J = np.array(result.jac)
        S = np.array(res).T.dot(np.array(res))
        H=2*J.T.dot(J);
        degfr=len(res)-2;
        G = np.linalg.pinv(H)
        var_1=2*S*G[0,0]/degfr;
        var_2=2*S*G[1,1]/degfr;

        
        var_1 = np.sqrt(var_1)
        var_2 = np.sqrt(var_2)

                        
        if result.success:
            
            df=pd.DataFrame([],columns=['estimate','err'],index=['gm','lump'])
            
            df.loc['gm','estimate'] = result.x[0]
            df.loc['lump','estimate'] = result.x[1]
            
            df.loc['gm','err'] = var_1
            df.loc['lump','err'] = var_2
           
            return df
        else:
            raise ValueError(result.message)
            return []       
  
    
    def get_vcmax_data_Bush(self,Rd,Jmax,Theta,K2LL):
        cols = ['Replicate','Rd','Theta','Jmax','k2LL','Ci','Iinc','A','O','curve']  # RD THETA JMAX K2LL CI IINC A;
        vcmax_data = pd.DataFrame([],columns = cols)
        O2 = [210,20]
        for O in O2:
            self.gas_exch_measurement.set_O2(O/1000)
            ACIH = self.gas_exch_measurement.get_ACI_data()
            replicates = ACIH['Replicate'].unique()
            df_vcmax = pd.DataFrame([],columns = cols)
            count = 0 
            for replicate in replicates:
                df = pd.DataFrame([],columns = cols)
                ACIH_r = ACIH[ACIH['Replicate']==replicate]
                ci = ACIH_r['Intercellular_CO2_concentration'].values*constants.atm/10**5  
                i_inc = ACIH_r['Irradiance'].values
                A = ACIH_r['Net_CO2_assimilation_rate'].values  

                rd = Rd[replicate-1]
                theta = Theta[replicate-1]
                jmax = Jmax[replicate-1]               
                df.loc[:,'A'] = A                        
                df.loc[:,'Rd'] = rd*-1
                df.loc[:,'Theta'] = theta
                df.loc[:,'Jmax'] = jmax
                df.loc[:,'k2LL'] = K2LL
                df.loc[:,'Ci'] = ci
                df.loc[:,'Iinc'] = i_inc
                df.loc[:,'O'] = O    
                df.loc[:,'curve'] = 'ACI'    
                df.loc[:,'Replicate']=replicate    
                       
                df_vcmax = df_vcmax.append(df)
                count+=1
            vcmax_data = vcmax_data.append(df_vcmax)   
            
        return vcmax_data

    
    
    def calculate_A_Bush(self,xo,df_vcmax):
        
        """
        FvCB model of photosynthesis extended to include nitrogen assimilation
        Yin et al. 2021 10.1111/pce.14070 
        Estimation procedure is explianed also in Bush et al. 2018 10.1038/s41477-017-0065-x
        
        """
        
        alphaGMax, alphaSMax, Nmax,vcmax1,vcmax2,vcmax3,vcmax4,TP1,TP2,TP3,TP4 = xo
        vcmaxs=[vcmax1,vcmax2,vcmax3,vcmax4]
        tps=[TP1,TP2,TP3,TP4]
        
        kmc = 267
        kmo = 164

        sco = 3.259;
        alphaT = 0;
            
        replicates = df_vcmax['Replicate'].unique()
        A_mod_arr = np.array([])
        for replicate in replicates:
            df_vcmax_r=df_vcmax[df_vcmax['Replicate']==replicate];
            ci = df_vcmax_r['Ci'].values.astype(float)
            i_inc = df_vcmax_r['Iinc'].values.astype(float)
            theta = df_vcmax_r['Theta'].values.astype(float)  
            O = df_vcmax_r['O'].values.astype(float)
            rd = df_vcmax_r['Rd'].values.astype(float)
            jmax = df_vcmax_r['Jmax'].values.astype(float)
            
            k2LL = df_vcmax_r['k2LL'].values.astype(float)     
            vcmax =  vcmaxs[replicate-1]
            # jmax =  jmaxs[replicate-1]
            TP =  tps[replicate-1]

            # Rubisco-limited part
            phi = O/(sco*ci);
            kmcmo = kmc*(1+O/kmo);
            wc = vcmax*ci/(ci+kmcmo);
            voc = phi*wc;
            beta = 3*alphaGMax/(3*alphaGMax+2*alphaSMax);
            alphaG = np.minimum(alphaGMax,Nmax*beta/voc);
            alphaS = np.minimum(alphaSMax,3*Nmax*(1-beta)/(2*voc));
            gammax = (0.5*(1-alphaG)+alphaT)*O/sco;
            AR = wc*(1-gammax/ci)-rd;
            
            # Electron transport limited part;
            BB = k2LL*i_inc + jmax;

            J = (BB-(BB**2-4*theta*jmax*k2LL*i_inc)**0.5)/(2*theta);
            if J[0] > Nmax*(2*beta + 6):
                alphaG = np.minimum(alphaGMax,4*Nmax*beta*(1/phi+1)/(J-Nmax*(2*beta+6)));
                alphaS = np.minimum(alphaSMax,6*Nmax*(1-beta)*(1/phi+1)/(J-Nmax*(2*beta+6)));
            else:
                alphaG=alphaGMax;
                alphaS=alphaSMax;
                
            gammax = (0.5*(1-alphaG)+alphaT)*O/sco;
            wj = J/(phi*(4+8*alphaG+4*alphaS));
            AJ = wj*(1-gammax/ci)-rd;
            
            # TPU limited part;
            wp = 3*TP/(1-0.5*(1+3*alphaG+4*alphaS)*phi);
            alphaG = np.minimum(alphaGMax,Nmax*beta*(2/phi-1)/(6*TP+3*Nmax*(2-beta)));
            alphaS = np.minimum(alphaSMax,1.5*Nmax*(1-beta)*(2/phi-1)/(6*TP+3*Nmax*(2-beta)));
            gammax = (0.5*(1-alphaG)+alphaT)*O/sco;
            AP = wp*(1-gammax/ci)-rd;
        
            A1 = np.minimum(AR,AJ)
            A_mod = np.minimum(A1,AP) 
            A_mod_arr = np.append(A_mod_arr,A_mod)

        return A_mod_arr


    def model_Vcmax_Bush(self,xo,df_vcmax):
        A_mod = self.calculate_A_Bush(xo,df_vcmax)    
        return df_vcmax['A'].values - A_mod

    
    def estimate_Vcmax_Bush(self,inputs):
        
        rds = inputs['Rd']
        jmaxs = inputs['Jmax']
        thetas = inputs['Theta']
        k2LL = inputs['k2LL']        
        df_vcmax = self.get_vcmax_data_Bush(rds,jmaxs,thetas,k2LL)
      
        #alphaGMax, alphaSMax, Nmax, Vcmax1-4,TP1-4
        bnds=((0.0001,0,0,0,0,0,0,0,0,0,0),(1,1,20,500,500,500,500,50,50,50,50)) 
        x0 = np.array([0.05,0.04,1.41,180,180,180,180,25,25,25,25])                
        result = optimize.least_squares(self.model_Vcmax_Bush,x0,args=[df_vcmax],method='trf',bounds=bnds)
        # result = optimize.least_squares(self.model_Vcmax_Bush,x0,args=[inputs],method='lm')
        
        res = self.model_Vcmax_Bush(result.x,df_vcmax)
        J = np.array(result.jac)
        S = np.array(res).T.dot(np.array(res))
        H=2*J.T.dot(J);
        degfr=len(res)-11;
        G = np.linalg.pinv(H)
        var_1=2*S*G[0,0]/degfr;
        var_2=2*S*G[1,1]/degfr;
        var_3=2*S*G[2,2]/degfr;
        var_4=2*S*G[3,3]/degfr;
        var_5=2*S*G[4,4]/degfr;
        var_6=2*S*G[5,5]/degfr;
        var_7=2*S*G[6,6]/degfr;
        var_8=2*S*G[7,7]/degfr;
        var_9=2*S*G[8,8]/degfr;
        var_10=2*S*G[9,9]/degfr;
        var_11=2*S*G[10,10]/degfr;
        
        var_1 = np.sqrt(var_1)
        var_2 = np.sqrt(var_2)
        var_3 = np.sqrt(var_3)
        var_4 = np.sqrt(var_4)
        var_5 = np.sqrt(var_5)
        var_6 = np.sqrt(var_6)
        var_7 = np.sqrt(var_7)
        var_8 = np.sqrt(var_8)
        var_9 = np.sqrt(var_9)
        var_10 = np.sqrt(var_10)
        var_11 = np.sqrt(var_11)

        if result.success:
            # self.plot_A_vcmax_jmax(result.x,rd,theta,gms,k2LL,sco)
            ind = ['alphaGMax','alphaSMax','Nmax','vcmax1','vcmax2','vcmax3','vcmax4','Tp1','Tp2','Tp3','Tp4']
            df=pd.DataFrame([],columns=['estimate','err'],index=ind)
            df.loc['alphaGMax','estimate'] = result.x[0]
            df.loc['alphaSMax','estimate'] = result.x[1]
            df.loc['Nmax','estimate'] = result.x[2]
            
            df.loc['vcmax1','estimate'] = result.x[3]
            df.loc['vcmax2','estimate'] = result.x[4]
            df.loc['vcmax3','estimate'] = result.x[5]
            df.loc['vcmax4','estimate'] = result.x[6]
            
            df.loc['Tp1','estimate'] = result.x[7]
            df.loc['Tp2','estimate'] = result.x[8]
            df.loc['Tp3','estimate'] = result.x[9]
            df.loc['Tp4','estimate'] = result.x[10]

            df.loc['alphaGMax','err'] = var_1
            df.loc['alphaGMax','err'] = var_2
            df.loc['Nmax','err'] = var_3
            
            df.loc['vcmax1','err'] = var_4
            df.loc['vcmax2','err'] = var_5
            df.loc['vcmax3','err'] = var_6
            df.loc['vcmax4','err'] = var_7
            
            df.loc['Tp1','err'] = var_8
            df.loc['Tp2','err'] = var_9
            df.loc['Tp3','err'] = var_10
            df.loc['Tp4','err'] = var_11
            
            return df
        else:
            raise ValueError(result.message)
            return []       

    # def calculate_A_Bush_ave(self,xo,df_vcmax,ci,i_inc):
        
    #     alphaG, alphaS,vcmax1,vcmax2,vcmax3,vcmax4,TP1,TP2,TP3,TP4,R = xo
    #     vcmaxs=[vcmax1,vcmax2,vcmax3,vcmax4]
    #     tps=[TP1,TP2,TP3,TP4]
        
    #     ci = df_vcmax_r['Ci'].values.astype(float)
    #     i_inc = df_vcmax_r['Iinc'].values.astype(float)
    #     theta = df_vcmax_r['Theta'].values.astype(float)  
    #     O = df_vcmax_r['O'].values.astype(float)
    #     rd = df_vcmax_r['Rd'].values.astype(float)
    #     jmax = df_vcmax_r['Jmax'].values.astype(float)
            
    #     k2LL = df_vcmax_r['k2LL'].values.astype(float)     
    #     vcmax =  vcmaxs[replicate-1]
    #     # jmax =  jmaxs[replicate-1]
    #     TP =  tps[replicate-1]

    #     # Rubisco-limited part

    #     gamma_st = 0.5*O/sco;
    #     gamma_x = (gamma_st*(1-alphaG)+alphaT);
            
            
    #     #  Rubisco-limited part;
    #     KMCMO = kmc*(1+O/kmo);
    #     X1R = vcmax;
    #     X2R = KMCMO;
            
    #     PR = gm0*(X2R+gamma_x)+(X1R-rd)*R;
    #     QR2 = (ci-gamma_x)*X1R-(ci+X2R)*rd;
    #     QR = (X2R+gamma_x*(1-m))*(X1R-rd);
    #     AAR = X2R+gamma_x*(1-m)+R*(ci+X2R);
    #     BBR = m*(rd*X2R+gamma_x*X1R)-QR-PR*(ci+X2R)-R*QR2;
    #     CCR = -m*(rd*X2R+gamma_x*X1R)*(X1R-rd)+PR*QR2;
    #     AR = (-BBR-(BBR**2-4.*AAR*CCR)**0.5)/(2.*AAR);
            
    #     # Electron transport limited part;
            
    #     BB = k2LL*i_inc + jmax;
    #     J = (BB-(BB**2-4*theta*jmax*k2LL*i_inc)**0.5)/(2*theta);
    #     X1R = J/4;
    #     X2R = (1+2*alphaG-alphaT+alphaS)*2.*gamma_st;
    #     PR = gm0*(X2R+gamma_x)+(X1R-rd)*R;
    #     QR2 = (ci-gamma_x)*X1R-(ci+X2R)*rd;
    #     QR = (X2R+gamma_x*(1-m))*(X1R-rd);
    #     AAR = X2R+gamma_x*(1-m)+R*(ci+X2R);
    #     BBR = m*(rd*X2R+gamma_x*X1R)-QR-PR*(ci+X2R)-R*QR2;
    #     CCR = -m*(rd*X2R+gamma_x*X1R)*(X1R-rd)+PR*QR2;
    #     AJ = (-BBR-(BBR**2-4.*AAR*CCR)**0.5)/(2.*AAR);
            
    #     #  TPU limited part;            
    #     X1R = 3*TP;
    #     X2R = -(1+3*alphaG+6*alphaT+4*alphaS)*gamma_st;
    #     PR = gm0*(X2R+gamma_x)+(X1R-rd)*R;
    #     QR2 = (ci-gamma_x)*X1R-(ci+X2R)*rd;
    #     QR = (X2R+gamma_x*(1-m))*(X1R-rd);
    #     AAR = X2R+gamma_x*(1-m)+R*(ci+X2R);
    #     BBR = m*(rd*X2R+gamma_x*X1R)-QR-PR*(ci+X2R)-R*QR2;
    #     CCR = -m*(rd*X2R+gamma_x*X1R)*(X1R-rd)+PR*QR2;
    #     AP = (-BBR-(BBR**2-4.*AAR*CCR)**0.5)/(2.*AAR);
        
    #     A1 = np.minimum(AR,AJ)
    #     A_mod = np.minimum(A1,AP)
    #     return A_mod

    def calculate_A_Bush_XY(self,xo,df_vcmax,gms):
        
        """
        FvCB model of photosynthesis extended to include nitrogen assimilation
        Model based on Yin et al. 2021 10.1111/pce.14070 
        
        """
        
        alphaG, alphaS,vcmax1,vcmax2,vcmax3,vcmax4,TP1,TP2,TP3,TP4,R = xo
        vcmaxs=[vcmax1,vcmax2,vcmax3,vcmax4]
        tps=[TP1,TP2,TP3,TP4]
        
        kmc = 267
        kmo = 164

        sco = 3.259;
        alphaT = 0;
        m=0;
        # gm0 = 0
        replicates = df_vcmax['Replicate'].unique()
        A_mod_arr = np.array([])
        for replicate in replicates:
            df_vcmax_r=df_vcmax[df_vcmax['Replicate']==replicate];
            ci = df_vcmax_r['Ci'].values.astype(float)
            i_inc = df_vcmax_r['Iinc'].values.astype(float)
            theta = df_vcmax_r['Theta'].values.astype(float)  
            O = df_vcmax_r['O'].values.astype(float)
            rd = df_vcmax_r['Rd'].values.astype(float)
            jmax = df_vcmax_r['Jmax'].values.astype(float)
            
            k2LL = df_vcmax_r['k2LL'].values.astype(float)     
            vcmax =  vcmaxs[replicate-1]
            # jmax =  jmaxs[replicate-1]
            TP =  tps[replicate-1]
            gm0 = gms[replicate-1]
            # Rubisco-limited part

            gamma_st = 0.5*O/sco;
            gamma_x = (gamma_st*(1-alphaG)+alphaT);
            
            
            #  Rubisco-limited part;
            KMCMO = kmc*(1+O/kmo);
            X1R = vcmax;
            X2R = KMCMO;
            
            PR = gm0*(X2R+gamma_x)+(X1R-rd)*R;
            QR2 = (ci-gamma_x)*X1R-(ci+X2R)*rd;
            QR = (X2R+gamma_x*(1-m))*(X1R-rd);
            AAR = X2R+gamma_x*(1-m)+R*(ci+X2R);
            BBR = m*(rd*X2R+gamma_x*X1R)-QR-PR*(ci+X2R)-R*QR2;
            CCR = -m*(rd*X2R+gamma_x*X1R)*(X1R-rd)+PR*QR2;
            AR = (-BBR-(BBR**2-4.*AAR*CCR)**0.5)/(2.*AAR);
            
            # Electron transport limited part;
            
            BB = k2LL*i_inc + jmax;
            J = (BB-(BB**2-4*theta*jmax*k2LL*i_inc)**0.5)/(2*theta);
            X1R = J/4;
            X2R = (1+2*alphaG-alphaT+alphaS)*2.*gamma_st;
            PR = gm0*(X2R+gamma_x)+(X1R-rd)*R;
            QR2 = (ci-gamma_x)*X1R-(ci+X2R)*rd;
            QR = (X2R+gamma_x*(1-m))*(X1R-rd);
            AAR = X2R+gamma_x*(1-m)+R*(ci+X2R);
            BBR = m*(rd*X2R+gamma_x*X1R)-QR-PR*(ci+X2R)-R*QR2;
            CCR = -m*(rd*X2R+gamma_x*X1R)*(X1R-rd)+PR*QR2;
            AJ = (-BBR-(BBR**2-4.*AAR*CCR)**0.5)/(2.*AAR);
            
            #  TPU limited part;            
            X1R = 3*TP;
            X2R = -(1+3*alphaG+6*alphaT+4*alphaS)*gamma_st;
            PR = gm0*(X2R+gamma_x)+(X1R-rd)*R;
            QR2 = (ci-gamma_x)*X1R-(ci+X2R)*rd;
            QR = (X2R+gamma_x*(1-m))*(X1R-rd);
            AAR = X2R+gamma_x*(1-m)+R*(ci+X2R);
            BBR = m*(rd*X2R+gamma_x*X1R)-QR-PR*(ci+X2R)-R*QR2;
            CCR = -m*(rd*X2R+gamma_x*X1R)*(X1R-rd)+PR*QR2;
            AP = (-BBR-(BBR**2-4.*AAR*CCR)**0.5)/(2.*AAR);
        
            A1 = np.minimum(AR,AJ)
            A_mod = np.minimum(A1,AP)
            A_mod_arr = np.append(A_mod_arr,A_mod)
        
        return A_mod_arr


    def plot_A_Bush_XY(self,xo,df_vcmax,gms):
        plt.rcParams["figure.figsize"] = (15,8) 
        plt.rcParams.update({'font.size': 12})

        self.gas_exch_measurement.set_O2(0.21)      
        ACI = self.gas_exch_measurement.get_ACI_data()
        
        A = ACI['Net_CO2_assimilation_rate'].values 
        A = np.reshape(A,(-1,4),order='F')
        A_exp = np.mean(A,axis=1) 
        A_err = np.std(A,axis=1)/2 
        
        ci = ACI['Intercellular_CO2_concentration'].values
        ci = np.reshape(ci,(-1,4),order='F')
        ci = np.mean(ci,axis=1)

        A_mod_ci = self.calculate_A_Bush_XY(xo,df_vcmax,gms)
        df_vcmax['A_mod_ci']=A_mod_ci
        df_vcmax21 = df_vcmax[df_vcmax['O']==210]
        df_vcmax_ci = df_vcmax21[df_vcmax21['curve']=='ACI']
        
        A_mod_ci = df_vcmax_ci['A'].values 
        A_mod_ci = np.reshape(A_mod_ci,(-1,4),order='F')
        A_mod_ci = np.mean(A_mod_ci,axis=1)   
        
        self.gas_exch_measurement.set_O2(0.21)      
        AI = self.gas_exch_measurement.get_AI_data()
        A_i = AI['Net_CO2_assimilation_rate'].values

        A_i = np.reshape(A_i,(-1,4),order='F')

        A_exp_i = np.mean(A_i,axis=1)   
        A_err_i = np.std(A_i,axis=1)/2     
        
        i_inc_i = AI['Irradiance'].values       
        i_inc_i = np.reshape(i_inc_i,(-1,4),order='F')
        i_inc_i = np.mean(i_inc_i,axis=1) 


        df_vcmax_i = df_vcmax21[df_vcmax21['curve']=='AI']

        A_mod_i = df_vcmax_i['A'].values 
        A_mod_i = np.reshape(A_mod_i,(-1,4),order='F')
        A_mod_i = np.mean(A_mod_i,axis=1)   
        
        
        fig,(ax1,ax2) = plt.subplots(ncols=2,constrained_layout=True)        
        ax1.errorbar(ci,A_exp,A_err,fmt='ko',label='Expt.',mfc='white',mec='black',markersize=8)
        ax1.plot(ci,A_mod_ci,'k-',label='Model')
        ax1.set_xlabel('Intercellular $CO_2$ (µmol $mol^{-1}$)',fontsize=20)
        ax1.set_ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)',fontsize=20)
        ax1.xaxis.set_ticks(np.arange(0, 2050, 250))        
        ax2.errorbar(i_inc_i,A_exp_i,A_err_i,fmt='ko',label='Expt.',mfc='white',mec='black',markersize=8)
        ax2.plot(i_inc_i,A_mod_i,'k-',label='Model')
        ax2.set_xlabel('Irradiance (µmol $m^{-2}$ $s^{-1}$)',fontsize=20)
        ax2.set_ylabel('Net photosynthesis (µmol $m^{-2}$ $s^{-1}$)',fontsize=20)    
        ax2.legend(loc='lower right', fontsize='x-large')     
        ax2.xaxis.set_ticks(np.arange(0, 2200, 300))            
        plt.show()  
        
        
    def model_Vcmax_Bush_XY(self,xo,df_vcmax,gms):
        A_mod = self.calculate_A_Bush_XY(xo,df_vcmax,gms)    
        return df_vcmax['A'].values - A_mod

    
    def estimate_Vcmax_Bush_XY(self,inputs):
        
        rds = inputs['Rd']
        jmaxs = inputs['Jmax']
        thetas = inputs['Theta']
        k2LL = inputs['k2LL']    
        gms = inputs['gms']
        df_vcmax = self.get_vcmax_data(rds,jmaxs,thetas,k2LL)
      
        #alphaG, alphaS, Vcmax1-4,TP1-4,sigma
        bnds=((0.0001,0.0004,0,0,0,0,0,0,0,0,0),(1,1,500,500,500,500,50,50,50,50,5)) 
        x0 = np.array([0.05,0.04,180,180,180,180,25,25,25,25,0.1])                
        result = optimize.least_squares(self.model_Vcmax_Bush_XY,x0,args=[df_vcmax,gms],method='trf',bounds=bnds)
        # result = optimize.least_squares(self.model_Vcmax_Bush,x0,args=[inputs],method='lm')
        
        res = self.model_Vcmax_Bush_XY(result.x,df_vcmax,gms)
        J = np.array(result.jac)
        S = np.array(res).T.dot(np.array(res))
        H=2*J.T.dot(J);
        degfr=len(res)-11;
        G = np.linalg.pinv(H)
        var_1=2*S*G[0,0]/degfr;
        var_2=2*S*G[1,1]/degfr;
        var_3=2*S*G[2,2]/degfr;
        var_4=2*S*G[3,3]/degfr;
        var_5=2*S*G[4,4]/degfr;
        var_6=2*S*G[5,5]/degfr;
        var_7=2*S*G[6,6]/degfr;
        var_8=2*S*G[7,7]/degfr;
        var_9=2*S*G[8,8]/degfr;
        var_10=2*S*G[9,9]/degfr;
        var_11=2*S*G[10,10]/degfr;
        
        var_1 = np.sqrt(var_1)
        var_2 = np.sqrt(var_2)
        var_3 = np.sqrt(var_3)
        var_4 = np.sqrt(var_4)
        var_5 = np.sqrt(var_5)
        var_6 = np.sqrt(var_6)
        var_7 = np.sqrt(var_7)
        var_8 = np.sqrt(var_8)
        var_9 = np.sqrt(var_9)
        var_10 = np.sqrt(var_10)
        var_11 = np.sqrt(var_11)

        if result.success:
            self.plot_A_Bush_XY(result.x,df_vcmax,gms)
            ind = ['alphaG','alphaS','vcmax1','vcmax2','vcmax3',\
                   'vcmax4','Tp1','Tp2','Tp3','Tp4','sigma']
            df=pd.DataFrame([],columns=['estimate','err'],index=ind)
            df.loc['alphaG','estimate'] = result.x[0]
            df.loc['alphaS','estimate'] = result.x[1]
            
            df.loc['vcmax1','estimate'] = result.x[2]
            df.loc['vcmax2','estimate'] = result.x[3]
            df.loc['vcmax3','estimate'] = result.x[4]
            df.loc['vcmax4','estimate'] = result.x[5]
            
            df.loc['Tp1','estimate'] = result.x[6]
            df.loc['Tp2','estimate'] = result.x[7]
            df.loc['Tp3','estimate'] = result.x[8]
            df.loc['Tp4','estimate'] = result.x[9]
            
            df.loc['sigma','estimate'] = result.x[10]

            df.loc['alphaG','err'] = var_1
            df.loc['alphaS','err'] = var_2
            
            df.loc['vcmax1','err'] = var_3
            df.loc['vcmax2','err'] = var_4
            df.loc['vcmax3','err'] = var_5
            df.loc['vcmax4','err'] = var_6
            
            df.loc['Tp1','err'] = var_7
            df.loc['Tp2','err'] = var_8
            df.loc['Tp3','err'] = var_9
            df.loc['Tp4','err'] = var_10
            df.loc['sigma','err'] = var_11
            
            
            return df
        else:
            raise ValueError(result.message)
            return []       