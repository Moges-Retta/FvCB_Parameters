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

        AI = self.gas_exch_measurement.average_A_I()
        Iave = AI['Irradiance'].values
        phiPS2 = AI['PhiPS2'].values
        std_phiPS2 = AI['PhiPS2_err'].values
        
        Jmax,theta = params
        k2LL = s*PHI2LL
        J = s*Iave*phiPS2
        SQ = ((k2LL*Iave+Jmax)**2-4*theta*k2LL*Jmax*Iave)**0.5
        J_mod = (k2LL*Iave+Jmax-SQ)/(2*theta) 
        fig, ax = plt.subplots()
        ax.errorbar(Iave,J,std_phiPS2*J,fmt='ko',label='CF') 
        ax.plot(Iave,J_mod,'k-',linewidth=2.0,label='Model')  
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

    def get_vcmax_data(self,Rd,jmax,theta,K2LL):
        self.gas_exch_measurement.set_O2(0.21)
        ACIH = self.gas_exch_measurement.get_ACI_data()
        replicates = ACIH['Replicate'].unique()
        cols = ['Rd','Theta','Jmax','k2LL','Ci','Iinc','A']  # RD THETA JMAX K2LL CI IINC A;
        df_vcmax = pd.DataFrame([],columns = cols)
        count = 0 
        for replicate in replicates:
            df = pd.DataFrame([],columns = cols)
            ACIH_r = ACIH[ACIH['Replicate']==replicate]
            ci = ACIH_r['Intercellular CO2 concentration'].values*constants.atm/10**5  
            i_inc = ACIH_r['Irradiance'].values
            A = ACIH_r['Net CO2 assimilation rate'].values            
            rd = Rd[count]
            k2LL = K2LL[count]
            df.loc[:,'A'] = A                        
            df.loc[:,'Rd'] = rd*-1
            df.loc[:,'Theta'] = theta
            df.loc[:,'Jmax'] = jmax
            df.loc[:,'k2LL'] = k2LL
            df.loc[:,'Ci'] = ci
            df.loc[:,'Iinc'] = i_inc
            df_vcmax = df_vcmax.append(df)
            count+=1
        return df_vcmax
       
    def calculate_A(self,xo,rds,jmaxs,thetas,k2LLs):
        vcmax,TP,R = xo
        df_vcmax = self.get_vcmax_data(rds,jmaxs,thetas,k2LLs)
        rd = df_vcmax['Rd'].values
        jmax = df_vcmax['Jmax'].values
        theta = df_vcmax['Theta'].values
        k2LL = df_vcmax['k2LL'].values
        ci = df_vcmax['Ci'].values
        i_inc = df_vcmax['Iinc'].values
        
        O = 210 #mbar
        sco = 3.259
#        sco = 2.64
        gamma_x = 0.5*O/sco
        gm0 = 0
#        R = 1;        
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
    
    def plot_A(self,xo,rds,jmaxs,thetas,k2LLs):
#        df_vcmax = self.get_vcmax_data(rds,jmaxs,thetas,k2LLs)
#        A = df_vcmax['A'].values  
#        ci = df_vcmax['Ci'].values
        ACI = self.gas_exch_measurement.average_A_CI()
        A = ACI['Net CO2 assimilation rate'].values 
        A_err = ACI['Photo_err'].values 
        ci = ACI['Intercellular CO2 concentration'].values
        A_mod = self.calculate_A(xo,rds,jmaxs,thetas,k2LLs)
        A_mod = A_mod.reshape((ACI.shape[0], 4))

        A_mod_ave = np.nanmean(A_mod,axis=1)
        A_mod_ave = np.sort(A_mod_ave, axis=None)

        fig,ax = plt.subplots()        
        ax.errorbar(ci,A,A_err,fmt='ko')
#        ax.plot(ci,A,'ko')
        ax.plot(ci,A_mod_ave)
        plt.show()  

    
    def model_Vcmax(self,xo,rds,jmaxs,thetas,k2LLs):
        A_mod = self.calculate_A(xo,rds,jmaxs,thetas,k2LLs)
        df_vcmax = self.get_vcmax_data(rds,jmaxs,thetas,k2LLs)
        A = df_vcmax['A'].values        
        return A - A_mod
    
    
    def estimate_Vcmax(self,inputs):
        rd = inputs.get('Rd')*-1
        jmax = inputs.get('Jmax')
        theta = inputs.get('Theta')
        k2LL = inputs.get('k2LL')
        bnds=((0,0,0),(700,40,10)) 
        x0 = np.array([250,26,5])        #Vcmax, TP ,R     
        result = optimize.least_squares(self.model_Vcmax,x0,args=[rd,jmax,theta,k2LL],method='trf',bounds=bnds)
        res = self.model_Vcmax(result.x,rd,jmax,theta,k2LL)
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
            self.plot_A(result.x,rd,jmax,theta,k2LL)
            cols = ['Vcmax','Vcmax_err','Tp','Tp_err','Sigma_gm','Sigma_gm_err']
            df=pd.DataFrame([],columns=cols)
            df.loc[0,'Vcmax'] = result.x[0]
            df.loc[0,'Tp'] = result.x[1]
            df.loc[0,'Sigma_gm'] = result.x[2]
            df.loc[0,'Vcmax_err'] = var_1
            df.loc[0,'Tp_err'] = var_2
            df.loc[0,'Sigma_gm_err'] = var_3
            print(df)            
            return df
        else:
            raise ValueError(result.message)
            return []        


    def get_vcmax_data_individual(self,Rd,jmax,theta,K2LL,replicate):
        self.gas_exch_measurement.set_O2(0.21)
        ACIH = self.gas_exch_measurement.get_ACI_data()
        cols = ['Rd','Theta','Jmax','k2LL','Ci','Iinc','A']  # RD THETA JMAX K2LL CI IINC A;
        df_vcmax = pd.DataFrame([],columns = cols)
        df = pd.DataFrame([],columns = cols)
        ACIH_r = ACIH[ACIH['Replicate']==replicate]
        ci = ACIH_r['Intercellular CO2 concentration'].values*constants.atm/10**5  
        i_inc = ACIH_r['Irradiance'].values
        A = ACIH_r['Net CO2 assimilation rate'].values            
        df.loc[:,'A'] = A                        
        df.loc[:,'Rd'] = Rd*-1
        df.loc[:,'Theta'] = theta
        df.loc[:,'Jmax'] = jmax
        df.loc[:,'k2LL'] = K2LL
        df.loc[:,'Ci'] = ci
        df.loc[:,'Iinc'] = i_inc
        df_vcmax = df_vcmax.append(df)
        return df_vcmax
    
       
    def calculate_A_individual(self,xo,rds,jmaxs,thetas,k2LLs,replicate):
        vcmax,TP,R = xo
        df_vcmax = self.get_vcmax_data_individual(rds,jmaxs,thetas,k2LLs,replicate)
        rd = df_vcmax['Rd'].values
        jmax = df_vcmax['Jmax'].values
        theta = df_vcmax['Theta'].values
        k2LL = df_vcmax['k2LL'].values
        ci = df_vcmax['Ci'].values
        i_inc = df_vcmax['Iinc'].values
        
        O = 210 #mbar
        sco = 3.259
#        sco = 2.64
        gamma_x = 0.5*O/sco
        gm0 = 0
#        R = 1;        
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

 
    def model_Vcmax_individual(self,xo,rds,jmaxs,thetas,k2LLs,replicate):
        A_mod = self.calculate_A_individual(xo,rds,jmaxs,thetas,k2LLs,replicate)
        df_vcmax = self.get_vcmax_data_individual(rds,jmaxs,thetas,k2LLs,replicate)
        A = df_vcmax['A'].values        
        return A - A_mod
    
    def plot_A_individual(self,xo,rds,jmaxs,thetas,k2LLs,replicate):
        df_vcmax = self.get_vcmax_data_individual(rds,jmaxs,thetas,k2LLs,replicate)
        A = df_vcmax['A'].values  
        ci = df_vcmax['Ci'].values
        A_mod = self.calculate_A_individual(xo,rds,jmaxs,thetas,k2LLs,replicate)
        fig,ax = plt.subplots()        
        ax.plot(ci,A,'ko')
        ax.plot(ci,A_mod)
        plt.show()  
        
       
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
            k2LL = k2LLs[count]            
            bnds=((0,0,0),(700,40,10)) 
            x0 = np.array([250,26,5])        #Vcmax, TP ,R     
            result = optimize.least_squares(self.model_Vcmax_individual,x0,args=[rd,jmax,theta,k2LL,replicate],method='trf',bounds=bnds)
            res = self.model_Vcmax_individual(result.x,rd,jmax,theta,k2LL,replicate)
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
                self.plot_A_individual(result.x,rd,jmax,theta,k2LL,replicate)
                df=pd.DataFrame([],columns=cols)
                df.loc[count,'Replicate'] = count                
                df.loc[count,'Vcmax'] = result.x[0]
                df.loc[count,'Tp'] = result.x[1]
                df.loc[count,'Sigma_gm'] = result.x[2]
                df.loc[count,'Vcmax_err'] = var_1
                df.loc[count,'Tp_err'] = var_2
                df.loc[count,'Sigma_gm_err'] = var_3
                vcmax_values = vcmax_values.append(df)
                print(vcmax_values)
                count+=1
            else:
                raise ValueError(result.message)
                return []
        return vcmax_values        