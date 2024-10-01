# FvCB_Parameters
The programs extract data from raw gas exchange data file in .csv format. The raw data is cleand using Analyze_csv_files.py, the data is analysed using Gas_exchange_measurement.py 
and parameters of FvCB photosynthesis kinetics are estimated using FvCB_Parameters.py. 

## Note on the experiment: 
Plants: _Hirschfeldia_ _incana_ and _Brassica_ _nigra_
Treatment: growth light intensity was low light (LL) or high light (HL)
Experiment: combined gas exchange and chlorophyll fluorescence - CO<sub> 2</sub> and light response of photosynthesis at 21 % O<sub> 2</sub> and 2 % O<sub> 2</sub>

### Analyze_csv_files.py

make_data(response,Oxygen,species_code,treatment,measurement_days)

Extract A-CI or A-I curve for a given species, treatment and  measurement condition and day of measurement)
 
response : CO<sub> 2</sub> or light

Oxygen : 21 % or 2 % O<sub> 2</sub>

Species-code : H. incana or B. nigra

treatment : HL or LL

measurement_days :  date of the gas exchange measurement

The extracted data is plotted or saved as excel file using the following functions:

plot_response(treatment,data,measurement_days)

replicates_to_Excel(data_frame,species,Oxygen,curve,treatment)


### Gas_exchange_measurement.py


Model gas exchange measurement using species type, treatement and oxygen concentration used in the measurement as attributes.

The extracted gas exchange data from Analyze_csv_files.py is loaded.

Static variables are A-CI data and A-I data.

The class has following methods : 

1. plot_A_CI() : plot CO<sub> 2</sub> response of Gas_exchange_measurement object.

2. plot_A_I() : plot light response of Gas_exchange_measurement object.

3. get_AI_data() : return light response of photosynthesis as pandas data frame.

4. get_ACI_data() : return CO2 response of photosynthesis as pandas data frame.

5. plot_ave_A_CI() : plots average values of photosynthesis in response to CO<sub> 2</sub>.

6. plot_ave_A_I() : plots average values of photosynthesis in response to light.

7. plot_ave_gs_CI() : plots average values of stomatal conductance in response to CO<sub> 2</sub>.

8. plot_ave_gs_I() : plots average values of stomatal conductance in response to light.

9. show_significant() : show asterisk for significant mean values of photosynthesis.

10. show_significant_gs() : show asterisk for significant mean values of stomatal conductance.

11. t_test_A_I() and t_test_A_CI() : compares mean values of photosynthesis using t-test.

11. compare_A() :  compare the responses of photosynthesis for the species and calls t_test_A_I() or t_test_ACI() and show_significant() function to denote significance

12. compare_gs() :  compare the responses of stomatal conductance for the species and calls t_test_gs_I() or t_test_gs_CI() and show_significant() function to denote significance

13. compare_PhiPSII () : compare the responses of quantum efficiency of PSII for the species and calls for show_significant() function to denote significance.

14. correct_leak() : correct the gas exchange data using model model_delta_co2() developed from leakage correction data and returns a data frame of leakage corrected data.


### FvCB_parameters.py

Estimate parameters of photosynthesis kinetics using the methods of [Yin and Struik 2009](10.1111/j.1365-3040.2009.02016.x) and [Yin et al., 2009](10.1111/j.1365-3040.2009.01934.x).

The class has gas exchange data as attribute.

The class has the following methods:

1. estimate_Rd () : Estimate day respiration (Rd) as intercept of linear regression between A and PhiPSII*Iinc/4. The light response of photosynthesis at 2 % O<sub> 2</sub> is filtered for values with irradiance less than or equal to rd_cut_light. Rd for each replicate is estimated and the fit is plotted along with r-square values. 

The slope of the linear regression is the calibration factor according to Yin et al. 2009. 

2. estimate_Rd_common(): estimated one Rd value for all the replicates with gas exchange data at 2 % O<sub> 2</sub> and 21 % O<sub> 2</sub>.

3. estimate_individual_Jmax() : Estimate Jmax and convexity factor (theta) using quantum efficiency of PSII (Phi2).

4. estimate_individual_phi2LL(): Estimate the quantum efficiency of PSII e flow at strictly limiting light level (Phi2LL).

5. estimate_Jmax () : Estimate a common Jmax and convexity factor (theta) using quantum efficiency of PSII (Phi2).

6. estimate_bH_bL(): Estimate initial carboxylation efficiency at 21 % O<sub> 2</sub> (bH) and 2 % O<sub> 2</sub> (bL) as slope of initial part of A-CI (50<=CO<sub> 2</sub><=150). Uses get_SCO_data()

to extract the data for estimating bH and bL.

7. estimate_Sco() : Estimate rubisco specificity factor Sc/o using bH and bL 

8. estimate_Vcmax(): estimate maximum rate of Rubisco carboxylation, Vcmax. Function calls 
    1. get_Vcmax_data() to extract data required for Vcmax estimation
    2. model_vcmax() gives the objective for minimization, A_measured - A_model and,
    3. plot_A() to plot the result of the fitting
    
9. estimate_individual_Vcmax() estimate Vcmax, Trios phosphate utilization rate, Tp and mesophyll conductance, gm for each replicate

10. estimate_Vcmax_Jmax ():  estimate Vcmax,  Tp and maximum rate of electron transport, Jmax and sigma, factor for response of g<sub> m</sub>, simultaneously

11. estimate_Vcmax_constant_gm () :  estimate Vcmax, Tp assuming gm is constant (sigma = 0) across CO2 and light levels

12. estimate_Vcmax_var_gm () : estimate Vcmax, Tp and sigma, factor for response of gm, assuming gm is varible across CO<sub> 2</sub> and light levels.

13. NRH_A_gm () : estimate g<sub> m</sub> and calibration factor,lump, based on NRH-A method (Yin et al. 2009,doi 10.1111/j.1365-3040.2009.02016.x). 



