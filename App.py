# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 18:56:57 2021

@author: retta001
"""
from Gas_exchange_measurement import Gas_exchange_measurement

import tkinter as tk
from PIL import Image, ImageTk

root=tk.Tk()

canvas = tk.Canvas(root,width=600,height=300)
canvas.grid(columnspan=4,rowspan=4)
#ACI and AI
logo = Image.open('logo.tif')
logo = ImageTk.PhotoImage(logo)
logo_label = tk.Label(image=logo)
logo_label.image = logo
logo_label.grid(column=4,row=0)


logo2 = Image.open('WUR_logo.png')
logo2 = ImageTk.PhotoImage(logo2)
logo_2_label = tk.Label(image=logo2)
logo_2_label.image = logo
logo_2_label.grid(column=4,row=1)

r = tk.StringVar()  # Select species
r.set("'B.Nigra'")
r2 = tk.StringVar() # Treatment, HL or LL
r2.set("LL")
r3 = tk.IntVar() # Oxygen, 0.21 or 0.02
r3.set(0.21)

frame_species = tk.LabelFrame(root,text ="Species", padx=5,pady=5)
frame_species.grid(column=0,row=0)
frame_treatment = tk.LabelFrame(root,text ="Treatment", padx=5,pady=5)
frame_treatment.grid(column=1,row=0)
frame_Oxygen = tk.LabelFrame(root,text ="Oxygen", padx=5,pady=5)
frame_Oxygen.grid(column=2,row=0)

#Species and treatment
btn1=tk.Radiobutton(frame_species,text="B.Nigra",variable=r,value="B.Nigra")
btn2=tk.Radiobutton(frame_species,text="H.Incana",variable=r,value="H.Incana")
btn3=tk.Radiobutton(frame_treatment,text="HL",variable=r2,value="HL")
btn4=tk.Radiobutton(frame_treatment,text="LL",variable=r2,value="LL")
btn5=tk.Radiobutton(frame_Oxygen,text="0.21",variable=r3,value=210)
btn6=tk.Radiobutton(frame_Oxygen,text="0.02",variable=r3,value=20)

btn1.grid(column=0,row=0)
btn2.grid(column=1,row=0)
btn3.grid(column=0,row=0)
btn4.grid(column=1,row=0)
btn5.grid(column=1,row=1)
btn6.grid(column=2,row=1)
  
#Instructions
text_box = tk.Text(root,height=10,width=50)
text_box.grid(column=4,row=2,rowspan=2)
text_box.insert(1.0,"Examine the response of:\n \n1. photosynthesis to CO2 and light\n \n2. the response of gs to CO2 and light\n\
                \nProject: Extremophile\nMoges Retta")
text_box.tag_configure("center",justify="center")

# Photosynthesis and refixation responses    
def calculate_values():

    species = r.get()
    treatment = r2.get()
    O = r3.get()/1000
    gas_exch_measurement = Gas_exchange_measurement(O,species,treatment)
    [I_ave_ci,Ci_ave_ci,A_ave_ci,gs_ave_ci,A_std,gs_std] = gas_exch_measurement.average_A_CI()
    gas_exch_measurement.plot_ave_A_CI(Ci_ave_ci,A_ave_ci,A_std)
#    gas_exch_measurement.plot_ave_gs_CI(Ci_ave_ci,gs_ave_ci,gs_std)
#    [I_ave_i,i_ave_i,A_ave_i,gs_ave_i,A_std,gs_std] = gas_exch_measurement.average_A_I()
#    gas_exch_measurement.plot_ave_A_I(I_ave_i,A_ave_i,A_std)
#    gas_exch_measurement.plot_ave_gs_I(I_ave_i,gs_ave_i,gs_std)
    logo = Image.open('A_BN_LL.png')
    logo = ImageTk.PhotoImage(logo)
    logo_label = tk.Label(image=logo)
    logo_label.image = logo
    logo_label.grid(column=0,row=3)
    
    logo = Image.open('A_BN_LL.png')
    logo = ImageTk.PhotoImage(logo)
    logo_label = tk.Label(image=logo)
    logo_label.image = logo
    logo_label.grid(column=1,row=3)

frame_compute = tk.LabelFrame(root,text ="Calculate photosynthesis")
frame_compute.grid(column=0,row=1)

#frame_sensitivity = tk.LabelFrame(root,text ="Calculate sensitivity")
#frame_sensitivity.grid(column=0,row=3)
    
##Calculate rdiff
browse_text = tk.StringVar()
browse_btn = tk.Button(frame_compute,textvariable=browse_text, command=lambda:calculate_values(),font="Raleway",padx=25,pady=25,bd=5)
browse_text.set("COMPUTE")
browse_btn.grid(column=1,row=1)
#
##Calculate sensitivity
#browse_2_text = tk.StringVar()
#browse_2_btn = tk.Button(frame_sensitivity,textvariable=browse_2_text, command=lambda:sensitivity_wall(),font="Raleway",padx=25,pady=25,bd=5)
#browse_2_text.set("Cell wall")
#browse_2_btn.grid(column=0,row=2)
#
#browse_3_text = tk.StringVar()
#browse_3_btn = tk.Button(frame_sensitivity,textvariable=browse_3_text, command=lambda:sensitivity_chloroplast_envelop(),font="Raleway",padx=17,pady=25,bd=5)
#browse_3_text.set("Chl.envlope")
#browse_3_btn.grid(column=1,row=2)
#
#browse_4_text = tk.StringVar()
#browse_4_btn = tk.Button(frame_sensitivity,textvariable=browse_4_text, command=lambda:sensitivity_cytosol(),font="Raleway",padx=25,pady=25,bd=5)
#browse_4_text.set("Cytosol")
#browse_4_btn.grid(column=2,row=2)
#
#browse_5_text = tk.StringVar()
#browse_5_btn = tk.Button(frame_sensitivity,textvariable=browse_5_text, command=lambda:sensitivity_stroma(),font="Raleway",padx=27,pady=25,bd=5)
#browse_5_text.set("Stroma")
#browse_5_btn.grid(column=0,row=4)
#
#browse_6_text = tk.StringVar()
#browse_6_btn = tk.Button(frame_sensitivity,textvariable=browse_6_text, command=lambda:sensitivity_palisade(),font="Raleway",padx=26,pady=25,bd=5)
#browse_6_text.set("Palisade")
#browse_6_btn.grid(column=1,row=4)
#
#browse_7_text = tk.StringVar()
#browse_7_btn = tk.Button(frame_sensitivity,textvariable=browse_7_text, command=lambda:sensitivity_spongy(),font="Raleway",padx=25,pady=25,bd=5)
#browse_7_text.set("Spongy")
#browse_7_btn.grid(column=2,row=4)

root.after(2000)
root.mainloop()