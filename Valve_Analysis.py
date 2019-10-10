#!/usr/bin/env python
# coding: utf-8

# # Import Dependencies and Define Functions

#%matplotlib notebook

#Import dependencies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import os
from tkinter import Tk
from tkinter.filedialog import askdirectory
from iapws import IAPWS97 as steam_prop
import matplotlib.animation as animation
from matplotlib.widgets import Slider, RadioButtons, Button
import itertools
import PIL
import time
from scipy.stats import linregress
from statistics import mean
np.seterr(all='raise')


#Define critical pressure ratio
alpha = 0.55

#Define a function that calculates the density for a given pressure and temperature
def calc_rho(p, t):
    
    #convert psia to MPA
    pressure = p*0.00689476

    #convert farenheit to Kelvin
    temp = (t - 32)*5/9 + 273.15

    #get the density 
    rho = steam_prop(P=pressure, T=temp).rho
    
    #convert density from kg/M^3 to lb/ft^3
    rho = rho*0.062428
    
    return rho

# Piecewise linear fit
def piecewise_linear(x, y0, y1, b0, b1, b2):
    global x0
    global x1

    return np.piecewise(x, 
                        [x < x0, 
                         (x >= x0) & (x < x1), 
                         x >= x1], 
                        [lambda x: b0*x + y0, 
                         lambda x: b1*x + y1-b1*x1,
                         lambda x: b2*x + y1-b2*x1])


# Piecewise rev2
def piecewise_2(x, y0, y1, m0, m1, brk):
    return np.piecewise(x, 
                        [x < brk],
                        [lambda x: m0*x + y0, 
                         lambda x: m1*x + y1-m1*brk])

# Piecewise rev3
def piecewise_3(x, y0, y1, b0, b1, b2, x0, x1):
    return np.piecewise(x, 
                        [x < x0, 
                         (x >= x0) & (x < x1), 
                         x >= x1], 
                        [lambda x: b0*x + y0, 
                         lambda x: b1*x + y1-b1*x1,
                         lambda x: b2*x + y1-b2*x1])

# Square root function
def root(x, a, b):
    return a*(x**0.5) + b

# cubic function
def cubic(x, a, b, c, d):
    return a*(x**3) + b*(x**2) + c*(x) + d

# quadratic function
def quadratic(x, a, b, c):
    return a*(x**2) + b*(x) + c

# Fourth power function
def fourth(x, a, b, c, d, e):
    return a*(x**4) + b*(x**3) + c*(x**2) + d*(x) + e

# Power function
def power(x, a, n, b):
    return a*(x**n) + b
    
#Function for getting user input on curve fitting
def user_plot_fit(x0_start, x1_start, title, xdata, ydata, xlabel, ylabel, xunits, yunits):
    
    #re-initialize global x0 - assumption for first breakpoint
    #re-initialize global x1 - assumption for second breakpoint
    global x0 
    global x1
    x0 = x0_start
    x1 = x1_start

    #optimize piecewise fit with user-defined breakpoints
    p, e = optimize.curve_fit(piecewise_linear, xdata, ydata)

    #Define range for governor demand model
    model_x = np.linspace(-1, 100, 100)

    # Create figure and axis objects
    fig, ax = plt.subplots()

    #add plot labels
    plt.title(f"Use Sliders to Adjust Curve Fit!\n\n{title}")
    plt.ylabel(f"{ylabel}\n({yunits})")
    plt.xlabel(f"{xlabel}\n({xunits})")

    #plot historical/design data
    data, = plt.plot(xdata, ydata, 'o')

    #plot piecewise curve fit
    fit, = plt.plot(model_x, piecewise_linear(model_x, *p))

    #left, bottom, width, height
    ax_break1 = plt.axes([0.25, 0.05, 0.50, 0.02])
    ax_break2 = plt.axes([0.25, 0.02, 0.50, 0.02])

    #Create slider animations for the breakpoints
    slider1 = Slider(ax_break1, 'Breakpoint 1', 0, 100, valinit=x0_start)
    slider2 = Slider(ax_break2, 'Breakpoint 2', 0, 100, valinit=x1_start)
    
    def update(val):
        global x0
        global x1
        x0 = slider1.val
        x1 = slider2.val
        # update curve
        p, e = optimize.curve_fit(piecewise_linear, xdata, ydata)
        fit.set_ydata(piecewise_linear(model_x, *p))
        # redraw canvas while idle
        fig.canvas.draw_idle()

    # call update function on slider value change
    slider1.on_changed(update)
    slider2.on_changed(update)

    #show plot
    plt.tight_layout(pad=4)
    plt.show()

    #Save results to DataFrame
    dictionary = {
        xlabel : ax.get_children()[1]._x,
        ylabel : ax.get_children()[1]._y
        }
     
    model_df = pd.DataFrame(dictionary)
    model_df.to_csv(f"projects/{project}/{ylabel}_model.csv")
    
    return model_df

#define a function that gets the flow regime for a given jratio nd pratio
def get_regime(jratio, pratio):
    # return 1 if "Neither Choked"
    # return 2 if "Governor Choked"
    # return 3 if "Nozzle Choked"
    # return 4 if "Both Choked"
    
    global alpha
    
    #Define the intersection of the curves
    triple_point = alpha + (alpha - 1) *0.55**2/alpha
    
    #if jratio is to the right of the triple point, test for neither choked or nozzle choked
    if jratio > 0.55:
        
        #define the limit below which the nozzle is choked
        limit = alpha*jratio*(jratio-(-4*alpha+jratio**2+4)**0.5)/(2*(alpha-1))
        
        #test if the pratio is in "neither choked" or "nozzle choked" regime
        if pratio > limit:
            return 1
        else:
            return 3
    
    #if jratio is to the left of the triple point, test for neither choked, governor choked, or both choked
    else:
        
        #define the limit below which both are choked
        limit = triple_point/alpha*jratio
        
        #if the pratio is below the limit, the flow is "both choked"
        if pratio < limit:
            return 4
        
        #if not, define the limit below which the governover is choked
        else:
            limit = alpha + (alpha-1)*jratio**2/alpha
            
            #if the pratio is below the limit, the flow is "governor choked"
            if pratio < limit:
                return 2
            else:
                return 1

#Define a function that calculates the Bowl Pressure Ratio from flow regime, jratio, and pratio
#case 1 neither choked
# pbr = 1/2(-jr^2 + pxr + sqrt(4*jr^2 + jr^4 - 2*jr^2*pxr + pxr^2))
# flow = jr*sqrt(rho_in*(1-pbr))

#case 2 governor choked
#pbr = 1/2(pxr + sqrt(4*jr^2 +pxr^2 - 4*jr^2*alpha))
# flow = jr*sqrt(rho_in*(1-alpha)) 

#case 3 nozzle choked
#pbr = 1/2( - jr^2/(1-alpha) - jrsqrt(4+jr^2-4*alpha)/(-1+alpha))
#flow = jr*sqrt(rho_in*(1-pbr))

#case 4 nozzle choked
#pbr = jr
#flow = jr*sqrt(rho_in*(1-alpha))

def calc_pbr(regime, j_ratio, p_ratio):
        
        #Use notebook alpha
        global alpha
        
        if regime == 1:
            
            #Calculate bowl pressure
            pbr = 0.5*(-j_ratio**2 + p_ratio + (4*j_ratio**2 + j_ratio**4 - 2*p_ratio*j_ratio**2 + p_ratio**2)**0.5)

        elif regime == 2:
            #pbr = 1/2(pxr + sqrt(4*jr^2 +pxr^2 - 4*jr^2*alpha))

            #Calculate bowl pressure
            pbr = 0.5*(p_ratio + (4*j_ratio**2 + p_ratio**2 - 4*j_ratio**2*alpha)**0.5)

        elif regime == 3:
            #pbr = 1/2( - jr^2/(1-alpha) - (jr/(-1+alpha))*sqrt(4+jr^2-4*alpha))
            
            #Calculate bowl pressure
            pbr = 0.5*(-j_ratio**2/(1-alpha) -(j_ratio/(alpha-1))*((4+j_ratio**2 - 4*alpha)**0.5))

        elif regime == 4:
            
            #pbr =??????

            #Calculate bowl pressure
            pbr = j_ratio
        
        return pbr

    
#Define a function that calculates the flow in KPPH from flow regime, inlet pressure, governor j, and pb
def calc_flow(regime, pi, pb, jgv, r):

        #Use notebook alpha
        global alpha
        
        if (regime == 1):

            flow = 3.6*jgv*(r*(pi-pb))**0.5
        
        elif regime ==3:
            
            flow = 3.6*jgv*(r*(pi-pb))**0.5

        elif regime == 2:

            flow = 3.6*jgv*(r*(alpha*pi))**0.5
        
        elif regime ==4:
            
            #PLACEHOLDER
            flow = 3.6*jgv*(r*(alpha*pi))**0.5
        
        return flow
    

#Define a function that calculates the Jgov from flow regime, bowl pressure, inlet pressure, inlet density, and flow
def solve_j(regime, pi, pb, r, flow):
    
    #Use notebook alpha
    global alpha
    
    if (regime == 1):

        #flow = 3.6*jgv*(r*(pi-pb))**0.5
        jgv = flow/3.6/((r*(pi-pb))**0.5)

    elif regime ==3:

        #flow = 3.6*jgv*(r*(pi-pb))**0.5
        jgv = flow/3.6/(r*(pi-pb))**0.5

    elif regime == 2:

        #flow = 3.6*jgv*(r*(alpha*pi))**0.5
        jgv = flow/3.6/(r*(alpha*pi))**0.5

    elif regime ==4:
        
        #PLACEHOLDER
        #flow = 3.6*jgv*(r*(alpha*pi))**0.5
        jgv = flow/3.6/(r*(alpha*pi))**0.5

    return jgv

    
# ## Define Trim Curves
# The lifts to evaluate at are pre-determined
def define_trim(res, trim_guesses):
    
    global gov_no

    #guesses = [[gv1 0, 50, 100, 25....], [gv2 0, 50, 100, 25...]

    # initialize
    trims_list = []
    
    for x in range(gov_no):
        
        lift_list = [0,100]
        area_list = [0,100]
        
        #Define the lifts
        for i in range(res):
            lift_list.insert(i+1, 100/(1+res)*(i+1))
        
            area_list.insert(i+1, trim_guesses[x][i])
    
        trims_list.append(lift_list)
        trims_list.append(area_list)

    #Create a list of column titles for Pandas dataframe
    titles = []
    for i in range(gov_no):
        titles.append("GV"+str(i+1)+" lift")
        titles.append("GV"+str(i+1)+" area")

    #Define a dictionary of the governor trim curves
    dictionary = {}
    for i in range(gov_no):
        dictionary.update({titles[2*i] : trims_list[2*i]})
        dictionary.update({titles[2*i+1] : trims_list[2*i+1]})

    #Use dictionary to create a pandas dataframe
    gv_trim_df = pd.DataFrame(dictionary)

    #Save Dataframe to csv
    gv_trim_df.to_csv(f"projects/{project}/gov_trim_curve.csv")
    
    return gv_trim_df


def calc_ks(performance_df, gov_noz_ratios):
    #---------------------------------------------
    #---------------------------------------------
    #
    # Calculate the govenor and nozzle flow coeffiecients for the given governor_k / nozzle_k ratio
    #
    #---------------------------------------------
    #---------------------------------------------
    global gov_no
    
    #determine the flow at 100% governor demand 
    max_flow = performance_df.loc[performance_df["Governor Demand"] == 100, "Steam Flow Target"].to_list()[0]

    #Determine the throttle pressure at 100% governor demand
    max_tp = performance_df.loc[performance_df["Governor Demand"] == 100, "Throttle Pressure"].to_list()[0]

    #determine the p_ratio at 100% governor demand
    max_p_ratio = performance_df.loc[performance_df["Governor Demand"] == 100, "Px/Pi"].to_list()[0]

    #determine the bowl densities at 100% governor demand   
    max_rho = performance_df.loc[performance_df["Governor Demand"] == 100, "Inlet Density"].to_list()[0]
    
    #determine the first stage pressure at 100% governor demand
    max_fsp = performance_df.loc[performance_df["Governor Demand"] == 100, "First Stage Pressure"].to_list()[0]

    
    max_regime = []
    max_pb = []
    max_rb = []
    factor = 0
    #For each governor valve, determine the flow regime at 100% governor demand
    for i in range(gov_no):
        max_regime.append(get_regime(gov_noz_ratios[i], max_p_ratio))
        
        #Calculate the bowl pressure
        max_pb.append(max_tp * calc_pbr(max_regime[i], gov_noz_ratios[i], max_p_ratio))
        
        #determine the bowl densities at 100% governor demand
        max_rb.append(max_rho*max_pb[i]/max_tp)
                      
        if max_regime[i] < 3:
        #nozzle not choked
                      
            factor = factor + (max_rb[i]*(max_pb[i] - max_fsp))**0.5


        else:
        #nozzle choked
            
            factor = factor + (max_rb[i]*alpha*max_pb[i])**0.5

                      
    #Calculate the necessary jgov and jnoz
    try:
        noz_k = max_flow/3.6/factor
    except:
        print(f"factor = {factor}, max_flow = {max_flow}, max_tp = {max_tp}, max_p_ratio = {max_p_ratio}, max_rho = {max_rho}, max_fsp = {max_fsp}")
    
    #Create list for governor valve Ks
    gov_ks = []
    for i in range(gov_no):
        gov_ks.append(noz_k*gov_noz_ratios[i])
    
    return (gov_ks, noz_k)

def unravel_parameters(parameters):
    # This function takes a list with the following structure:
    # [Avg Gov/Noz Ratio, GV1 gov/noz var...GVX gov/noz var, GV1 trim decrement factors...GVX Trim decrement factors]
    # and returns a list of the individual govnernor valve K / nozzle K ratios, and a list of
    # lists with the following form [[GV1 trim guesses], [GV2 trim guesses]...[GVx trim guesses]]
    
    # Determine how many trim points are being used
    res = int((len(parameters) - (gov_no+1))/gov_no)
    
    # Get the first gov_no+1 parameters and turn it into individual gov/noz size ratios
    gov_noz_inputs = parameters[:(gov_no+1)]
    
    # Create empty list to hold the individual gov / noz k ratios
    gov_noz_ratios = []
    
    # For each governor valve, the gov/noz k ratio is (average gov/noz ratio)*(individual variance)
    for x in range(gov_no):
        gov_noz_ratios.append(gov_noz_inputs[x+1] / gov_noz_inputs[0])
        
    # The trim-related parameters are at the tail of the guess list. Take all but the first gov_no+1 parameters
    trim_parameters = initial_guess[(gov_no+1):]
    
    # Create empty list to hold the lists of govenror trim factors
    trim_factors = []

    # Split the trim related parameters into individual lists for each governor valve
    for i in range(gov_no):
        sliceObj = slice(i*res, (i+1)*res)
        trim_factors.append(trim_parameters[sliceObj])

    # Create empty list to hold the final list of trim guesses in %
    trim_guess = []

    # Iterate over the number of governor valves
    for x in range(gov_no):

        # Create empty list to hold the current governor valve's trim guesses in %
        trim_guess_indiv = []

        # If only one point in the lift vs. area curve need to be defined, use an alternative calculation
        if res == 1:

            # The guess for the first trim point is 100*(first trim factor)
            trim_guess_indiv.append(100*trim_factors[x][0])
        else:

            # The guess for the first trim point is 100*(first trim factor)
            trim_guess_indiv.append(100*trim_factors[x][0])

            # The remaining trim points are (previous area)*(next trim factor)
            for i in range(res-1):
                trim_guess_indiv.append(trim_guess_indiv[i]*trim_factors[x][i+1])

        # Reverse the trim guesses so that the smaller lifts are at the start of the list
        trim_guess_indiv.reverse()

        # Add the individual trim points to the over trim guess, then move on
        trim_guess.append(trim_guess_indiv)
              
    return (gov_noz_ratios, trim_guess)



def calc_performance(parameters, performance_df, plot_flag, iteration):

    global gov_no
    
    # Determine how many trim points are being used
    res = int((len(parameters) - (gov_no+1))/gov_no)
    
    (gov_noz_ratios, trim_guess) = unravel_parameters(parameters)
    (gov_k_list, noz_k) = calc_ks(performance_df, gov_noz_ratios)
    trim_df = define_trim(res, trim_guess)

    #---------------------------------------------
    #---------------------------------------------
    #
    # Calculate Flow Coefficients and Predicted Flows 
    #
    #---------------------------------------------
    #---------------------------------------------

    global single_dmd_df
    global seq_demand_df
    
    gov_dmd = performance_df.loc[:,"Governor Demand"].to_list()
    p_ratio = performance_df.loc[:, "Px/Pi"].to_list()
    tp = performance_df.loc[:, "Throttle Pressure"].to_list()
    rho = performance_df.loc[:, "Inlet Density"].to_list()

    for i in range(gov_no):

        #ITERATE OVER GVs
        #re-initialize problem
        single_j_list = []
        single_jratio_list = []
        single_region_list = []
        single_pb_list = []
        single_f_list = []

        seq_j_list = []
        seq_jratio_list = []
        seq_region_list = []
        seq_pb_list = []
        seq_f_list = []

        for x in range(len(gov_dmd)):

                #----------------------------------------------------------
                #Calculate the jratio and flow regime for single valve mode
                #----------------------------------------------------------

                #Calculate the valve lift
                try:
                    single_lift = np.interp(gov_dmd[x], single_dmd_df["Governor Demand"], single_dmd_df["Valve Position Demand"])
                except:
                    print("single_lift")
                    
                #Calculate the valve area
                try:
                    single_j_list.append(gov_k_list[i]/100*np.interp(single_lift, trim_df[f"GV{i+1} lift"], trim_df[f"GV{i+1} area"]))
                except:
                    print("single_j_list")
                
                #calculate the jratio
                try:
                    single_jratio_list.append(single_j_list[x]/noz_k)
                except:
                    print("single_jratio_list")

                #determine the flow regime for each governor valve
                try:
                    single_region_list.append(get_regime(single_jratio_list[x], p_ratio[x]))
                except:
                    print("single_region_list")
                #----------------------------------------------------------
                # Calculate Pb and flow for single valve mode
                #----------------------------------------------------------
                try:
                    pbowl =  tp[x] * calc_pbr(single_region_list[x], single_jratio_list[x], p_ratio[x])
                except:
                    print("single_pbowl")
                
                single_pb_list.append(pbowl)

                #Calculate flow
                try:
                    single_f_list.append(calc_flow(single_region_list[x], tp[x], pbowl, single_j_list[x], rho[x]))
                except:
                    print("single_f_list")


                #----------------------------------------------------------
                #Calculate the jratio and flow regime for sequential valve mode
                #----------------------------------------------------------

                #Calculate the valve lift
                seq_lift = np.interp(gov_dmd[x], seq_dmd_df.iloc[:,2*i], seq_dmd_df.iloc[:,2*i+1])

                #Calculate the valve area
                seq_j_list.append(gov_k_list[i]/100*np.interp(seq_lift, trim_df[f"GV{i+1} lift"], trim_df[f"GV{i+1} area"]))
                
                #calculate the jratio
                seq_jratio_list.append(seq_j_list[x]/noz_k)

                #determine the flow regime for each governor valve
                seq_region_list.append(get_regime(seq_jratio_list[x], p_ratio[x]))

                #----------------------------------------------------------
                # Calculate Pb and flow for sequential valve mode
                #----------------------------------------------------------

                pbowl =  tp[x] * calc_pbr(seq_region_list[x], seq_jratio_list[x], p_ratio[x])
                seq_pb_list.append(pbowl)

                #Calculate flow in KPPH
                seq_f_list.append(calc_flow(seq_region_list[x], tp[x], pbowl, seq_j_list[x], rho[x]))



        performance_df[f"Single Flow Coefficient {i+1}"] = single_j_list
        performance_df[f"Single Flow Region {i+1}"] = single_region_list
        performance_df[f"Single J Ratio {i+1}"] = single_jratio_list
        performance_df[f"Single Bowl Presusre {i+1}"] = single_pb_list
        performance_df[f"Single Flow {i+1}"] = single_f_list


        performance_df[f"Sequential Flow Coefficient {i+1}"] = seq_j_list
        performance_df[f"Sequential Flow Region {i+1}"] = seq_region_list
        performance_df[f"Sequential J Ratio {i+1}"] = seq_jratio_list
        performance_df[f"Sequential Bowl Presusre {i+1}"] = seq_pb_list
        performance_df[f"Sequential Flow {i+1}"] = seq_f_list

        #ITERATE OVER GVs


    single_flow_list = []
    seq_flow_list = []

    for i in range(len(gov_dmd)):

        total_flow_single = 0
        total_flow_seq =0
        for x in range(gov_no):
            total_flow_single += performance_df[f"Single Flow {x+1}"].tolist()[i]
            total_flow_seq  += performance_df[f"Sequential Flow {x+1}"].tolist()[i]

        single_flow_list.append(total_flow_single)
        seq_flow_list.append(total_flow_seq)

    performance_df["Total Single Flow"] = single_flow_list
    performance_df["Total Sequential Flow"] = seq_flow_list

    #---------------------------------------------
    #---------------------------------------------
    #
    # END: Calculate Flow Coefficients and Predicted Flows 
    #
    #---------------------------------------------
    #---------------------------------------------

      
    #Calculate error
    error = ((performance_df["Total Single Flow"] - performance_df["Steam Flow Target"])**2 + \
            (performance_df["Total Sequential Flow"] - performance_df["Steam Flow Target"])**2).sum()
    
    #Save the plot if the current error is lowest error so far
    if plot_flag == 1:
        fig, ax = plt.subplots()

        plt.plot(performance_df["Governor Demand"], performance_df["Steam Flow Target"])
        plt.plot(performance_df["Governor Demand"], performance_df["Total Single Flow"])
        plt.plot(performance_df["Governor Demand"], performance_df["Total Sequential Flow"])
        plt.legend(["Target", "Single", "Sequential"],loc="best")
        plt.title("Steam Flow vs. Governor Demand")
        plt.xlabel("Govenor Demenad\n(%)")
        plt.ylabel("Steam Flow\n (KPPH)")
        plt.tight_layout()
        #plt.show()

        fig.savefig(f'projects/{project}/plot{iteration}.png')
        plt.close(fig)
        
        trim_df.to_csv(f'projects/{project}/trim{iteration}.csv')
        print(f"gov ks = {gov_k_list}")
        print(f"noz_k = {noz_k}")
    
    return error
    # # Open a Project

#Ask if user would like to open a project
open_flag = input("Would you like to return to an existing project?(y/n)")

#If user requests to open project, request project name
if open_flag == "y":
    try:
        path = askdirectory(title="Select Project Folder")
    except:
        print("File open failed.")

# In[4]:
    #get project name
    list_path = path.split(sep="/")
    open_project = list_path[-1]
    project = open_project

    #Load available dataframes
    try:
        single_dmd_df = pd.read_csv(f"projects/{open_project}/single_dmd_curve.csv")
    except:
        print("Couldn't load 'single_dmd_curve.csv' from project")
        
    try:
        seq_dmd_df = pd.read_csv(f"projects/{open_project}/sequential_dmd_curve.csv")
        #Use integer division to get number of governor valves
        gov_no = len(seq_dmd_df.columns)//2
    except:
        print("Couldn't load 'sequential_dmd_curve.csv' from project")
        
    try:
        design_df = pd.read_csv(f"projects/{open_project}/design_data.csv")
    except:
        print("Couldn't load 'design_data.csv' from project")
    
    try:
        steam_flow_model_df = pd.read_csv(f"projects/{open_project}/steam_flow_model.csv")
    except:
        print("Couldn't load 'design_data.csv' from project")
    
    try:
        tp_model_df = pd.read_csv(f"projects/{open_project}/tp_model.csv")
    except:
        print("Couldn't load 'design_data.csv' from project")
    
    try:
        steam_temp_model_df = pd.read_csv(f"projects/{open_project}/steam_temp_model.csv")
    except:
        print("Couldn't load 'design_data.csv' from project")
    
    try:
        first_stage_model_df = pd.read_csv(f"projects/{open_project}/first_stage_model.csv")
    except:
        print("Couldn't load 'design_data.csv' from project")
    
# # Define a New Project

else:
    # Get a project name from the user
    project = input("What would you like to call this project?")

    # Create a Directory for the project
    try:
        os.mkdir("projects")
    except:
        print("Couldn't make new 'projects' directory")
    try:
        os.mkdir(f"projects/{project}")
    except:
        print(f"Couldn't make new '{project}' directory")

    # Load data to define Single Mode Governor Demand to Valve Position Curve
    single_dmd_df = pd.read_csv('data/single_dmd_curve.csv')

    # Save Dataframe to csv in project folder
    single_dmd_df.to_csv(f"projects/{project}/single_dmd_curve.csv")

    # Plot the governor demand single valve curve
    fig1 = plt.plot(single_dmd_df["Governor Demand"], single_dmd_df["Valve Position Demand"], "o-")
    plt.title("Single Valve Mode\nValve Position Demand vs Governor Demand\n")
    plt.ylabel("Valve Position Demand (%)")
    plt.xlabel("\nGovernor Valve Demand (%)")
    plt.tight_layout()
    plt.show()

    # Save the figure to the project directory
    plt.savefig(f"projects/{project}/single_dmd_curve.png")


    # Load data to define Sequential Mode Governor Demand to Valve Positions Curves
    seq_dmd_df = pd.read_csv("data/sequential_dmd_curve.csv")
    
    # Get number of governor valves from the user
    gov_no = int(input("How many governor valves are there?"))

    # Save Dataframe to csv in project folder
    seq_dmd_df.to_csv(f"projects/{project}/sequential_dmd_curve.csv")
    
    # Plot trim curves
    
    # Plot will have a common x axis for all plots. Plots stacked vertically for easy compare
    fig, axes = plt.subplots(gov_no, sharex=True, figsize = (5,10))
    
    # Iterate over the number of governor valves and create a plot for each with label and legend
    for i in range(gov_no):
        axes[i].plot(seq_dmd_df.iloc[:,i*2], seq_dmd_df.iloc[:,i*2+1], 'o-', label = f"GV{i+1}" )
        axes[i].set_ylabel(f'Lift (%)')
        axes[i].legend(loc="upper left")
    
    # Set x axis label on bottom figure only
    axes[(gov_no-1)].set_xlabel('Governor Demand (%)')
    
    # Set figure title
    fig.suptitle("Sequential Mode - Valve Lift vs. Governor Demand")
    
    # Use tight layout to bring the plot under the super title [left, bottom, right, top]
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Save figure for future reference
    fig.savefig(f'projects/{project}/sequential_dmd_curve.png')

    # Load data to define the Design Performance
    design_df = pd.read_csv("data/design_data.csv")
    
    # Save dataframe to csv in the project folder
    design_df.to_csv(f"projects/{project}/design_data.csv")
    
    # Calculate the Inlet Density and Equivalent J = F/sqrt(rho*dp) based on input data
    # Create empty lists to hold calculations
    rho = []
    factor = []
    
    # Convert Dataframe to lists
    f = design_df["Steam Flow (Design)"].to_list()
    tp = design_df["Throttle Pressure (Design)"].to_list()
    fsp = design_df["First Stage Pressure (Design)"].to_list()
    t = design_df["Main Steam Temperature (Design)"].to_list()
    
    # For each data point, calculate inlet density and equivalent J
    for i in range(len(design_df)):
        rho.append(calc_rho(tp[i], t[i]))
        
        # If pressure drop is choked use alpha instead of total pressure drop
        if (tp[i] - fsp[i]) > alpha:
            dp = alpha*tp[i]
        else:
            dp = tp[i] - fsp[i]
            
        factor.append(f[i]/(rho[i]*dp)**0.5)
    
    # Add columns for inlet density and Equivalent J
    design_df["Inlet Density (Design)"] = rho
    design_df["Equivalent J (Design)"] = factor
    
    # Plot equivalent J vs Governor Demand
    x_axis = design_df.loc[ design_df["Governor Demand (Design)"] > 0, "Governor Demand (Design)"].to_list()
    y_axis = design_df.loc[ design_df["Governor Demand (Design)"] > 0, "Equivalent J (Design)"].to_list()
    
    # Insert point (0,0)
    x_axis.insert(0,0)
    y_axis.insert(0,0)

    # Create 100 data points for the fit
    model_x = np.linspace(min(x_axis) - 0.01*(max(x_axis) - min(x_axis)), max(x_axis) + 0.01*(max(x_axis) - min(x_axis)), 100)

    # Calculate linear fit to start with
    (slope, intercept, _, _, _) = linregress(x_axis, y_axis)
    fit_x_data = model_x
    fit_y_data = slope * model_x + intercept

    # Calculate lower boundary for outliers
    
    # The x points are the fit curve points
    lb_model_x = fit_x_data
    
    # The y-points are the fit curve points - value
    lb_model_y = fit_y_data - 0.05*mean(fit_y_data)

    # calculate upper boundary for outliers
    
    # The x points are the fit curve points
    ub_model_x = fit_x_data
    
    # The y-points are the fit curve points - value
    ub_model_y = fit_y_data + 0.05*mean(fit_y_data)
    
    # For each data point, determine if it is oustide the lower boundary or upper boundary via linear interpolation
    ll_list = np.interp( x_axis, lb_model_x, lb_model_y)
    ul_list = np.interp( x_axis, ub_model_x, ub_model_y)
    
    outlier_x_data = []
    outlier_y_data = []
    filt_x_data = []
    filt_y_data = []
    for i in range(len(y_axis)):
        if (y_axis[i] > ul_list[i]) or (y_axis[i] < ll_list[i]):
            outlier_x_data.append(x_axis[i])
            outlier_y_data.append(y_axis[i])
            filt_x_data.append(x_axis[i])
            filt_y_data.append(np.nan)
        else:
            outlier_x_data.append(x_axis[i])
            outlier_y_data.append(np.nan)
            filt_x_data.append(x_axis[i])
            filt_y_data.append(y_axis[i])

    # Create figure and axis objects
    fig, ax = plt.subplots()

    # Plot data identified as outliers in gray
    outliers = plt.scatter( outlier_x_data, outlier_y_data, color='black', alpha=0.1)

    # Plot curve-fit as a dotted blue line
    fit, = plt.plot(model_x, slope * model_x + intercept, '--')

    # Plot in red, semi-opaque
    lb, = plt.plot( lb_model_x, lb_model_y, color='red', alpha=0.2)

    # Plot in red, semi-opaque
    ub, = plt.plot( ub_model_x, ub_model_y, color='red', alpha=0.2)

    # Plot data no identified as outliers in blue
    filt_data = plt.scatter( filt_x_data, filt_y_data, alpha=0.75)

    # Set plot range to 1% outside the values in the data
    plt.xlim( min(x_axis) - 0.01*(max(x_axis) - min(x_axis)), max(x_axis) + 0.01*(max(x_axis) - min(x_axis)) )

    #left, bottom, width, height
    axRadio  = plt.axes([0.4375, 0.01, 0.125, 0.2])
    butRadio = RadioButtons(axRadio, ('linear', 'x^2', 'x^0.5', 'x^3', 'x^4', 'x^y', '2-piecewise'))

    #left, bottom, width, height
    ax_tolerance = plt.axes([0.25, 0.22, 0.50, 0.02])

    #Create slider animations for the breakpoints
    slider1 = Slider(ax_tolerance, 'Outlier Tolerance', 0, 1, valinit=0.05)
    
    # [left, bottom, right, top]
    plt.tight_layout(rect=[0, 0.25, 1, 0.95])
    
    radioValue = 0
    def update(val):
        
        # Store the radio button selection in a variable
        radioValue = butRadio.value_selected

        # For each selection, pick curve fit to optimize with
        if radioValue == 'linear':            
            # Get best linear fit for all data
            (slope_new, intercept_new, r_value, p_value, stderr) = linregress(x_axis, y_axis)
            
            # calculate y-values using fit function
            fit_x_data = model_x
            fit_y_data = slope_new * model_x + intercept_new
            #print(fit_y_data)
            # Calculate lower boundary for outliers
    
            # The x points are the fit curve points
            lb_model_x = fit_x_data

            # The y-points are the fit curve points - value
            lb_model_y = fit_y_data - slider1.val*mean(fit_y_data)

            # calculate upper boundary for outliers

            # The x points are the fit curve points
            ub_model_x = fit_x_data

            # The y-points are the fit curve points - value
            ub_model_y = fit_y_data + slider1.val*mean(fit_y_data)

            # For each data point, determine if it is oustide the lower boundary or upper boundary via linear interpolation
            ll_list = np.interp( x_axis, lb_model_x, lb_model_y)
            ul_list = np.interp( x_axis, ub_model_x, ub_model_y)

            outlier_y_data = []
            filt_y_data = []
            for i in range(len(y_axis)):
                if (y_axis[i] > ul_list[i]) or (y_axis[i] < ll_list[i]):
                    outlier_y_data.append(y_axis[i])
                    filt_y_data.append(np.nan)
                else:
                    outlier_y_data.append(np.nan)
                    filt_y_data.append(y_axis[i])

            # Reset outlier data points
            outliers.set_offsets(np.c_[outlier_x_data, outlier_y_data])
            
            # Reset best fit line data points
            fit.set_ydata(fit_y_data)

            # Plot lower boundary for outliers
            lb.set_ydata( lb_model_y)

            # Plot upper boundary for outliers
            ub.set_ydata( ub_model_y)
            
            # Reset filtered data points
            filt_data.set_offsets(np.c_[filt_x_data, filt_y_data])

            # Display equation as the plot title
            fig.suptitle('y = ' +'{:.3e}'.format(slope_new) + " * x + " + '{:.3e}'.format(intercept_new))

            # redraw canvas
            fig.canvas.draw()
            fig.canvas.flush_events()

        if radioValue == 'x^0.5':            
            # Get best quadratic fit
            popt, pcov = optimize.curve_fit(root, x_axis, y_axis)

            # calculate y-values using fit function. Numpy does not allow fractional powers of negative numbers: requiring the extra math
            y_axis_new = popt[0]*(np.sign(model_x)*np.abs(model_x)**0.5) + popt[1]

            # Reset best fit line data points
            fit.set_ydata(y_axis_new)

            # Plot lower boundary for outliers
            lb.set_ydata( y_axis_new - slider1.val*y_axis_new)

            # Plot upper boundary for outliers
            ub.set_ydata( y_axis_new + slider1.val*y_axis_new)

            # Display equation as the plot title
            fig.suptitle('y = ' +'{:.3e}'.format(popt[0]) + " * x^0.5 + " + '{:.3e}'.format(popt[1]))

            # redraw canvas while idle
            fig.canvas.draw()
            fig.canvas.flush_events()

        if radioValue == 'x^2':
            # Get best quadratic fit
            popt, pcov = optimize.curve_fit(quadratic, x_axis, y_axis)

            # calculate y-values using fit function
            y_axis_new = popt[0]*(model_x**2) + popt[1]*(model_x) + popt[2]

            # Reset best fit line data points
            fit.set_ydata(y_axis_new)

            # Plot lower boundary for outliers
            lb.set_ydata( y_axis_new - slider1.val*y_axis_new)

            # Plot upper boundary for outliers
            ub.set_ydata( y_axis_new + slider1.val*y_axis_new)

            # Display equation as the plot title
            fig.suptitle('y = ' +'{:.3e}'.format(popt[0]) + " * x^2 + " + '{:.3e}'.format(popt[1]) + " * x + " + '{:.3e}'.format(popt[2]))

            # redraw canvas while idle
            fig.canvas.draw()
            fig.canvas.flush_events()
        if radioValue == 'x^3':
            # Get best cubic fit
            popt, pcov = optimize.curve_fit(cubic, x_axis, y_axis)

            # calculate y-values using fit function
            y_axis_new = popt[0]*(model_x**3) + popt[1]*(model_x**2) + popt[2]*(model_x) + popt[3]

            # Display equation as the plot title
            fig.suptitle('y = ' +'{:.3e}'.format(popt[0]) + " * x^3 + " + '{:.3e}'.format(popt[1]) + " * x^2 + " + '{:.3e}'.format(popt[2]) + " * x + " + '{:.3e}'.format(popt[3]))
            
            # Reset best fit line data points
            fit.set_ydata(y_axis_new)

            # Plot lower boundary for outliers
            lb.set_ydata( y_axis_new - slider1.val*y_axis_new)

            # Plot upper boundary for outliers
            ub.set_ydata( y_axis_new + slider1.val*y_axis_new)

            # redraw canvas while idle
            fig.canvas.draw()
            fig.canvas.flush_events()
        if radioValue == 'x^4':
            # Get best fourth power fit
            popt, pcov = optimize.curve_fit(fourth, x_axis, y_axis)

            # calculate y-values using fit function
            y_axis_new = popt[0]*(model_x**4) + popt[1]*(model_x**3) + popt[2]*(model_x**2) + popt[3]*(model_x) + popt[4]

            # Display equation as the plot title
            fig.suptitle('y = ' +'{:.3e}'.format(popt[0]) + " * x^4 + " + '{:.3e}'.format(popt[1]) + " * x^3 + " + '{:.3e}'.format(popt[2]) + " * x^2 + "\
                 + '{:.3e}'.format(popt[3]) + " * x + " + '{:.3e}'.format(popt[4]) )
            
            # Reset best fit line data points
            fit.set_ydata(y_axis_new)
            
            # Plot lower boundary for outliers
            lb.set_ydata( y_axis_new - slider1.val*y_axis_new)

            # Plot upper boundary for outliers
            ub.set_ydata( y_axis_new + slider1.val*y_axis_new)

            # redraw canvas while idle
            fig.canvas.draw()
            fig.canvas.flush_events()
        if radioValue == 'x^y':
            # Get best power fit
            popt, pcov = optimize.curve_fit(power, x_axis, y_axis)

            # calculate y-values using fit function
            
            # Numpy does not allow fractional powers of negative numbers: requiring the extra math
            y_axis_new = popt[0]*(np.sign(model_x)*np.abs(model_x)**(popt[1])) + popt[2]

            # Reset best fit line data points
            fit.set_ydata(y_axis_new)
            
            # Plot lower boundary for outliers
            lb.set_ydata( y_axis_new - slider1.val*y_axis_new)

            # Plot upper boundary for outliers
            ub.set_ydata( y_axis_new + slider1.val*y_axis_new)

            # Display equation as the plot title
            fig.suptitle('y = ' +'{:.3e}'.format(popt[0]) + " * x^(" + '{:.3e}'.format(popt[1]) + ") + " + '{:.3e}'.format(popt[2]))

            # redraw canvas while idle
            fig.canvas.draw()
            fig.canvas.flush_events()
        if radioValue == '2-piecewise':
            # Get best piecewise fit
            popt, pcov = optimize.curve_fit(piecewise_2, x_axis, y_axis)

            # calculate y-values using fit function
            y_axis_new = piecewise_2(model_x, *popt)

            # Reset best fit line data points
            fit.set_ydata(y_axis_new)

            # Plot lower boundary for outliers
            lb.set_ydata( y_axis_new - slider1.val*y_axis_new)

            # Plot upper boundary for outliers
            ub.set_ydata( y_axis_new + slider1.val*y_axis_new)

            # Display equation as the plot title
            fig.suptitle(popt)

            # redraw canvas while idle
            fig.canvas.draw()
            fig.canvas.flush_events()

    
    butRadio.on_clicked(update)
    slider1.on_changed(update)

    #show plot
    # [left, bottom, right, top]
    plt.tight_layout(rect=[0, 0.25, 1, 0.95])
    plt.show()
#%%
