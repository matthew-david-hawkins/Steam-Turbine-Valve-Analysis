#!/usr/bin/env python
# coding: utf-8

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Return to an existing Project or define a new project
#
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
from matplotlib.widgets import Slider


#Define critical pressure ratio
alpha = 0.55

#Define a function that calculates the density for a given pressure and temperature
def calc_rho(p, t):
    
    #convert psia to MPA
    pressure = p*0.00689476
    
    #convert farenheit to Kelvin
    temp = (t - 32)*5/9 + 273.15
    
    #get the density 
    r = steam_prop(P=pressure, T=temp).v
    rho = steam_prop(P=pressure, T=temp).rho
    
    #convert density from kg/M^3 to lb/ft^3
    rho = rho*0.062428
    
    return rho

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
        gv_demand_df = pd.read_csv(f"projects/{open_project}/sequential_dmd_curve.csv")
        #Use integer division to get number of governor valves
        gov_no = len(gv_demand_df.columns)//2
    except:
        print("Couldn't load 'sequential_dmd_curve.csv' from project")
        
    try:
        design_df = pd.read_csv(f"projects/{open_project}/design_data.csv")
    except:
        print("Couldn't load 'design_data.csv' from project")
    
# # Define a New Project

else:
    #Get a project name from the user
    project = input("What would you like to call this project?")

    #Create a Directory for the project
    try:
        os.mkdir("projects")
    except:
        print("Couldn't make new 'projects' directory")
    try:
        os.mkdir(f"projects/{project}")
    except:
        print(f"Couldn't make new '{project}' directory")


    # # Define Single Mode Governor Demand to Valve Position Curve

    #Import Governor Demand to Single Valve Demand curve
    user_x = input("Enter the x values for Single Valve Mode")
    user_y = input("Enter the y values for Single Valve Mode")
    list_x = user_x.split()
    list_y = user_y.split()
        
    #convert list of strings to list of floats
    x_axis = [float(i) for i in list_x]
    y_axis = [float(i) for i in list_y]
        
    #Tell user what was entered
    print(f"You entered {len(x_axis)} x values and {len(y_axis)} y values")
    data = {"Governor Demand" : x_axis, "Valve Position Demand" : y_axis}
    single_dmd_df = pd.DataFrame(data)

    #Save Dataframe to csv
    single_dmd_df.to_csv(f"projects/{project}/single_dmd_curve.csv")

    #plot the governor demand single valve curve
    plt.plot(single_dmd_df["Governor Demand"], single_dmd_df["Valve Position Demand"], "o-")
    plt.title("Single Valve Mode\nValve Position Demand vs Governor Demand\n")
    plt.ylabel("Valve Position Demand (%)")
    plt.xlabel("\nGovernor Valve Demand (%)")
    plt.tight_layout()

    #save the figure to the project directory
    plt.savefig(f"projects/{project}/single_dmd_curve.png")


    # # Define Sequential Mode Governor Demand to Valve Positions Curves

    #Get number of governor valves from the user
    gov_no = int(input("How many governor valves are there?"))

    #Create a list of column titles for Pandas dataframe
    titles = []
    for i in range(gov_no):
        titles.append("GV"+str(i+1)+" x")
        titles.append("GV"+str(i+1)+" y")
    print(titles)

    #For each governor valve enter the
    gv_demand_curves = []
    for i in range(gov_no):
        user_x = input(f"Copy the x values for GV #{i+1} ")
        user_y = input(f"Copy the y values for GV #{i+1} ")
        list_x = user_x.split()
        list_y = user_y.split()
        
        #convert list of strings to list of floats
        list_x = [float(i) for i in list_x]
        list_y = [float(i) for i in list_y]
        
        #Append lists to the master list
        gv_demand_curves.append(list_x)
        gv_demand_curves.append(list_y)

    #Define a dictionary of the governor valve demand curves
    dictionary = {}
    for i in range(2*gov_no):
        dictionary.update({titles[i] : gv_demand_curves[i]})

    #Use dictionary to create a pandas dataframe
    gv_demand_df = pd.DataFrame(dictionary)
    gv_demand_df

    #Save Dataframe to csv
    gv_demand_df.to_csv(f"projects/{project}/sequential_dmd_curve.csv")

    #Create a plot of all the governor demand curves
    for i in range(gov_no):
        plt.plot(gv_demand_df.iloc[:,i*2], gv_demand_df.iloc[:,i*2+1], 'o-')

    #Create Labels
    legend_list = [f"GV{i+1}" for i in range(gov_no)]
    plt.legend(legend_list)

    #Add labels
    plt.title("Sequential Valve Mode\nValve Position Demand vs Governor Demand\n")
    plt.ylabel("Valve Position Demand (%)")
    plt.xlabel("\nGovernor Valve Demand (%)")

    #save figure
    plt.savefig(f"projects/{project}/sequential_dmd_curve.png")


    # # Define the Design Performance

    #Define the desired performance

    #get a list from the user comparing governor demand and measured steam flow
    input_dmd = input("input governor demand historical data")
    input_steam_flow = input("input steam flow historical data")
    input_throttle_p = input("input throttle pressure historical data")
    input_steam_temp = input("input main steam temperature historical data")
    input_1st_p = input("input first stage pressure historical data")
    input_mwg = input("input Generator MW historical data")

    #split input by spaces
    list_dmd = input_dmd.split()
    list_steam_flow = input_steam_flow.split()
    list_throttle_p = input_throttle_p.split()
    list_steam_temp = input_steam_temp.split()
    list_1st_p = input_1st_p.split()
    list_mwg = input_mwg.split()
        
    #convert list of strings to list of floats
    list_dmd = [float(i) for i in list_dmd]
    list_steam_flow = [float(i) for i in list_steam_flow]
    list_throttle_p = [float(i) for i in list_throttle_p]
    list_steam_temp = [float(i) for i in list_steam_temp]
    list_1st_p = [float(i) for i in list_1st_p]
    list_mwg = [float(i) for i in list_mwg]

    #Define a dictionary of the governor valve demand curves
    dictionary = {
        "Governor Demand (Design)" : list_dmd,
        "Steam Flow (Design)" : list_steam_flow,
        "Throttle Pressure" : list_throttle_p,
        "Main Steam Temperature" : list_steam_temp,
        "First Stage Pressure (Design)" : list_1st_p,
        "Generator MWG" : list_mwg
    }

    #Use dictionary to create a pandas dataframe
    design_df = pd.DataFrame(dictionary)

    #Save dataframe to csv
    design_df.to_csv(f"projects/{project}/design_data.csv")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Create Model Definitions Relating Governor Demand and Flow / Thermo 
#
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# # Create a model that predicts the Flow for a given Governor Demand

# Define functions

#piecewise linear fit
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

    return model_df


#Define Steam Flow Model
steam_flow_model_df = user_plot_fit(
    33,
    66,
    "Steam Flow vs. Governor Demand",
    design_df["Governor Demand (Design)"].tolist(),
    design_df["Steam Flow (Design)"].tolist(),
    "Governor Demand",
    "Steam Flow",
    "%",
    "KPPH"
)

steam_flow_model_df.to_csv(f"projects/{project}/steam_flow_model.csv")

#Define Throttle Pressure Model
tp_model_df = user_plot_fit(
    33,
    66,
    "Throttle Pressure vs. Governor Demand",
    design_df["Governor Demand (Design)"].tolist(),
    design_df["Throttle Pressure"].tolist(),
    "Governor Demand",
    "Throttle Pressure",
    "%",
    "psia"
)

tp_model_df.to_csv(f"projects/{project}/tp_model.csv")

#Define Main Steam Temperature Model
steam_temp_model_df = user_plot_fit(
    33,
    66,
    "Steam Temperature vs. Governor Demand",
    design_df["Governor Demand (Design)"].tolist(),
    design_df["Main Steam Temperature"].tolist(),
    "Governor Demand",
    "Steam Temp",
    "%",
    "F"
)

steam_temp_model_df.to_csv(f"projects/{project}/steam_temp_model.csv")


#Define First Stage Pressure Model
first_stage_model_df = user_plot_fit(
    33,
    66,
    "First Stage Pressure vs. Governor Demand",
    design_df["Governor Demand (Design)"].tolist(),
    design_df["First Stage Pressure (Design)"].tolist(),
    "Governor Demand",
    "First Stage Pressure",
    "%",
    "psia"
)

first_stage_model_df.to_csv(f"projects/{project}/first_stage_model.csv")

