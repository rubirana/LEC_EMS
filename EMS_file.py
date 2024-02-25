# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 10:10:15 2021

@author: rubir
"""

import pandas as pd      
import numpy as np    
from pyomo.environ import *     
from pyomo.core.base.PyomoModel import ConcreteModel  
from data_file import *


def build_battery_model(model, Pmax_discharge, Pmax_charge,efficiency, initial,rating):
    model.T = RangeSet(0,23)  
    model.pin =   Var(model.T, domain=NonNegativeReals)
    model.pout =  Var(model.T, domain=NonNegativeReals)
    model.S =     Var(model.T, bounds=(0, rating))
    model.alpha=    Var(model.T, domain=Binary)
    model.PVbatt =  Var(model.T, domain=NonNegativeReals)
    @model.Constraint(model.T)       
    def storage_state(model, t):
        'Storage changes with flows in/out and efficiency losses'
      
        if t == 0:
           return (model.S[t] ==  (model.S[23] +(model.PVbatt[t]*efficiency*flag)
                                + (model.pin[t] * (efficiency)) 
                                - (model.pout[t] / (efficiency))))
        else:
           return (model.S[t] == (model.S[t-1] +(model.PVbatt[t]*efficiency*flag)
                                + (model.pin[t] * (efficiency)) 
                                - (model.pout[t] / (efficiency))))
    @model.Constraint(model.T)
    def discharge_constraint(model, t):
       
        "Maximum dischage within a single hour"
        return model.pout[t] <= Pmax_discharge*(1-model.alpha[t])
    @model.Constraint(model.T)
    def charge_constraint(model, t):
       
        "Maximum charge within a single hour"
        return model.pin[t] <=  Pmax_charge*model.alpha[t]   
   
       
    return model




             
    
def HEMS(load_profile,heater_profile):
   
    model = ConcreteModel()
    model.T = RangeSet(0,23)
    'Calling battery function'


    space_heater(model,heater_profile)
#    build_battery_model(model,residential_battery.Pmax_discharge,residential_battery.Pmax_charge,residential_battery.efficiency,residential_battery.initial_energy,residential_battery.maximum_energy)
 
          
    model.pgrid_import=   Var( model.T,domain=NonNegativeReals)


    
    @model.Constraint(model.T) 
    def balance_constraint(model,t):        
       # return (model.pout[t] ) +PVList[t].PVpower+model.pgrid_import[t]-model.pgrid_export[t] == (LoadList[t].Activepower)+(model.pin[t] ) +model.Pdr[t]
         
       return model.pgrid_import[t] == load_profile[t] + model.omega_t[t]
    
    
    @model.Objective()
    def obj(model):
        return  sum( model.pgrid_import[t]*PriceList[t].Price  for t in model.T)         
    return model




  


def space_heater(model,heater_profile):
    
    
    file_data = 'Profiles\Thermal_load_HFa.xlsx'
    xl_data = pd.ExcelFile(file_data)
    
   
    xl_data_sheet_Heater_Specs = xl_data.parse('Heater_Specs')
    xl_data_sheet_room_parameters = xl_data.parse('room_requirement')
    
    # select the sheet
    xl_data_sheet_price = xl_data.parse('El_price')
    
    EL_price = xl_data_sheet_price['Price'].to_dict()
    Flex_price = xl_data_sheet_price['fprice'].to_dict() 
   
 
    Heater_rate = xl_data_sheet_Heater_Specs['H_max'].to_dict()
    
    W_upper = xl_data_sheet_room_parameters['W_u'].to_dict()
    W_lower = xl_data_sheet_room_parameters['W_l'].to_dict()
    W_Set = xl_data_sheet_room_parameters['W_s'].to_dict()
    h=pd.DataFrame(heater_profile)
    #print(h)
    W = h.to_dict()
    W_loss=W[0]
    #print(W_loss)
    TC_Start = xl_data_sheet_room_parameters['TC_start'].to_dict()
    TC_end = xl_data_sheet_room_parameters['TC_end'].to_dict()
    w_residue = xl_data_sheet_room_parameters['w_r_t0'].to_dict()
    D_minimum = xl_data_sheet_room_parameters['D_min'].to_dict()
    D_maximum = xl_data_sheet_room_parameters['D_max'].to_dict()
    N_Maximum = xl_data_sheet_room_parameters['N_max'].to_dict()
    
    
    periodcount = xl_data_sheet_price.index
    
    
    

    model.periods = Set(initialize=periodcount)
    
    model.eprice     = Param(model.periods, initialize= EL_price, within=PositiveReals )
    model.fprice     = Param(initialize= Flex_price[0], within=PositiveReals )
    model.W_u = Param(model.periods, initialize= W_upper, within=PositiveReals )
    model.W_l = Param(model.periods, initialize= W_lower, within=PositiveReals )
    model.W_s = Param(model.periods, initialize= W_Set, within=PositiveReals )
    model.w_out = Param(model.periods, initialize= W_loss )
    model.H_max = Param(initialize=Heater_rate[0], within=PositiveReals )
    model.w_r_t0 = Param(initialize=w_residue[0], within=Reals )
    model.TC_start = Param(initialize=TC_Start[0],within=PositiveReals )
    model.TC_end = Param(initialize=TC_end[0], within=PositiveReals )
    model.D_min = Param(initialize= int(D_minimum[0]), within=NonNegativeIntegers )
    model.D_max = Param(initialize= int(D_maximum[0]), within=NonNegativeIntegers )
    model.N_max = Param(initialize= int(N_Maximum[0]), within=NonNegativeIntegers )
    
    model.w_r_t = Var( model.periods, within=Reals )
    model.omega_t = Var( model.periods, within=PositiveReals )
    model.energy_cost = Var( within=PositiveReals )
    model.flex_cost = Var( within=PositiveReals )
    model.total_cost = Var( within=PositiveReals )
    
    model.delta_run = Var( model.periods, within=Binary )
    model.delta_start = Var( model.periods, within=Binary )
    model.delta_end = Var( model.periods, within=Binary )
    

  
    @model.Constraint(model.periods)    
    def w_room_rule(model, t):
        if t==0:
            return model.w_r_t[t] == model.w_r_t0 + model.omega_t[t] - model.w_out[t]
        else:
            return model.w_r_t[t] == model.w_r_t[t-1] + model.omega_t[t] - model.w_out[t]

    
    @model.Constraint(model.periods)    
    def w_room_rule_control_period(model, t):
        if t<model.TC_start:
            return model.delta_run[t] == 0    
        elif t>=model.TC_end:
            return model.delta_run[t] == 0
        else:
            return Constraint.Skip
    

    
    @model.Constraint(model.periods)    
    def omega_t_max_rule(model, t):
        return model.omega_t[t] <= model.H_max
    
    @model.Constraint(model.periods)    
    def w_t_max_rule(model, t):
        return model.w_r_t[t] <= model.delta_run[t]*model.W_u[t] + (1-model.delta_run[t])*model.W_s[t]
    

    @model.Constraint(model.periods)    
    def w_t_min_rule(model, t):
        return model.w_r_t[t] >= model.delta_run[t]*model.W_l[t] + (1-model.delta_run[t])*model.W_s[t]
    

    
    
    @model.Constraint(model.periods)    
    
    def flex_run_rule(model, t):
        if t==0:        
            return model.delta_run[t] == model.delta_start[t]-model.delta_end[t]
        else:
            return model.delta_run[t]-model.delta_run[t-1] == model.delta_start[t]-model.delta_end[t]
    

    @model.Constraint(model.periods)    
    def flex_end_rule2(model, t):
        return model.delta_start[t] + model.delta_end[t] <= 1
    

   
    
   
def LEC_EMS(total_profile):
    model = ConcreteModel()
    model.T = RangeSet(0,23)
    'Calling battery function'
    build_battery_model(model,community_battery.Pmax_discharge,community_battery.Pmax_charge,community_battery.efficiency,community_battery.initial_energy,community_battery.maximum_energy)
    
    
    
    model.pgrid_import_community=   Var( model.T)
   
    
    @model.Constraint(model.T) 
    def balance_constraint(model,t):        
        return (model.pout[t] ) +18*PVList[t].PVpower+model.pgrid_import_community[t] == (total_profile[t])+(model.pin[t] )    
    @model.Objective()
    def obj(model):
        return  sum( model.pgrid_import_community[t]*PriceList[t].Price for t in model.T)
     
    return model  

def industry_EMS():
    model = ConcreteModel()
    model.T = RangeSet(0,23)
    'Calling battery function'
    build_battery_model(model,community_battery.Pmax_discharge,community_battery.Pmax_charge,community_battery.efficiency,community_battery.initial_energy,community_battery.maximum_energy)
    
    model.bin=Var(model.T, domain=Binary)
    model.PVgrid =  Var(model.T, domain=NonNegativeReals)
    
   
   
    
    @model.Constraint(model.T)
    def PV_constraint(model,t):
       return  model.PVbatt[t]+model.PVgrid[t]==18*PVList[t].PVpower
    @model.Constraint(model.T)
    def M_constraint1(model,t):
       return  5000*model.bin[t]+model.PVbatt[t]<=5000
   
    def M_constraint2(model,t):
        return 5000*(1-model.bin[t])+model.pin[t]<=5000
    @model.Objective()
    def obj(model):
        return  sum (model.pin[t]*PriceList[t].Price for t in model.T)- sum((model.pout[t]+model.PVgrid[t])*PriceList[t].Price  for t in model.T)
    return model  

    

def flexibility_measure(total_profile):            
    model = ConcreteModel()
    model.T = RangeSet(0,23)
    'Calling battery function'
    build_battery_model(model,community_battery.Pmax_discharge,community_battery.Pmax_charge,community_battery.efficiency,community_battery.initial_energy,community_battery.maximum_energy)
    
    
    
    model.pgrid_import_community=   Var( model.T)
   
    # @model.Constraint(model.T) 
    # def balance2_constraint(model,t):        
    #     return (model.pgrid_import_community[t]<=500 )    
    
    @model.Constraint(model.T) 
    def balance_constraint(model,t):        
        return (model.pout[t] ) +18*PVList[t].PVpower+model.pgrid_import_community[t] == (total_profile[t])+(model.pin[t] )    
    
    # @model.Objective()
    # def obj(model):
    #     return  sum( model.pgrid_import_community[t]*PriceList[t].Price for t in model.T)
     
    
    @model.Objective()
    def obj(model):
        return  sum( (model.pgrid_import_community[t]-450)**2 for t in model.T) #450 kW grid import limit
    
   
    return model


def flexibility_measure_2(total_profile):            
    model = ConcreteModel()
    model.T = RangeSet(0,23)
    'Calling battery function'
    build_battery_model(model,community_battery.Pmax_discharge,community_battery.Pmax_charge,community_battery.efficiency,community_battery.initial_energy,community_battery.maximum_energy)   
    model.pgrid_import_community=   Var( model.T)  
    @model.Constraint(model.T) 
    def balance2_constraint(model,t):        
        return (model.pgrid_import_community[t]<=450 )  #450 kW grid import limit      
    @model.Constraint(model.T) 
    def balance_constraint(model,t):        
        return (model.pout[t] ) +18*PVList[t].PVpower+model.pgrid_import_community[t] == (total_profile[t])+(model.pin[t] )        
    @model.Objective()
    def obj(model):
        return  sum( model.pgrid_import_community[t]*PriceList[t].Price for t in model.T)               
   
    return model


customer_list = []
water_heater_list=[]
customer_list += [MeasurementPoint(df_24_load[:, i]) for i in range(df_24_heater.shape[1] )] 
water_heater_list  += [heatermeasurement(df_24_heater[:, i]) for i in range(df_24_heater.shape[1] )] 
def power(measurement):
     
    return measurement.load

def water_heater_data(measurement):
    return measurement.heater_data


def solve(load_profile,heater_profile):
     
     model=HEMS(load_profile,heater_profile)  
    # model.pprint()
     #solverpath_exe = 'C:\\glpk-4.65\\w64\\glpsol'
     #solver = SolverFactory('glpk', executable=solverpath_exe)
     solver=  SolverFactory('glpk')
     solver.solve(model)
     pgrid_import=resultswrite(model)
     return  pgrid_import


def solve_LEC_EMS(total_profile):
    if congestion_case==0:
      model=LEC_EMS(total_profile)
    else:
      model=flexibility_measure_2(total_profile)  
      #Uncomment this if we want to do  load levelling
      #model=flexibility_measure(total_profile)
      
    #solverpath_exe = 'C:\\glpk-4.65\\w64\\glpsol'
    #solver = SolverFactory('glpk', executable=solverpath_exe)
    solver=  SolverFactory('glpk')
    solver.solve(model)
    pgrid_import_community,dff=resultswrite_community(model)
    return  pgrid_import_community,dff

def solve_industry_EMS():
    model=industry_EMS()
    # model.pprint()
    #solverpath_exe = 'C:\\glpk-4.65\\w64\\glpsol'
    #solver = SolverFactory('glpk', executable=solverpath_exe)
    solver=  SolverFactory('glpk')
    solver.solve(model)
    df_industry=resultswrite_industry(model)
    return  df_industry

def resultswrite(model):
    pgrid_import= [value(model.pgrid_import[i]) for i in model.T]
    
  
   
    return pgrid_import

def resultswrite_community(model):
    pgrid_import_community= [value(model.pgrid_import_community[i]) for i in model.T]
   
    pin = [value(model.pin[i])  for i in model.T]  
    
    pout = [value(model.pout[i])  for i in model.T]       
    charge_state = [value(model.S[i]) for i in model.T]  
    df_dict = dict(       
       pin=pin,
       pout=pout,      
       charge_state=charge_state,      
    )

    dff = pd.DataFrame(df_dict)
    return pgrid_import_community,dff
def resultswrite_industry(model):
    #pgrid_import_industry= [value(model.pgrid_import_community[i]) for i in model.T]
    
    pin = [value(model.pin[i])  for i in model.T]  
    
    pout = [value(model.pout[i])  for i in model.T]       
    charge_state = [value(model.S[i]) for i in model.T]  
    df_dict = dict(       
       pin=pin,
       pout=pout,      
       charge_state=charge_state,      
    )

    df_industry = pd.DataFrame(df_dict)
    return df_industry



def scaling_load(scaling_factor,active_power):
  
  my_new_list_active  =[i*scaling_factor for i in active_power] 
 
  return my_new_list_active

original_load=[]
  
#Simulating Case studies   
results_main=[]
for i,k in zip(customer_list,water_heater_list): 
   load_profile  =     power(i)   
   heater_profile=     water_heater_data(k)
   total=  load_profile+heater_profile
   original_load.append(total)
   pgrid_import=solve(load_profile,heater_profile) #HEMS optimization
   results_main.append(pgrid_import)     

# results_main is stoing results from HEMS optimisation
dfr = pd.DataFrame(results_main)  
dfr.to_excel(r"results_grid_import.xlsx") # storing resulst for individual customers 

j=(dfr.sum(axis = 0)) # summing grid import for 25 customers to be used in LEC community optimisation 
#Case#2a
# Set congestion case  is 1 to see overlaoding case else set it to zero
congestion_case=1
scaling_factor=8.15
flag=0 # used in battery model and it should be one only in case of industry
#  scaling a load 
my_new_list_active = scaling_load(scaling_factor, j)

pgrid_import_community,dff=solve_LEC_EMS(my_new_list_active)

dfr = pd.DataFrame(pgrid_import_community)  
dfr.to_excel(r"results_grid_import_community_overlaoding_prevent.xlsx")      
dfrtt = pd.DataFrame(dff)  
dfrtt.to_excel(r"resultscommunity_2_overloading_prevent.xlsx")

#Case#2b
# Set congestion case  is 1 to see overlaoding case else set it to zero
congestion_case=0
scaling_factor=8.15

#  scaling a load 
my_new_list_active = scaling_load(scaling_factor, j)

pgrid_import_community,dff=solve_LEC_EMS(my_new_list_active)

dfr = pd.DataFrame(pgrid_import_community)  
dfr.to_excel(r"results_grid_import_community_overlaoding.xlsx")      
dfrtt = pd.DataFrame(dff)  
dfrtt.to_excel(r"resultscommunity_2_overloading.xlsx")

#Case#1a
# Set congestion case  is 0 normal cost minimisation operation
congestion_case=0
scaling_factor=1

#  scaling a load 
my_new_list_active = scaling_load(scaling_factor, j)

pgrid_import_community,dff=solve_LEC_EMS(j)

dfr = pd.DataFrame(pgrid_import_community)  
dfr.to_excel(r"results_grid_import_community.xlsx")      
dfrtt = pd.DataFrame(dff)  
dfrtt.to_excel(r"resultscommunity_2.xlsx")
flag =1
#Case#1b
df_industry = solve_industry_EMS()
    
dfitt = pd.DataFrame(df_industry)  
dfitt.to_excel(r"resultsindustry.xlsx")


# original load without flexibility
df_original_load= pd.DataFrame(original_load)  
df_original_load.to_excel(r"results_grid_original.xlsx")   
k=(df_original_load.sum(axis = 0))