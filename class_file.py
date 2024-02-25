# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:43:32 2021

@author: rubir
"""
import pandas as pd      
import numpy as np    
from pyomo.environ import *     
from pyomo.core.base.PyomoModel import ConcreteModel  
#from data_file import *
#%%


class MeasurementPoint:
    def __init__(self, load):
        

        
        self.load = load
class heatermeasurement:
    def __init__(self, heater_data):
        

        
        self.heater_data = heater_data

class Individualprice:
      individual_price_count=0
      
      def __init__(self, Price=0.0):
          self.Price=Price
          self.PriceList=[]
          Individualprice.individual_price_count += 1
          
class IndividualPVpower:
      individual_PV_count_power=0
      
      def __init__(self, PVpower=0.0):
          self.PVpower=PVpower
          self.PVList=[]
          IndividualPVpower.individual_PV_count_power += 1

class batteryparameters:
    battery_count=0
    def __init__(self, Pmax_discharge=0.0, Pmax_charge=0.0, efficiency=0.0, initial_energy=0.0, maximum_energy=0.0):
          self.Pmax_discharge=Pmax_discharge
          self.Pmax_charge=Pmax_charge
          self.efficiency=efficiency
          self.initial_energy= initial_energy
          self.maximum_energy=maximum_energy
          batteryparameters.battery_count += 1
    

    