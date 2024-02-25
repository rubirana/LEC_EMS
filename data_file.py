# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 10:00:17 2021

@author: rubir
"""
import numpy as np
import pandas as pd
from class_file import *

df1= pd.read_excel(r'Profiles\price.xlsx')
df2= pd.read_excel(r'Profiles\PV_NO3_region_kW.xlsx')

#%%
load_data = pd.read_excel(r'Profiles\fixed_load_summer_week.xlsx')   
heater_data=pd.read_excel(r'Profiles\heater_load_input.xlsx')   
df_add = load_data.add(heater_data, fill_value=0)
print(df_add) 
df3= pd.read_excel(r'time.xlsx')

#%%
price=df1.values.tolist()
price_array=np.array(price)
PV=df2.values.tolist()
PV_array=np.array(PV)


first_model_hour=df3.loc[0,'tim']
last_model_hour=df3.loc[0,'tim'] +23
slots=(last_model_hour-first_model_hour)+1

df_24_load =    load_data.iloc[first_model_hour:last_model_hour+1].to_numpy()

df_24_heater =  heater_data.iloc[first_model_hour:last_model_hour+1].to_numpy()

OnePV=PV_array[first_model_hour:last_model_hour+1,0]
dfr = pd.DataFrame(OnePV)  

oneprice=price_array[first_model_hour:last_model_hour+1,0]






PVList = []
LoadList=[]

#instance_EV_list=[]
PriceList=[]
for t in range(slots):
    EV_list=[]
    pv1=  IndividualPVpower(OnePV[t])
    price1=Individualprice(oneprice[t])
   
#    load1 =IndividualLoadpower(OneLoad[t])
   
    PVList.append(pv1)
    PriceList.append(price1)

         
 
      
#residential_battery=batteryparameters(1.2,1.2,0.9,0.6,3)
community_battery=batteryparameters(50,50,0.98,65,65)
#battery_list=[residential_battery,community_battery]
