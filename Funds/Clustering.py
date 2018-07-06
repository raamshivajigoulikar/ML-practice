import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


all_data=pd.read_csv("C:\\Users\\rgoulika\\Downloads\\Python\\Funds\\Data_Present_Combined.csv")


inv_prof=all_data.iloc[:,0:12]

inv_prof.describe()

grp_data=inv_prof


grp_data['Dummy_Exp_Profitability_HighProfit_HighRisk']=np.where((grp_data['Profitability_Looking_For']=="The maximum profitability possible. Regardless of the risk.") | \
                                     (grp_data['Profitability_Looking_For']=="5-6% yearly. Equity growth assuming a higher level of risk."),1,0)

grp_data['Dummy_Exp_Profitability_LowProfit_LowRisk']= np.where((grp_data['Profitability_Looking_For']=="I don't care about profitability. I don't want to lose money in any case.") | \
         (grp_data['Profitability_Looking_For'] == "1-2% yearly. Protection against inflation and patrimonial stability."),1,0)

grp_data.columns

grp_data['Dummy_Market_Crisis_Handling_Sell']=np.where((grp_data['Stock_Market_Crisis_Handling']=="Sell All") | \
                                                       (grp_data['Stock_Market_Crisis_Handling']=="Sell Something"),1,0)

