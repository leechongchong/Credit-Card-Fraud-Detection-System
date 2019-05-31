# -*- coding: utf-8 -*-
"""
Credit Card Fraud Payment Analysis

"""

import pandas as pd
import time

data = pd.read_excel("Credit Card Payment Fraud Clean.xlsx")

data[["recordnum"]] = data[["recordnum"]].astype('category')
data[["cardnum"]] = data[["cardnum"]].astype('category')
data[["date"]] = data[["date"]].astype('datetime64[ns]')
data[["merchnum"]] = data[["merchnum"]].astype('category')
data[["merch.description"]] = data[["merch.description"]].astype('str')
data[["merch.state"]] = data[["merch.state"]].astype('category')
data[["merch.zip"]] = data[["merch.zip"]].astype('category')
data[["transtype"]] = data[["transtype"]].astype('category')
data[["amount"]] = data[["amount"]].astype('float')
data[["fraud"]] = data[["fraud"]].astype('category')

data=(data.sort_values(by="date")).reset_index(drop=True)


data_test = data.loc[data["date"]<'2010-02-01',]

# Build Variables
# Type I Variable: unusual transaction frequency at both card and merchant level
def Build_A_Var(df, time_window, key):
    
    ###########################
    # df: the name of the cleaned data frame 
    # time_window: 3 or 7 or 14 or 28
    # key: "card" or "merchant"
    ###########################
    
    df["record"]=1
    df[key+"_"+"frequency_"+str(time_window)] = 1
    
    for i in range(df.shape[0]):
        current_date = df.loc[df.index==i,"date"].values[0]
        
        if key == "card":
            current_key = df.loc[df.index==i,"cardnum"].values[0]
            subset = df.loc[(df["cardnum"]==current_key) & (df["date"]<=current_date) \
                             & (df["date"]>=current_date-pd.Timedelta(days=time_window)) \
                             ,["cardnum","record"]]
            run_frequency = subset["record"].sum()
            df.loc[df.index==i,key+"_"+"frequency_"+str(time_window)] = run_frequency
        
        elif key == "merchant":
            current_key = df.loc[df.index==i,"merchnum"].values[0]
            subset = df.loc[(df["merchnum"]==current_key) & (df["date"]<=current_date) \
                             & (df["date"]>=current_date-pd.Timedelta(days=time_window)) \
                             ,["merchnum","record"]]
            run_frequency = subset["record"].sum()
            df.loc[df.index==i,key+"_"+"frequency_"+str(time_window)] = run_frequency
    del df["record"]
    
    return df

############################################
start = time.time()                        #
data_test = Build_A_Var(data_test,3,"card")#
end = time.time()                          #
print((end - start)/60)                    #
data_test.to_csv("data_test.csv")          #
############################################

# Type II Variable: unusual transaction amounts at both card and merchant level
def Build_B_Var(df, time_window, key):
    
    ###########################
    # df: the name of the cleaned data frame 
    # time_window: 3 or 7 or 14 or 28
    # key: "card" or "merchant"
    ###########################
    
    df[key+"_"+"amount_to_avg_"+str(time_window)] = 1
    df[key+"_"+"amount_to_max_"+str(time_window)] = 0
    df[key+"_"+"amount_to_median_"+str(time_window)] = 1
    df[key+"_"+"amount_to_total_"+str(time_window)] = 0
    
    
    for i in range(df.shape[0]):
        # print(i)
        current_date = df.loc[df.index==i,"date"].values[0]
        current_amount = df.loc[df.index==i,"amount"].values[0]
        
        if key == "card":
            current_key = df.loc[df.index==i,"cardnum"].values[0]
            subset = df.loc[(df["cardnum"]==current_key) & (df["date"]<current_date) \
                             & (df["date"]>=current_date-pd.Timedelta(days=time_window))  \
                             ,["cardnum","amount"]]
            
        elif key == 'merchant':
            current_key = df.loc[df.index==i,"merchnum"].values[0]
            subset = df.loc[(df["merchnum"]==current_key) & (df["date"]<current_date) \
                             & (df["date"]>=current_date-pd.Timedelta(days=time_window)) \
                             ,["merchnum","amount"]]
        # print(subset.shape[0])
        
        if subset.shape[0] != 0:
            run_avg = subset["amount"].mean()
            run_max = subset["amount"].max()
            run_median = subset["amount"].median()
            run_total = subset["amount"].sum()
            
            df.loc[df.index==i,key+"_"+"amount_to_avg_"+str(time_window)] = current_amount/run_avg
            df.loc[df.index==i,key+"_"+"amount_to_max_"+str(time_window)] = current_amount/run_max
            df.loc[df.index==i,key+"_"+"amount_to_median_"+str(time_window)] = current_amount/run_median
            df.loc[df.index==i,key+"_"+"amount_to_total_"+str(time_window)] = current_amount/run_total
    
    return df

############################################
start = time.time()                        #
data_test = Build_B_Var(data_test,3,"card")#
end = time.time()                          #
print((end - start)/60)                    #
data_test.to_csv("data_test.csv")          #
############################################ 


# Type III Variable: unusual transaction location at both card and merchant level
def Build_C_Var(df, time_window, key):
    
    ###########################
    # df: the name of the cleaned data frame 
    # time_window: 3 or 7 or 14 or 28
    # key: "card" or "merchant"
    ###########################
    
    df[key+"_"+"distinct_state_"+str(time_window)] = 1
    df[key+"_"+"distinct_zip_"+str(time_window)] = 1
    
    for i in range(df.shape[0]):
        current_date = df.loc[df.index==i,"date"].values[0]
        
        if key == "card":
            current_key = df.loc[df.index==i,"cardnum"].values[0]
            subset = df.loc[(df["cardnum"]==current_key) & (df["date"]<=current_date) \
                             & (df["date"]>=current_date-pd.Timedelta(days=time_window)) \
                             ,["cardnum","merch.state","merch.zip"]]
            
            distinct_state = subset["merch.state"].nunique()
            distinct_zip = subset["merch.zip"].nunique()
    
            df.loc[df.index==i,key+"_"+"distinct_state_"+str(time_window)] = distinct_state
            df.loc[df.index==i,key+"_"+"distinct_zip_"+str(time_window)] = distinct_zip
        
        elif key == "merchant":
            current_key = df.loc[df.index==i,"merchnum"].values[0]
            subset = df.loc[(df["merchnum"]==current_key) & (df["date"]<=current_date) \
                             & (df["date"]>=current_date-pd.Timedelta(days=time_window)) \
                             ,["merchnum","merch.state","merch.zip"]]
            
            distinct_state = subset["merch.state"].nunique()
            distinct_zip = subset["merch.zip"].nunique()
    
            df.loc[df.index==i,key+"_"+"distinct_state_"+str(time_window)] = distinct_state
            df.loc[df.index==i,key+"_"+"distinct_zip_"+str(time_window)] = distinct_zip
    
    return df


################################################
start = time.time()                            #
data_test = Build_C_Var(data_test,3,"merchant")#
end = time.time()                              #
print((end - start)/60)                        #
data_test.to_csv("data_test.csv")              #
################################################ 
 

# Type IV Variable: unusual transaction interactions between card and merchant
def Build_D_Var(df, time_window, key):
    
    ###########################
    # df: the name of the cleaned data frame 
    # time_window: 3 or 7 or 14 or 28
    # key: "card" or "merchant"
    ###########################
    
    for i in range(df.shape[0]):
        current_date = df.loc[df.index==i,"date"].values[0]
        
        if key == "card":
            current_key = df.loc[df.index==i,"cardnum"].values[0]
            
            subset = df.loc[(df["cardnum"]==current_key) & (df["date"]<=current_date) \
                             & (df["date"]>=current_date-pd.Timedelta(days=time_window)) \
                             ,["cardnum","merchnum"]]
            
            distinct_merchnum = subset["merchnum"].nunique()
    
            df.loc[df.index==i,key+"_"+"distinct_merchnum_"+str(time_window)] = distinct_merchnum
         
        elif key == "merchant":
            current_key = df.loc[df.index==i,"merchnum"].values[0]
            
            subset = df.loc[(df["merchnum"]==current_key) & (df["date"]<=current_date) \
                             & (df["date"]>=current_date-pd.Timedelta(days=time_window)) \
                             ,["merchnum","cardnum"]]
            
            distinct_cardnum = subset["cardnum"].nunique()
    
            df.loc[df.index==i,key+"_"+"distinct_cardnum_"+str(time_window)] = distinct_cardnum
         
    return df

################################################
start = time.time()                            #
data_test = Build_D_Var(data_test,28,"card")   #
end = time.time()                              #
print((end - start)/60)                        #
data_test.to_csv("data_test.csv")              #
################################################ 

start = time.time() 
for key in ["card","merchant"]:
    for time_window in [3,7,14,28]:
        print(key)
        print(time_window)
        data=Build_A_Var(data,time_window,key)
        data=Build_B_Var(data,time_window,key)
        data=Build_C_Var(data,time_window,key)
        data=Build_D_Var(data,time_window,key)
end = time.time()    
print((end - start)/60)     

data.to_excel("Credit Card Payment Fraud Features.xlsx",index=False)

      
        

