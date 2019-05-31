# -*- coding: utf-8 -*-
"""
Credit Card Fraud Payment Analysis

"""

import pandas as pd

data = pd.read_excel("Credit Card Payment Fraud.xlsx")

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

test=data.dropna()

# First Round
# merch.zip
merchzip=data.loc[(data["merchnum"].notnull())&(data["merch.description"].notnull())&(data["merch.state"].notnull())&(data["merch.zip"].notnull()),
                  ["merchnum","merch.description","merch.state","merch.zip"]].drop_duplicates()
merchzip_miss=data.loc[data["merch.zip"].isnull(),]
merchzip_nomiss=data.loc[data["merch.zip"].notnull(),]

merchzip_miss_fix=pd.merge(merchzip,merchzip_miss,how='right',
               left_on=["merch.description","merch.state","merchnum"],
               right_on=["merch.description","merch.state","merchnum"])

merchzip_miss_fix=merchzip_miss_fix.iloc[:,[0,1,2,3,4,5,6,8,9,10]]
merchzip_miss_fix.rename(columns={'merch.zip_x':'merch.zip'}, inplace=True)

data=pd.concat([merchzip_nomiss,merchzip_miss_fix])

# merchnum
merchnum=data.loc[(data["merchnum"].notnull())&(data["merch.description"].notnull())&(data["merch.state"].notnull())&(data["merch.zip"].notnull()),
                  ["merchnum","merch.description","merch.state","merch.zip"]].drop_duplicates()
merchnum_miss=data.loc[data["merchnum"].isnull(),]
merchum_nomiss=data.loc[data["merchnum"].notnull(),]

merchnum_miss_fix=pd.merge(merchnum,merchnum_miss,how='right',
               left_on=["merch.description","merch.state","merch.zip"],
               right_on=["merch.description","merch.state","merch.zip"])

merchnum_miss_fix=merchnum_miss_fix.iloc[:,[0,1,2,3,4,5,6,7,9,10]]
merchnum_miss_fix.rename(columns={'merchnum_x':'merchnum'}, inplace=True)

data=pd.concat([merchum_nomiss,merchnum_miss_fix])

# merchstate
merchstate=data.loc[(data["merchnum"].notnull())&(data["merch.description"].notnull())&(data["merch.state"].notnull())&(data["merch.zip"].notnull()),
                  ["merchnum","merch.description","merch.state","merch.zip"]].drop_duplicates()
merchstate_miss=data.loc[data["merch.state"].isnull(),]
merchstate_nomiss=data.loc[data["merch.state"].notnull(),]

merchstate_miss_fix=pd.merge(merchstate,merchstate_miss,how='right',
               left_on=["merch.description","merchnum","merch.zip"],
               right_on=["merch.description","merchnum","merch.zip"])

merchstate_miss_fix=merchstate_miss_fix.iloc[:,[0,1,2,3,4,5,6,7,9,10]]
merchstate_miss_fix.rename(columns={'merch.state_x':'merch.state'}, inplace=True)

data=pd.concat([merchum_nomiss,merchnum_miss_fix])

test_1=data.dropna()


# Second Round
# merchzip
merchzip_miss_x=data.loc[data["merch.zip"].isnull(),]
merchzip_nomiss_x=data.loc[data["merch.zip"].notnull(),]
merchdescrip=merchzip_miss_x[["merch.description"]].drop_duplicates()
merchdescrip["merch.zip"]=range(1,497)
merchdescrip["merch.zip"]='M '+merchdescrip["merch.zip"].astype('str')

merchzip_miss_fix_x=pd.merge(merchzip_miss_x,merchdescrip, how='left',
                             left_on="merch.description",
                             right_on="merch.description")

merchzip_miss_fix_x=merchzip_miss_fix_x.iloc[:,[0,1,2,3,4,5,7,8,9,10]]
merchzip_miss_fix_x.rename(columns={'merch.zip_y':'merch.zip'}, inplace=True)


data = pd.concat([merchzip_nomiss_x,merchzip_miss_fix_x])

# merchnum
merchnum_miss_x=data.loc[data["merchnum"].isnull(),]
merchnum_nomiss_x=data.loc[data["merchnum"].notnull(),]
merchdescrip=merchnum_miss_x[["merch.description"]].drop_duplicates()
merchdescrip["merchnum"]=range(1,578)
merchdescrip["merchnum"]='M '+merchdescrip["merchnum"].astype('str')

merchnum_miss_fix_x=pd.merge(merchnum_miss_x,merchdescrip, how='left',
                             left_on="merch.description",
                             right_on="merch.description")

merchnum_miss_fix_x=merchnum_miss_fix_x.iloc[:,[0,1,2,3,4,5,6,8,9,10]]
merchnum_miss_fix_x.rename(columns={'merchnum_y':'merchnum'}, inplace=True)


data = pd.concat([merchnum_nomiss_x,merchnum_miss_fix_x])


# merchstate
merchstate_miss_x=data.loc[data["merch.state"].isnull(),]
merchstate_nomiss_x=data.loc[data["merch.state"].notnull(),]
merchdescrip=merchstate_miss_x[["merch.description"]].drop_duplicates()
merchdescrip["merch.state"]=range(1,152)
merchdescrip["merch.state"]='M '+merchdescrip["merch.state"].astype('str')

merchstate_miss_fix_x=pd.merge(merchstate_miss_x,merchdescrip, how='left',
                             left_on="merch.description",
                             right_on="merch.description")

merchstate_miss_fix_x=merchstate_miss_fix_x.iloc[:,[0,1,2,3,4,6,7,8,9,10]]
merchstate_miss_fix_x.rename(columns={'merch.state_y':'merch.state'}, inplace=True)

data = pd.concat([merchstate_nomiss_x,merchstate_miss_fix_x])

test_2=data.dropna()


data.to_excel("Credit Card Payment Fraud Clean.xlsx",index=False)



