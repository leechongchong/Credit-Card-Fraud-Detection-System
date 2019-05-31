# -*- coding: utf-8 -*-
"""
Credit Card Fraud Payment Analysis

"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("Credit Card Payment Fraud.xlsx")

# Overall EDA
print(data.columns)
print(data.head(10))
print(data.count)

## recordnum
print(data[["recordnum"]].dtypes)
data[["recordnum"]] = data[["recordnum"]].astype('category')
print(data[["recordnum"]].dtypes)
print(data["recordnum"].unique())

## cardnum
print(data[["cardnum"]].dtypes)
data[["cardnum"]] = data[["cardnum"]].astype('category')
print(data[["cardnum"]].dtypes)

print(data[["cardnum"]].isnull().sum())

cardnum = pd.DataFrame(data.groupby("cardnum")["recordnum"].count())
print(cardnum["recordnum"].describe())

cardnum = cardnum.sort_values(by="recordnum", ascending=False)
cardnum_viz=cardnum.head(20).reset_index()
cardnum_viz.plot(kind='bar',x='cardnum', y="recordnum")
plt.show()

## date *
print(data[["date"]].dtypes)
print(data[["date"]].isnull().sum())

date = pd.DataFrame(data.groupby("date")["recordnum"].count())
date = date.sort_values(by="recordnum", ascending=False).reset_index()
print(date[["date"]].min())
print(date[["date"]].max())

date["month"]=date["date"].dt.month
date["weekday"]=date["date"].dt.weekday_name

month=pd.DataFrame(date.groupby("month")["recordnum"].sum()).reset_index()
weekday=pd.DataFrame(date.groupby("weekday")["recordnum"].sum()).reset_index()

month.plot(kind='bar',x='month', y="recordnum")
plt.show()
weekday.plot(kind='bar',x='weekday', y="recordnum")
plt.show()

## merchnum *
print(data[["merchnum"]].dtypes)
data[["merchnum"]] = data[["merchnum"]].astype('category')
print(data[["merchnum"]].dtypes)

print(data[["merchnum"]].isnull().sum())

merchnum = pd.DataFrame(data.groupby("merchnum")["recordnum"].count())
print(merchnum['recordnum'].describe())

merchnum = merchnum.sort_values(by="recordnum", ascending=False)
merchnum_viz=merchnum.head(20).reset_index()
merchnum_viz.plot(kind='bar',x='merchnum', y="recordnum")
plt.show()

## merch.description
## merch.state
## merch.zip
## transtype 
print(data[["transtype"]].dtypes)
data[["transtype"]] = data[["transtype"]].astype('category')
print(data[["transtype"]].dtypes)
print(data["transtype"].unique())

## amount *
print(data[["amount"]].dtypes)
print(data[["amount"]].isnull().sum())

amount = data[["amount"]]
amount_uni=amount["amount"].unique()
print(amount["amount"].describe())
amount = pd.DataFrame(amount.sort_values(by="amount"))
amount["percentile"]=amount["amount"].rank(pct=True)

amount_viz=amount.loc[amount["percentile"]<=0.95,]
plt.hist(amount_viz["amount"],bins=50)

## fraud *
print(data[["fraud"]].dtypes)
data[["fraud"]] = data[["fraud"]].astype('category')
print(data[["fraud"]].dtypes)
print(data[["fraud"]].isnull().sum())
fraud = pd.DataFrame(data.groupby("fraud")["recordnum"].count()).reset_index()
fraud["total"]=95007
fraud["percentage"]=100*fraud["recordnum"]/fraud["total"]



