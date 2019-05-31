#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:40:49 2019

@author: lichong
"""
import numpy as np
import pandas as pd

from sklearn.preprocessing import scale
from sklearn.feature_selection import RFECV

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt



data = pd.read_excel("Credit Card Payment Fraud Features.xlsx")

print(data.columns)
print(data.head(10))

data = data.sort_values(by=['date'])

data_mdl = data[['card_frequency_3', 'card_amount_to_avg_3', 'card_amount_to_max_3',\
                 'card_amount_to_median_3', 'card_amount_to_total_3',\
                 'card_distinct_state_3', 'card_distinct_zip_3',\
                 'card_distinct_merchnum_3', 'card_frequency_7', 'card_amount_to_avg_7',\
                 'card_amount_to_max_7', 'card_amount_to_median_7',\
                 'card_amount_to_total_7', 'card_distinct_state_7',\
                 'card_distinct_zip_7', 'card_distinct_merchnum_7', 'card_frequency_14',\
                 'card_amount_to_avg_14', 'card_amount_to_max_14',\
                 'card_amount_to_median_14', 'card_amount_to_total_14',\
                 'card_distinct_state_14', 'card_distinct_zip_14',\
                 'card_distinct_merchnum_14', 'card_frequency_28',\
                 'card_amount_to_avg_28', 'card_amount_to_max_28',\
                 'card_amount_to_median_28', 'card_amount_to_total_28',\
                 'card_distinct_state_28', 'card_distinct_zip_28',\
                 'card_distinct_merchnum_28', 'merchant_frequency_3',\
                 'merchant_amount_to_avg_3', 'merchant_amount_to_max_3',\
                 'merchant_amount_to_median_3', 'merchant_amount_to_total_3',\
                 'merchant_distinct_state_3', 'merchant_distinct_zip_3',\
                 'merchant_distinct_cardnum_3', 'merchant_frequency_7',\
                 'merchant_amount_to_avg_7', 'merchant_amount_to_max_7',\
                 'merchant_amount_to_median_7', 'merchant_amount_to_total_7',\
                 'merchant_distinct_state_7', 'merchant_distinct_zip_7',\
                 'merchant_distinct_cardnum_7', 'merchant_frequency_14',\
                 'merchant_amount_to_avg_14', 'merchant_amount_to_max_14',\
                 'merchant_amount_to_median_14', 'merchant_amount_to_total_14',\
                 'merchant_distinct_state_14', 'merchant_distinct_zip_14',\
                 'merchant_distinct_cardnum_14', 'merchant_frequency_28',\
                 'merchant_amount_to_avg_28', 'merchant_amount_to_max_28',\
                 'merchant_amount_to_median_28', 'merchant_amount_to_total_28',\
                 'merchant_distinct_state_28', 'merchant_distinct_zip_28',\
                 'merchant_distinct_cardnum_28','fraud']]


labels = data_mdl[['fraud']]
features = data_mdl.loc[:, data_mdl.columns !='fraud']

# Normalization with z-scaling

features = pd.DataFrame(scale(features))
features.columns = ['card_frequency_3', 'card_amount_to_avg_3', 'card_amount_to_max_3',\
                 'card_amount_to_median_3', 'card_amount_to_total_3',\
                 'card_distinct_state_3', 'card_distinct_zip_3',\
                 'card_distinct_merchnum_3', 'card_frequency_7', 'card_amount_to_avg_7',\
                 'card_amount_to_max_7', 'card_amount_to_median_7',\
                 'card_amount_to_total_7', 'card_distinct_state_7',\
                 'card_distinct_zip_7', 'card_distinct_merchnum_7', 'card_frequency_14',\
                 'card_amount_to_avg_14', 'card_amount_to_max_14',\
                 'card_amount_to_median_14', 'card_amount_to_total_14',\
                 'card_distinct_state_14', 'card_distinct_zip_14',\
                 'card_distinct_merchnum_14', 'card_frequency_28',\
                 'card_amount_to_avg_28', 'card_amount_to_max_28',\
                 'card_amount_to_median_28', 'card_amount_to_total_28',\
                 'card_distinct_state_28', 'card_distinct_zip_28',\
                 'card_distinct_merchnum_28', 'merchant_frequency_3',\
                 'merchant_amount_to_avg_3', 'merchant_amount_to_max_3',\
                 'merchant_amount_to_median_3', 'merchant_amount_to_total_3',\
                 'merchant_distinct_state_3', 'merchant_distinct_zip_3',\
                 'merchant_distinct_cardnum_3', 'merchant_frequency_7',\
                 'merchant_amount_to_avg_7', 'merchant_amount_to_max_7',\
                 'merchant_amount_to_median_7', 'merchant_amount_to_total_7',\
                 'merchant_distinct_state_7', 'merchant_distinct_zip_7',\
                 'merchant_distinct_cardnum_7', 'merchant_frequency_14',\
                 'merchant_amount_to_avg_14', 'merchant_amount_to_max_14',\
                 'merchant_amount_to_median_14', 'merchant_amount_to_total_14',\
                 'merchant_distinct_state_14', 'merchant_distinct_zip_14',\
                 'merchant_distinct_cardnum_14', 'merchant_frequency_28',\
                 'merchant_amount_to_avg_28', 'merchant_amount_to_max_28',\
                 'merchant_amount_to_median_28', 'merchant_amount_to_total_28',\
                 'merchant_distinct_state_28', 'merchant_distinct_zip_28',\
                 'merchant_distinct_cardnum_28']


# Train - Test Split  (70% train, 30% test)
feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.3, random_state=10)

print(sum(label_train["fraud"]) / len(label_train))
print(sum(label_test["fraud"]) / len(label_test))


# Feature Selection
def feature_selection(model):
    
    clf_rfe = RFECV(model , scoring='roc_auc')
    clf_rfe = clf_rfe.fit(feature_train, label_train)
    
    features_num = clf_rfe.n_features_
    score = clf_rfe.grid_scores_[features_num-1] 
    
    features = np.array(feature_train.columns.to_list())
    features_final = features[clf_rfe.support_].tolist()

    print(features_num)
    print(score)
    
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("ROC AUC CV score")
    plt.plot(range(1, len(clf_rfe.grid_scores_) + 1), clf_rfe.grid_scores_)
    plt.show()
     
    return features_final


# ML Algorithms
    
def model_evaluation(final_features, model):
    ## Train
    feature_train_selected = feature_train.loc[:,final_features]
    clf = model
    clf.fit(feature_train_selected, label_train)
    #prediction_train = clf.predict(feature_train_selected)
    prediction_train = clf.predict_proba(feature_train_selected)[:,1] >= 0.3
    
    auc_train = roc_auc_score(label_train, prediction_train)
    matrix_train = confusion_matrix(label_train, prediction_train)
    accuracy_train = (matrix_train[1,1] + matrix_train[0,0])/len(label_train)
    recall_train = matrix_train[1,1]/(matrix_train[1,1] + matrix_train[1,0]) 
    fp_train = matrix_train[0,1]/(matrix_train[0,1] + matrix_train[0,0])

    
    
     ## Test
    feature_test_selected = feature_test.loc[:,final_features]
    #prediction_test = clf.predict(feature_test_selected)
    prediction_test = clf.predict_proba(feature_test_selected)[:,1] >= 0.3
    
    
    auc_test = roc_auc_score(label_test, prediction_test)
    matrix_test = confusion_matrix(label_test, prediction_test)
    accuracy_test = (matrix_test[1,1] + matrix_test[0,0])/len(label_test)
    recall_test = matrix_test[1,1]/(matrix_test[1,1] + matrix_test[1,0]) 
    fp_test = matrix_test[0,1]/(matrix_test[0,1] + matrix_test[0,0])
    
    
    model_evaluation = pd.DataFrame({"auc": [round(auc_train,4), round(auc_test,4)],
                                     "accuracy": [round(accuracy_train,4), round(accuracy_test,4)],
                                     "true_fdr": [round(recall_train,4), round(recall_test,4)],
                                     "false_fdr": [round(fp_train,4), round(fp_test,4)]},\
                                     index = ["Train","Test"])    
    return model_evaluation



def model_evaluation_org(model):
    ## Train
    clf = model
    clf.fit(feature_train, label_train)
    
    prediction_train = clf.predict(feature_train) 
    
    auc_train = roc_auc_score(label_train, prediction_train)
    matrix_train = confusion_matrix(label_train, prediction_train)
    accuracy_train = (matrix_train[1,1] + matrix_train[0,0])/len(label_train)
    recall_train = matrix_train[1,1]/(matrix_train[1,1] + matrix_train[1,0]) 
    fp_train = matrix_train[0,1]/(matrix_train[0,1] + matrix_train[0,0])

       
     ## Test
    prediction_test = clf.predict(feature_test)  
    
    
    auc_test = roc_auc_score(label_test, prediction_test)
    matrix_test = confusion_matrix(label_test, prediction_test)
    accuracy_test = (matrix_test[1,1] + matrix_test[0,0])/len(label_test)
    recall_test = matrix_test[1,1]/(matrix_test[1,1] + matrix_test[1,0]) 
    fp_test = matrix_test[0,1]/(matrix_test[0,1] + matrix_test[0,0])
    
    
    model_evaluation = pd.DataFrame({"auc": [round(auc_train,4), round(auc_test,4)],
                                     "accuracy": [round(accuracy_train,4), round(accuracy_test,4)],
                                     "true_fdr": [round(recall_train,4), round(recall_test,4)],
                                     "false_fdr": [round(fp_train,4), round(fp_test,4)]},\
                                     index = ["Train","Test"])    
    return model_evaluation


###################################### Without Feature Selection & Threshold Adjustment ###################################### 
## Naive Bayes
model = GaussianNB()                                                                                                                            
NB_evaluation = model_evaluation_org(model)
NB_evaluation["Model"] = 'Naive Bayes'
        

## Logistic Regression
model = LogisticRegression()                                                                                                                        
LR_evaluation = model_evaluation_org(model)
LR_evaluation["Model"] = 'Logistic Regression'  


## Decision Tree
model = DecisionTreeClassifier()                                                                                                                           
DT_evaluation = model_evaluation_org(model)
DT_evaluation["Model"] = 'Decision Tree'


## Random Forest
model = RandomForestClassifier()                                                                                                                         
RF_evaluation = model_evaluation_org(model)
RF_evaluation["Model"] = 'Random Forest'

## Boosting Tree
model = AdaBoostClassifier()                                                                                                                      
BT_evaluation = model_evaluation_org(model)
BT_evaluation["Model"] = 'Gradient Boosting Tree'


## XGboost
model = XGBClassifier()                                                                                                                         
XGB_evaluation = model_evaluation_org(model)
XGB_evaluation["Model"] = 'XGBoost'


## SVM
model = SVC()                                                                                                                      
SVC_evaluation = model_evaluation_org(model)
SVC_evaluation["Model"] = 'SVM'


## ANN
model = MLPClassifier()                                                                                                                        
NN_evaluation = model_evaluation_org(model)
NN_evaluation["Model"] = 'Neural Network'



evaluation_org = pd.concat([NB_evaluation, LR_evaluation, DT_evaluation, 
                                 RF_evaluation, BT_evaluation, XGB_evaluation,
                                 SVC_evaluation, NN_evaluation])

evaluation_org = model_evaluation.reset_index()

evaluation_test_org = evaluation_org.loc[evaluation_org.index == 'Test',:]



###################################### With Feature Selection & Threshold Adjustment ###################################### 

## Decision Tree *
clf_DT = DecisionTreeClassifier()
DT_features = feature_selection(clf_DT)
DT_evaluation_update = model_evaluation(DT_features, clf_DT)
DT_evaluation_update["Model"] = 'Decision Tree'


## Random Forest *
clf_RF = RandomForestClassifier()
RF_features = feature_selection(clf_RF)
RF_evaluation_update = model_evaluation(RF_features, clf_RF)
RF_evaluation_update["Model"] = 'Random Forest'



## Gradient Boosting Tree *
clf_BT = AdaBoostClassifier()
BT_features = feature_selection(clf_BT)
BT_evaluation_update = model_evaluation(BT_features, clf_BT)
BT_evaluation_update["Model"] = 'Gradient Boosting Tree'


## XGboost *
clf_XGB = XGBClassifier()
XGB_features = feature_selection(clf_XGB)
XGB_evaluation_update = model_evaluation(XGB_features, clf_XGB)
XGB_evaluation_update["Model"] = 'XGboost'


## SVM * --> too slow
clf_SVC = SVC()
SVC_features = feature_selection(clf_SVC)
SVC_evaluation_update = model_evaluation(SVC_features, clf_SVC)
SVC_evaluation_update["Model"] = 'SVM'


    
    




















