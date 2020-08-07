#!/usr/bin/env python
# coding: utf-8

# ### Week 2 Assessment
# -Author: Alexandre Kanouni
# -Last modified: May 28 2020
# 

# ***Import packages***
# 
# 

# In[1]:


import os
import pandas as pd
import numpy as np

#sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# ***Functions***
# 

# In[2]:


def fn_MAE(actuals, predictions):
#     return np.round(np.mean(np.abs(predictions - actuals)))
    return (np.mean(np.abs(predictions - actuals)))

def fn_MSE(actuals, predictions):
#     return np.round(np.mean(np.square(predictions - actuals)))
    return (np.mean(np.square(predictions - actuals)))

def getThreshold(actuals, predictions):
    threshold = 0
    return threshold

def fn_fixPredictions(threshold, predictions):
    i=0
    predictions_temp = predictions.copy()
    while i<len(predictions_temp):
        if predictions_temp[i]>threshold:
            #then predict 1
            predictions_temp[i]=1
        else:
            predictions_temp[i]=0
        i+=1
    return predictions_temp

def fn_MAE_TRIAL(threshold, actuals, predictions):
#     return np.round(np.mean(np.abs(predictions - actuals)))
    predictions1 = fn_fixPredictions(threshold,predictions)
    return (np.mean(np.abs(predictions1 - actuals)))

def fn_getTrainAccuracy(th,actuals,predictions):
    temp = fn_fixPredictions(th,predictions)
    accuracy = np.mean(temp==y_train)
    return accuracy


# **Directories and paths**

# In[3]:


# Set directories
print(os.getcwd())
dirRawData = "../input/"
dirPData   = "../PData/"


# *** Read the data ***

# In[4]:


df_train = pd.read_csv(dirRawData+'train.csv')
# df_test = pd.read_csv(dirRawData+'test.csv')


# In[5]:


# df_train.head(10) #inspect data
# df_train.shape
# srs_missing_train = df_train.isnull().sum(axis=0) 
# # srs_missing_test = df_test.isnull().sum(axis=0) 
# print('train: \n',srs_missing_train[srs_missing_train>0]) #show which features have missing values
# # print('test: \n',srs_missing_test[srs_missing_test>0]) #show which features have missing values


###NOTE THAT WILL HAVE TO DO SOMETHING ABOUT:
# AGE IN BOTH
#CABIN IN BOTH
# Emarked in train
# Cabin in test


# ***Data manipulation***

# In[ ]:





# In[6]:


# df_all = df_train.append(df_test, sort=False)



### TESTING FUNCTION ONLY PASES DF_TRAIN SO DONT USE DF_TEST:
df_all = df_train

vars_all = df_all.columns.values
# df_train.shape
# df_test.shape
df_all.shape #double check number of rows matches what expected
var_dep = ['Survived']

vars_notToUse = ['PassengerId', 'Name']
#create list of independent variables
vars_ind = [var for var in vars_all if var not in (vars_notToUse+var_dep)] #use list comprehension (see below examples)
#create list of numeric independent variables
vars_ind_numeric = [var for var in vars_ind if df_all[var].dtype != 'object']
vars_ind_categorical = [var for var in vars_ind if df_all[var].dtype == 'object']
# print(vars_ind_categorical)


# In[7]:


# print(var_dep)
# print(vars_ind_numeric)
# print(vars_ind_categorical)


# ***Create one hot variables for any categorical variables***
# - though not if they are high cardinality - over 30 diff cats. Such vars should be excluded from further analysis. Use pandas as in lect video

# In[8]:


###check cardinality
print(df_all[vars_ind_categorical].nunique())
# len(df_all['Name'].unique())
vars_ind_cat_exclude = [var for var in vars_ind_categorical if len(df_all[var].unique())>30]
##vars_ind and remove those to be excluded
print(vars_ind_cat_exclude)
vars_ind_cat_include = [var for var in vars_ind_categorical if var not in vars_ind_cat_exclude]
print(vars_ind_cat_include)
df_all[vars_ind_cat_include].nunique()


# In[9]:


# df_all.head(10)


# In[10]:


###DROP VARIABLES
# vars_toDrop = vars_ind_cat_exclude+var_dep
vars_toDrop = vars_ind_cat_exclude ### FOR NOW
###FOR DEBUGGING, check indiv. data types of vars to drop
# print(df_all['lot_frontage'].dtype)
# print(df_all['garage_yr_blt'].dtype)
# print(df_all['mas_vnr_area'].dtype)
###ALTERNATIVELY, check all at once
# [df_all[var].dtype for var in vars_toDrop]
df_all.drop(labels=vars_toDrop,
            axis=1,
            inplace=True)
df_all.head(10)


# In[11]:


##update vars_ind and remove those to be excluded
# vars_ind_cat_include = [var for var in vars_ind_categorical if var not in vars_ind_cat_exclude]

vars_ind_onehot = []

df_all_onehot = df_all.copy()

# for col in vars_ind_categorical:
for col in vars_ind_cat_include:

    print(col)
    
    # use pd.get_dummies on  df_all[col] 
    df_oh = pd.get_dummies(df_all[col], drop_first=False)
    
    # Find the column name of the most frequent category
    col_mostFreq =  df_oh.sum(axis=0).idxmax() 
    
    # Drop the column of the most frequent category
    df_oh = df_oh.drop(col_mostFreq, axis=1, inplace=False)
        
    # Rename the columns to have the original variable name as a prefix
    oh_names = col+'_'+df_oh.columns
    df_oh.columns = oh_names
    
    df_all_onehot = pd.concat([df_all_onehot, df_oh], axis = 1, sort = False)

    del df_all_onehot[col]
    vars_ind_onehot.extend(oh_names)
df_all_onehot.head(10)
# print(vars_ind_onehot)


###UPDATE vars_ind
vars_ind = vars_ind_numeric + vars_ind_onehot 


# *** Separate train, test data ***

# In[12]:


# idx_train  = np.where(df_all_onehot['PassengerId'].isin(np.arange(0,6)))[0]
idx_train  = np.where(df_all_onehot['PassengerId'].isin(np.arange(0,892)))[0]
# idx_test   = np.where(df_all_onehot['PassengerId'].isin(np.arange(892,1310)))[0]


# ***Dealing with missing data***

# In[13]:


# print(len(idx_train))
# print(len(idx_test))
# print(idx_test)
# df_all_onehot.tail(10)


# In[14]:


print(df_all_onehot.shape)
#collapse axis = 0 i.e. sum missing values,
#store as series
# df_all.isnull()
srs_missing = df_all_onehot.isnull().sum(axis=0) 
print(srs_missing[srs_missing>0]) #show which features have missing values


# In[15]:


# df_original_all = df_all.copy()


# In[16]:


###CREATE an indicator for the missing values of age:
#0 when age is not missing, 1 when age is missing

# print(df_all['Age'].isnull())
age_missing = 1*df_all_onehot['Age'].isnull()
# type(age_missing) #double check type
# age_missing


# In[17]:


temp_nanmean = np.nanmean(df_all['Age'])


# In[ ]:





# In[18]:


###REPLACE missing values of age with the mean of the non missing values


# In[19]:


i=0
temp=df_all_onehot['Age'].copy()
while i<len(df_all):
    if np.isnan(df_all_onehot['Age'].iloc[i]):
        #then replace with mean of non-missing values
        temp.iloc[i]=temp_nanmean
    i+=1
df_all_onehot['Age'] = temp
df_all_onehot.head(10) #just to double check NaNs have been changed


# In[20]:


# ###ONLY NEEDED WHEN TEST DATA INCLUDED
# ###ALSO NEED TO DROP RECORD WITH MISSING FARE
# df_all_onehot.dropna(subset=['Fare'],inplace=True)


# In[21]:


### ENSURE NO MORE MISSING DATA

print(df_all_onehot.shape)
#collapse axis = 0 i.e. sum missing values,
#store as series
# df_all.isnull()
srs_missing = df_all_onehot.isnull().sum(axis=0) 
print(srs_missing[srs_missing>0]) #show which features have missing values


# In[22]:


# # df_all.head(1)
# # temp.head(10)
# not_missing = df_all['Age'][age_missing==0]
# sum = not_missing.sum(axis=0)
# # print(len(age_missing[age_missing==0]))
# # df_all['Age'][age_missing==0]
# temp_mean = sum/len(not_missing)
# print(temp_mean)


# In[23]:


# ?np.zeros()


# In[24]:


# df_all_onehot['Age'] = [temp_nanmean for var in age_missing if age_missing.loc[var]==1]


# ***Prepare data ***

# In[25]:


# # idx_train  = np.where(df_all_onehot['PassengerId'].isin(np.arange(0,6)))[0]
# train_range = (0,len(df_all_onehot['Survived'][~np.isnan(df_all_onehot['Survived'])])+1)
# # test_range = (train_range[1], len(df_all_onehot))
# idx_train  = np.where(df_all_onehot['PassengerId'].isin(np.arange(train_range[0],train_range[1])))[0]
# idx_test   = np.where(df_all_onehot['PassengerId'].isin(np.arange(test_range[0],test_range[1])))[0]
# df_all_onehot.tail(10)
# df_all_onehot[vars_ind].values[idx_train]


# In[26]:


# vars_all_onehot = df_all_onehot.columns.values
# vars_ind_onehot = [var for var in vars_all_onehot if var not in (vars_notToUse+var_dep)] #use list comprehension (see below examples)
# X = df_all_onehot[vars_ind_onehot].values
X = df_all_onehot[vars_ind].values
# df_all_onehot.head(10)
y = df_all_onehot[var_dep].values
# df_all_onehot[vars_ind].tail(100)


# In[27]:


X_train = X[idx_train]
# X_test = X[idx_test]
y_train = y[idx_train]
# print(len(y))
# print(len(y_train))


# ***Fit a linear regression model to the data***
# - use only: from sklearn.linear_model import LinearRegression

# In[28]:


#instatiate
lm_ = LinearRegression()


# In[29]:


#fit to TRAIN data
lm_.fit(X_train,y_train)


# ***Predictions***
# - The predictions from your linear regression will not be limited to 0 or 1 - they can be any real number. You will need to decide when to predict 1 and when to predict 0. Do this by choosing some threshold, th (rounded to 3 d.p.) If the prediction of your linear regression model is greater than (do not use greater or equals) th then predict 1 otherwise predict 0. For the threshold, choose the value of th (rounded to 3 d.p.) which maximises Accuracy on the train data. Note carefully your threshold and Accuracy on the train data - you will need to type this exactly into your function.

# In[30]:


# type(X_test)
# df_all_onehot['Survived']
# X_train.shape
# X_test.shape
# np.any(np.isnan(X_test)) #debugging


# In[31]:


lm__pred_train = lm_.predict(X_train)
# lm__pred_test = lm_.predict(X_test)
# lm__pred_train #check predictions
# lm__pred_test

# # print(lm__pred_train)
# ###JUST TO TEST METHODS, define basic threshold
# # th = 2.2370679652352625
# train_predictions = fn_fixPredictions(opt_th,lm__pred_train)
# # print(train_predictions)
# print(np.mean(np.abs(lm__pred_train-y_train)))
# train_MSE = fn_MSE(y_train, train_predictions)
# train_MAE = fn_MAE(y_train, train_predictions)
# # th = fn_getThreshold(df_train)
# print(train_MSE,train_MAE)
# # print(train_predictions)
# # print(y_train)


# ##### *** Optimise threshold *** 
# - Need to <b>maximise</b> Accuracy on train data. That is, must find threshold which minimises train error:
# <ol>
#     <li>We choose a measure of train error: MAE</li>
#     <li> Use <code>scipy.optimise</code> module to minimise MAE accross different thresholds</li>
# </ol>

# ***Optimise threshold using scipy, by minimisng MAE***

# In[32]:


###OPTIMISE THRESHOLD
from scipy.optimize import minimize
th=0
res = minimize(fn_MAE_TRIAL, [th,], args=(y_train, lm__pred_train), tol=1e-3, method="Powell")
print('Minimum {} attained at {}'.format(-res.fun, res.x))
opt_th = res.x


# ***ALTERNATIVELY, optimise threshold manually:***
# - Using a loop, naively find optimal threshold according to highest possible train accuracy

# In[33]:


#set starting threshold, and starting error
step=1e-3 #lowest possible value as need threshold rounded to 3 dp
opt_accuracy = 0 #lowest possible value
i=0
opt_th = -1 #initialise arbitrary non applicable number
#loop until have tried all thresholds, incrementing by 0.001. I.e. 1000 iterations
while i<1000:
    temp_th = step*i #create temporary threshold for each iteration
    #if current temp threshold leads to HIGHER train accuracy than current minimum, 
    #then this is the optimal threshold SO FAR
    
    temp_accuracy = fn_getTrainAccuracy(temp_th,y_train,lm__pred_train)
    if temp_accuracy > opt_accuracy: #using > instead of >= bc I want lowest possible threshold, but with highest possible accuracy 
        #this threshold leads to higher accuracy, so it is a better threshold
        opt_th = temp_th
        #temp_accuracy is now the accuracy to beat, so update variable
        opt_accuracy = temp_accuracy
    
    i+=1 #increment i
opt_th


# In[34]:


print('Train Accuracy:',fn_getTrainAccuracy(opt_th, y_train, lm__pred_train))


# In[35]:


# ###DEBUGGING
# # print(train_predictions)
# # print(lm__pred_train)
# temp = fn_fixPredictions(0.616,lm__pred_train)
# # temp = fn_fixPredictions(0.496,lm__pred_train)
# # print(lm__pred_train)
# # print(temp)
# # mean_predicted = np.mean(lm__pred_train)
# mean_predicted = np.mean(temp)
# mean_actual = np.mean(y_train)
# print('mean predicted survival rate:', mean_predicted)
# print('mean actual survival rate:', mean_actual)
# #Print accuracy 
# print('Train Accuracy:',np.mean(temp==y_train))


# In[ ]:





# In[36]:


### NOTE that even if the optimal threshold found by the methods differs, the train accuracy
### should be the same.
fn_getTrainAccuracy(opt_th, y_train, lm__pred_train)==fn_getTrainAccuracy(res.x, y_train, lm__pred_train)


# In[37]:


### In fact, there is a set of optimal thresholds around 0.61


# ***FINAL predictions, threshold (3dp) and Train accuracy***

# In[38]:


f_th = np.round(res.x,3)
f_train_predictions = fn_fixPredictions(th,lm__pred_train)
f_train_accuracy = np.round(fn_getTrainAccuracy(f_th,y_train,lm__pred_train),3)

print(f_th, f_train_accuracy)


# In[39]:


##LIST OF ITEMS TO RETURN
lst_toReturn = [lm_, X, y_train, 0.612, 0.817]


# In[40]:


type(lst_toReturn)


# In[ ]:





# In[ ]:




