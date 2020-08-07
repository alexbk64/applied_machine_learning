def fn_titanic(df_args):
        #!/usr/bin/env python
    # coding: utf-8

    #######Week 2 Assessment######
    # -Author: Alexandre Kanouni #
    # -Last modified: May 28 2020#
    ##############################
    
    ###############################################################################################
    # This function takes as input a pandas Dataframe object made up of training from the Titanic # 
    # dataset, and returns a list containing five items, in this order:                           #
    # - A trained linear model object: lm_                                                        #
    # - The X matrix used to make predictions. Note that X and X_train are the same as code was   #
    #   originally written in order to also accomodate test data.                                 #
    # - The y array to which predictions are compared                                             #
    # - HARDCODED threshold rounded to 3dp                                                        #
    # - HARDCODED train accuracy, calculated on rounded threshold and also rounded to 3dp         #
    ###############################################################################################
    

    ########### ---------------------------------------- Import packages ------------------------------------------- ############
    # 
    # 

    import os
    import pandas as pd
    import numpy as np

    #sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    
    #scipy
    from scipy.optimize import minimize


    ########### ----------------------------------- Self-contained functions -------------------------------------- ############
    # 
    #

    # returns mean absolute error (not rounded)
    def fn_MAE(actuals, predictions):
    #     return np.round(np.mean(np.abs(predictions - actuals)))
        return (np.mean(np.abs(predictions - actuals)))

    # returns mean standard error (not rounded)
    def fn_MSE(actuals, predictions):
    #     return np.round(np.mean(np.square(predictions - actuals)))
        return (np.mean(np.square(predictions - actuals)))
    
    # determines whether to predict 1 or 0 according to threshold passed as argument
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

    # combines fn_MAE and fn_fixPredictioons to do both at once
    # used later on to optimise threshold by minimising MAE
    def fn_MAE_THRESH(threshold, actuals, predictions):
    #     return np.round(np.mean(np.abs(predictions - actuals)))
        predictions1 = fn_fixPredictions(threshold,predictions)
        return (np.mean(np.abs(predictions1 - actuals)))

    # simple function which returns train accuracy of predictions,
    # where predictions is the original predictions array returned by LinearRegression().predict
    # and temp is the fixed predictions according to the threshold th (see fn_fixPredictions)
    def fn_getTrainAccuracy(th,actuals,predictions):
        temp = fn_fixPredictions(th,predictions)
        accuracy = np.mean(temp==y_train)
        return accuracy


    def fn_optimizeThreshold_scipy(actuals, predictions):
        ###SCIPY: OPTIMISE THRESHOLD
        th=0
        res = minimize(fn_MAE_THRESH, [th,], args=(actuals, predictions), tol=1e-3, method="Powell")
        opt_th = res.x
        return opt_th


    def fn_optimizeThreshold_naive(actuals, predictions):
        ### NAIVE
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

            temp_accuracy = fn_getTrainAccuracy(temp_th,actuals,predictions)
            #using > instead of >= bc I want lowest possible threshold, but with highest possible accuracy 
            if temp_accuracy > opt_accuracy: 
                #this threshold leads to higher accuracy, so it is a better threshold
                opt_th = temp_th
                #temp_accuracy is now the accuracy to beat, so update variable
                opt_accuracy = temp_accuracy

            i+=1 #increment i
        return opt_th


    ### declare local variable df_all (note that here df_all is df_train passed as argument to main function)
    df_all = df_args

    
    

    ########### ------------------------------ Declare and initialise variables ------------------------------------ ############
    #
    #
    
    vars_all = df_all.columns.values
    var_dep = ['Survived']

    vars_notToUse = ['PassengerId', 'Name']
    #create list of independent variables
    vars_ind = [var for var in vars_all if var not in (vars_notToUse+var_dep)] #use list comprehension (see below examples)
    #create list of numeric independent variables
    vars_ind_numeric = [var for var in vars_ind if df_all[var].dtype != 'object']
    vars_ind_categorical = [var for var in vars_ind if df_all[var].dtype == 'object']



    
    ########### ----------------------- Create one hot variables for any categorical variables --------------------- ############
    # - though not if they are high cardinality - over 30 diff cats. Such vars should be excluded from further analysis. 
    #   Use pandas as in lect video



    ###check cardinality
    
    #exclude those with high card
    vars_ind_cat_exclude = [var for var in vars_ind_categorical if len(df_all[var].unique())>30] 
    #include all those not excluded
    vars_ind_cat_include = [var for var in vars_ind_categorical if var not in vars_ind_cat_exclude] 



    ###DROP VARIABLES
    vars_toDrop = vars_ind_cat_exclude #drop variables with high cardinality as instructed
    df_all.drop(labels=vars_toDrop,
                axis=1,
                inplace=True)
    
    
    
    

    ########### -------------------------------------- Onehot using Pandas ---------------------------------------- ############
    #
    #

    vars_ind_onehot = []

    df_all_onehot = df_all.copy()

    for col in vars_ind_cat_include:


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


    ###UPDATE vars_ind
    vars_ind = vars_ind_numeric + vars_ind_onehot 



    

    ########### --------------------------------- Dealing with missing data ---------------------------------------- ############
    # NOTE: Can uncomment the below block to see missing data before dealing with it 



    ###CREATE an indicator for the missing values of age:
    #0 when age is not missing, 1 when age is missing

    age_missing = 1*df_all_onehot['Age'].isnull()
    
    #get mean of non-missing values
    temp_nanmean = np.nanmean(df_all['Age'])



    ###REPLACE missing values of age with the mean of the non missing values



    i=0
    temp=df_all_onehot['Age'].copy()
    while i<len(df_all):
        if np.isnan(df_all_onehot['Age'].iloc[i]):
            #then replace with mean of non-missing values
            temp.iloc[i]=temp_nanmean
        i+=1
    df_all_onehot['Age'] = temp


    ### MANUALLY COMPUTE MEAN OF NON-MISSING VALS, unecessary bc of built-in function but left for reference
    
    # # df_all.head(1)
    # # temp.head(10)
    # not_missing = df_all['Age'][age_missing==0]
    # sum = not_missing.sum(axis=0)
    # # print(len(age_missing[age_missing==0]))
    # # df_all['Age'][age_missing==0]
    # temp_mean = sum/len(not_missing)
    # print(temp_mean)




    ########### ---------------------------------------  Prepare data --------------------------------------------- ############
    #
    #




    ## Below code dynamically identifies whether data belongs to train or data set. Needed as indexes have changed.
    # Unnecessary here as we are only dealing with training data. 
    # NOTE: not relevant for this assignment, but code was originally written to accomodate train and test data
    
    
    train_range = (0,len(df_all_onehot['Survived'][~np.isnan(df_all_onehot['Survived'])])+1)
    idx_train  = np.where(df_all_onehot['PassengerId'].isin(np.arange(train_range[0],train_range[1])))[0]


    X = df_all_onehot[vars_ind].values
    y = df_all_onehot[var_dep].values



    X_train = X[idx_train] #Here X_train = X, as X is made up only of training data
    y_train = y[idx_train]


    ########### ----------------------------- Fit a linear regression model to the data --------------------------- ############
    # - use only: from sklearn.linear_model import LinearRegression
    #


    #instatiate
    lm_ = LinearRegression()

    #fit to TRAIN data
    lm_.fit(X_train,y_train)


    ########### ----------------------------------------- Predictions -------------------------------------------- ############
    # - The predictions from your linear regression will not be limited to 0 or 1 - they can be any real number. You will need to decide when to predict 1 and when to predict 0. Do this by choosing some threshold, th (rounded to 3 d.p.) If the prediction of your linear regression model is greater than (do not use greater or equals) th then predict 1 otherwise predict 0. For the threshold, choose the value of th (rounded to 3 d.p.) which maximises Accuracy on the train data. Note carefully your threshold and Accuracy on the train data - you will need to type this exactly into your function.



    # PREDICT
    lm__pred_train = lm_.predict(X_train)



    ########### ------------------------------------- Optimise threshold ------------------------------------------ ############
    # - Need to <b>maximise</b> Accuracy on train data. That is, must find threshold which minimises train error:
    # <ol>
    #     <li>We choose a measure of train error: MAE</li>
    #     <li> Use <code>scipy.optimise</code> module to minimise MAE accross different thresholds</li>
    # </ol>

    
    ### Optimise threshold using scipy, by minimisng MAE:
    
    # opt_th = fn_optimizeThreshold_scipy(y_train, lm__pred_train)


    # ALTERNATIVELY, optimise threshold manually:
    # Using a loop, naively find optimal threshold according to highest possible train accuracy

    #MANUAL
    opt_th = fn_optimizeThreshold_naive(y_train, lm__pred_train)
    
    


    ########### ----------------------------------- FINALLY objects to return ------------------------------------ ############
    #
    #

    f_th = np.round(opt_th,3)
    f_train_predictions = fn_fixPredictions(f_th,lm__pred_train)
    f_train_accuracy = np.round(fn_getTrainAccuracy(f_th,y_train,lm__pred_train),3)

    # PRINT RESULTS IN ORDER TO HARDCODE (SEE NEXT BLOCK)
    # print(f_th, f_train_accuracy)



    ##LIST OF ITEMS TO RETURN
    # NOTE: as both thresholds (whether using scipy or manual method) should yield the same train accuracy, in reality
    # either of the below lines can be used. That is, th can be hardcoded as 0.612 or 0.609 and yield the same accuracy.
    # Note however that predictions will not necessarily be the same.
    
    # lst_toReturn = [lm_, X_train, y_train, 0.612, 0.817] #if using scipy optmise method
    lst_toReturn = [lm_, X_train, y_train, 0.609, 0.817] #if using manual optimise method
    


    return lst_toReturn