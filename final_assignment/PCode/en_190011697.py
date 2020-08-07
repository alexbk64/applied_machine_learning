########## Final assignment: Part 1
## Note: The following modelling was done using the train dataset with 250k examples
## Author: Alexandre Kanouni
## Date: 15.07.2020

def fn_logistic(df_train, df_test):

    ############################# Import packages #############################
    import os
    import numpy as np
    import pandas as pd
    # import pickle
    import h2o
    from h2o.estimators.glm import H2OGeneralizedLinearEstimator
    from h2o.grid.grid_search import H2OGridSearch

    #category encoders
    from category_encoders import LeaveOneOutEncoder

    #needed for fn_computeRatiosOfNumerics()
    from itertools import permutations
    
    #stops the output of warnings when running models on test data which have different factor levels for categorical 
    #data than on the train data. I am aware this is not best practice, but makes the output more readable
    import warnings
    warnings.filterwarnings('ignore')



    ################################ Functions #############################

    def fn_MAE(actuals, predictions):
        return np.round(np.mean(np.abs(predictions - actuals)))

    def fn_RMSE(actuals, predictions):
        return np.round(np.sqrt(np.mean((predictions - actuals)**2)))

    def fn_tosplines(x):
        x = x.values
        # hack: remove zeros to avoid issues where lots of values are zero
        x_nonzero = x[x != 0]
        ptiles = np.percentile(x_nonzero, [10, 20, 40, 60, 80, 90])
        ptiles = np.unique(ptiles)
        print(var, ptiles)
        df_ptiles = pd.DataFrame({var: x})
        for idx, ptile in enumerate(ptiles):
            df_ptiles[var + '_' + str(idx)] = np.maximum(0, x - ptiles[idx])
        return(df_ptiles)

    def fn_computeRatiosOfNumerics(df, variables):
    ## Process:
    # 1. Gets passed most important numeric variables
    # 2. Computes all pairwise ratios between each of these i.e
    # - get all permutations of length 2, and divide term 1 by term 2
    # e. Returns a dataframe containing engineered variables, with appropriately named columns

        pairs = []
        lst_series = []
        for i in range(len(variables)+1):
            for subset in permutations(variables, i):
                if len(subset)==2: pairs.extend([subset])
        temp_colnames = []
        for elem in pairs:
            ## create column names
            temp_colname = 'ratio_{}.{}'.format(elem[0],elem[1])
            temp_colnames.append(temp_colname)
            #compute ratio
            try: 
                srs_pair_ratio = df[elem[0]]/df[elem[1]]
            except ZeroDivisionError:  
                #if denominator is 0, will catch error and assign nan value to that ratio
                srs_pair_ratio = np.nan
                srs_pair_ratio = np.nan
            srs_pair_ratio.rename(temp_colname, inplace=True)
            lst_series.append(srs_pair_ratio)
        #create dataframe with appropriate column names
        df_2 = pd.DataFrame(index = df.index, columns = temp_colnames)
        #fill dataframe with series
        for idx, col in enumerate(df_2):
            df_2[col] = lst_series[idx]


        # Seems df division already catches ZeroDivisonError and assigns infinity value when denom = 0 but not numerator 
        # In such case, want 0 coefficient.
        # Also want 0 coefficients when both numerator and denom are 0
        # therefore replace all inf and nan values with zeroes
        df_2.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
        return df_2

    def fn_createInteractions(df, factors):
        ## takes as input a pandas dataframe, and a LIST of column names on which to create interactions
        #create an h2o frame
        h2o_df_temp = h2o.H2OFrame(df[factors], destination_frame='df_interactions_temp')

        #use H2OFrame.interaction(factors, pairwise, max_factors, min_occurence, destination_frame=None)
        h2o_df_temp = h2o_df_temp.interaction(factors, pairwise=True, max_factors=100, min_occurrence=1)

        return h2o_df_temp.as_data_frame(use_pandas=True)




    ################################ DEFINE VARIABLES #############################

    vars_all = df_train.columns.values
    var_dep = ['target']

    vars_notToUse = ['unique_id'] 
    vars_ind = [var for var in vars_all if var not in (vars_notToUse + var_dep)]

    # find the categorical vars - this includes the hccv
    vars_ind_categorical = list(df_train.columns[df_train.dtypes == 'category'])
    # find numeric vars
    vars_ind_numeric = [var for var in vars_ind if var not in vars_ind_categorical]


    ## GET HCCV VARS
    ## If want to use some cardinality threshold other than 30, can edit threshold below:
    th_card = 30
    srs_card = df_train[vars_ind_categorical].nunique()
    vars_ind_hccv = srs_card[srs_card>th_card].index.values.tolist()  #stores names of categorical variables with cardinality higher than threshold


    # for convenience store dependent variable as y
    y = df_train[var_dep].values.ravel()


    ########################## Set index for train, val, design, test data #############################
    #### Create folds to seperate train data into train, val, design, test
    rng = np.random.RandomState(2020)
    fold = rng.randint(0, 10, df_train.shape[0])
    df_train['fold'] = fold

    #get indices for each subset
    idx_train  = df_train['fold'].isin(range(8))
    idx_val    = df_train['fold'].isin([7, 8])
    idx_design = df_train['fold'].isin(range(9))

    #drop fold column as no longer needed (and want to maintain similar structure to df_test
    df_train.drop(columns='fold', inplace=True)


    ############################## **Start and connect the H2O JVM** #############################
    # - Load the previous models in order to identify most important variables. To save time (and given that function can only take as input the train and test data), relevant code has been commented out but left in so that you may see my approach. I have instead hard-coded numeric and categorical variables I have found to be most important.

    # *Models are taking very long to run so have pre-loaded them below.*
    # - uncomment the below code to load the models but note that they must be in the PData directory 

    # ### Connect to H2O cluster
    h2o.init(port=54321)  # line left uncommented as I make use of H2O functions throughout the script

    # ### LOAD THE MODELS

    # # GLM basic, no interactions, no mean imputation for missing level values in test
    # # model name: GLM_model_basic
    # path_glm_basic = dirPData + 'GLM_model_basic'

    # # GLM basic, no interactions, WITH mean imputation for missing level values in test
    # # model name: GLM_model_basic_meanImpute
    # path_glm_basic_meanImpute = dirPData + 'GLM_model_basic_meanImpute'

    # # GLM numerical divisons, no interactions, WITH mean imputation for missing level values in test
    # # model name: GLM_model_numeric_meanImpute
    # path_glm_numeric_meanImpute = dirPData + 'GLM_model_numeric_meanImpute'

    # # GLM numerical divisons, with interactions, WITH mean imputation for missing level values in test
    # # model name: GLM_model_numeric_interactions_meanImpute

    # glm_basic = h2o.load_model(path = path_glm_basic) 
    # glm_basic_meanImpute = h2o.load_model(path = path_glm_basic_meanImpute)
    # glm_numeric_meanImpute = h2o.load_model(path = path_glm_numeric_meanImpute)


    ############################## DEAL WITH MISSINGS #############################

    #### IDENTIFY MISSINGS

    ## Check for missing numerics which have been replaced with -99 (placeholder, really it is missing)
    #get percentage of missing values for each feature
    srs_missing = pd.DataFrame(df_train.loc[:,:]==-99).sum(axis=0)/len(df_train)
    # print(srs_missing[srs_missing!=0])  #show which numerics have 'missing' placeholder values, and their percentage of missing values

    #get list of variables which have more than x% missing values
    #arbitrarily setting threshold to 50% but could tune this parameter if time permits
    missings_th = 0.5
    many_missings = [var for var in df_train.columns.values if srs_missing[var]>= missings_th ]  

    ## DO NOT USE VARIABLES WITH MORE THAN x% MISSINGS
    #add vars from many_missings to vars_notToUse, remove them from list of numeric variables
    vars_notToUse.extend(many_missings)
    #turn into set and set back into list - deals with issue of duplicates when running code multiple time
    vars_notToUse = list(set(vars_notToUse)) 

    #remove variables in many_missings from var_ind_numeric
    vars_ind_numeric = [var for var in vars_ind_numeric if var not in vars_notToUse]
    # print([var for var in vars_ind_numeric if var in vars_notToUse])  #double check they've been removed: printed list should be empty



    ### MEAN-IMPUTE MISSINGS

    # list of variables to impute
    vars_toImpute = [var for var in srs_missing[srs_missing>0].index.tolist() if var not in many_missings]

    #get subset dataframe (only cols which are in variables_toImpute)
    #get only values != -99 -> this will mean that the missings will be returned as NaN. Can then use fillna
    df_temp=df_train[vars_toImpute][df_train[vars_toImpute]!=-99].copy()  #make a working copy

    #use fillna: computing the mean of each column and filling NaNs with this mean value.
    df_temp.fillna(df_temp.mean(), inplace=True)

    df_train[vars_toImpute] = df_temp




    ############################## SPLINE HIGH CARDINALITY NUMERICS #############################
    ## Attempt at capturing non-linear relationships in model

    ### Spline numeric variables with cardinality higher than 8
    # define variables to spline
    vars_ind_tospline = df_train[vars_ind_numeric].columns[(df_train[vars_ind_numeric].nunique() > 8)].tolist()
    #Find the percentiles on train data only, then apply same percentiles to both train and test data, even if test data distribution is very different.
    #update df_train, df_test
    for var in vars_ind_tospline:
        df_ptiles = fn_tosplines(df_train[var])
        df_train.drop(columns=[var], inplace=True)
        df_test.drop(columns=[var], inplace=True)
        vars_ind_numeric.remove(var)
        df_train = pd.concat([df_train, df_ptiles], axis=1, sort=False)
        df_test = pd.concat([df_test, df_ptiles], axis=1, sort=False)
        vars_ind_numeric.extend(df_ptiles.columns.tolist())


    ############################## DEAL WITH HCCVs #############################
    # - note that any modifications made to train data must also be made to test data (engineered colums etc)

    ### HCCV ENCODING USING category_encoders

    enc = LeaveOneOutEncoder(cols=vars_ind_hccv, sigma=0.3)
    enc.fit(df_train[idx_design], y[idx_design])
    df_train = enc.transform(df_train)  #encode hccvs in train data
    # df_train[vars_ind_hccv].head()

    df_test['target'] = np.nan  #add NaN target column to test dataset in order for it to have same shape as df_train
    df_test = enc.transform(df_test)  #encode hccvs in test data
    df_test.drop(columns='target', inplace=True)  #drop target column from df_test 


    ############################## INTERACTIONS #############################
    # - same applies here, whatever interactions are in train data must also be in test data

    ### DEFINE FIVE MOST IMPORTANT CATEGORICAL VARS

    ### NOTE: The below interactions are created based on the largest
    ### coefficients in a previously-run model. The code below identifies
    ### those coefficients by loading the model and manipulating the data.
    ### However, as assignment requires only input to be train and test
    ### datasets, the most important variables have been hardcoded in.


    ### Inspect coefficients from basic model with no interactions
    ## Plot standardised coefficients
    # glm_basic.std_coef_plot(num_of_features=10)

    ## Get list of 5 most important variables via varimp()
    # note that glm_basic.varimp() contains some onehots created by H2o on the fly when building the model, and thus some aren't actually present in the train/test frames
    # therefore can't refer to them before running a model, and we need to refer to the original variables before h2o onehots them
    # we extract these by:
    # - getting only the name of the variable and not its values i.e. var[0] for var in glm_basic.varimp()
    # - splitting on onehot delimiter '.' and keeping only first part of result. This is name of original variable

    # # Get list of FIVE most important categorical variables
    # vars_mostImp_cat=[]
    # for var in glm_basic.varimp():
    #     orig_var = var[0].split('.')[0]
    #     if orig_var in vars_ind_categorical and orig_var not in vars_mostImp_cat:  #check if numeric
    #         #add to list of important categorical vars only if not already in list
    #         vars_mostImp_cat.append(orig_var)
    #     if len(vars_mostImp_cat)>= 5:
    #         break

    vars_mostImp_cat=['f09', 'f03', 'f07', 'f27', 'e11']  #comment this line if uncommenting the above block

    #Get dataframe of interactions all pairwise interactions between five most important categorical variables
    df_train_interactions = fn_createInteractions(df_train, vars_mostImp_cat)
    df_test_interactions = fn_createInteractions(df_test, vars_mostImp_cat)

    #append new columns to df_train and df_test
    df_train[df_train_interactions.columns.values] = df_train_interactions
    df_test[df_test_interactions.columns.values] = df_test_interactions

    # include new numeric variables in vars_ind_numeric
    vars_ind_categorical.extend(df_train_interactions.columns.tolist())





    ############################## OTHER FEATURES #############################
    # DIVISON OF NUMERICS
    # - must also add engineered columns to test data

    ### DEFINE THREE MOST IMPORTANT NUMERICAL VARS

    ### NOTE: The below interactions are created based on the largest
    ### coefficients in a previously-run model. The code below identifies
    ### those coefficients by loading the model and manipulating the data.
    ### However, as assignment requires only input to be train and test
    ### datasets, the most important variables have been hardcoded in.



    # # plot largest standardised coefficients
    # # glm_basic.std_coef_plot(num_of_features=10)
    # # Get list of THREE most important variables
    # vars_mostImp_numeric=[]
    # for var in glm_basic.varimp():
    #     orig_var = var[0].split('.')[0]
    #     if orig_var in vars_ind_numeric and orig_var not in vars_mostImp_numeric:  #check if numeric
    #         #add to list of important numeric vars
    #         vars_mostImp_numeric.append(orig_var)
    #     if len(vars_mostImp_numeric)>= 3:
    #         break

    vars_mostImp_numeric=['f11', 'f11_0', 'f11_1']  #comment this line if uncommenting the above block

    ### COMPUTE RATIO COLUMNS FOR BOTH DATASETS
    df_temp_train = fn_computeRatiosOfNumerics(df_train, vars_mostImp_numeric)
    df_temp_test = fn_computeRatiosOfNumerics(df_test, vars_mostImp_numeric)

    #append new columns to df_train and df_test
    df_train[df_temp_train.columns.values] = df_temp_train
    df_test[df_temp_test.columns.values] = df_temp_test

    # include new numeric variables in vars_ind_numeric
    vars_ind_numeric.extend(df_temp_train.columns.tolist())


    ############################## LOAD DATA INTO H2O JVM #############################

    ### START JVM
    # h2o.init(port=54321)  #commented as already connected to H2O cluster 
    # h2o.connect(port=54321)

    ### Remove all data previously loaded (if any) in JVM as no longer need it
    for key in h2o.ls()['key']:
        h2o.remove(key)


    #### Create H2OFrames in H2O cluster for df_train, df_test
    h2o_df_train = h2o.H2OFrame(df_train[vars_ind_numeric + vars_ind_categorical + var_dep],
                               destination_frame='df_train')
    h2o_df_test = h2o.H2OFrame(df_test[vars_ind_numeric + vars_ind_categorical],
                               destination_frame='df_test')


    ### Change target to enum type as we are building a classification model
    # h2o_df_train[var_dep].types
    h2o_df_train[var_dep] = h2o_df_train[var_dep].asfactor()
    # h2o_df_train[var_dep].types


    ############################## DEFINE THE FEATURES TO BE USED #############################

    features = vars_ind_numeric + vars_ind_categorical




    ###USE BOOLEAN MASKS TO INDEX TRAIN,VAL,DESIGN DATA
    idx_h2o_train  = h2o.H2OFrame(idx_train.astype('int').values)
    idx_h2o_val    = h2o.H2OFrame(idx_val.astype('int').values)
    idx_h2o_design = h2o.H2OFrame(idx_design.astype('int').values)


    ############################# MODELLING #############################

    ### H2O GRIDSEARCH - hyper-parameter tuning
    # ## Will use random grid search rather than cartesian to save some time

    ### NOTE: The below code is commented out as it takes approximately 1h
    ### to run. After running, the best model was selected according to AUC
    ### and its corresponding hyper-parameters were recorded. These are
    ### hard-coded later on in a single GLM estimation, in order to estimate
    ### only the best model and save on computational time/resources.


    # ## GLM hyper parameters

    # lambda_opts = [16. * 2.**-i for i in np.arange(15)]
    # alpha_opts = [0, 0.5, 0.99]
    # glm_params = {
    #     'alpha': alpha_opts,
    #     'lambda': lambda_opts
    # }
    # search_criteria = {
    #     'strategy': 'RandomDiscrete',
    #     'max_runtime_secs': 3600
    # }


    # ## Train and validate a random grid of GLMs
    # ##According to H2O documentation, must use logit link as we are estimating a binomial classification model. 
    # glm_grid = H2OGridSearch(
    #     model=H2OGeneralizedLinearEstimator(
    #         family='binomial',
    #         link='logit',
    #         nfolds=10,
    #         seed=2020,
    #         keep_cross_validation_models=False, 
    #         keep_cross_validation_predictions=False,
    #         keep_cross_validation_fold_assignment=False,
    #         missing_values_handling='mean_imputation'
    #     )
    #     , grid_id='glm_grid'
    #     , hyper_params=glm_params
    #     , search_criteria=search_criteria
    # #     , parallelism = 0 #adaptive parallelism, decided by H2O
    # )

    # glm_grid.train(x=features, 
    #                y='target',
    #                training_frame=h2o_df_train[idx_h2o_design, :],
    #                seed=2020)

    # ## Get the grid results, sorted by validation AUC
    # glm_grid_performance = glm_grid.get_grid(sort_by='auc', decreasing=True)
    # glm_grid_performance


    ############################### best model results ###########################
    # #     alpha         lambda          model_ids                 auc        # #
    # #0      [0.0]  [9.765625E-4]  glm_grid_model_38  0.8595786171889577      # #
    ##############################################################################



    ### ESTIMATE GLM via H2O, using hyper-params found through grid-search

    # We set family to bimonial as we are running a classification GLM model (with only two classes).
    # According to H2O documentation, must use logit link as we are estimating a binomial classification model. 
    # missing_values_handling -> MeanImputation: deals with new sample having categorical levels not seen in training. Replaces the unseen value with the most frequent level present in TRAINING SET.
    # keep_cross_valudation_* -> set to false to save some memory in H2o cluster.
    model=H2OGeneralizedLinearEstimator(  alpha=0.00
                                            , family='binomial'
                                            , link='logit'
                                            , lambda_= 9.765625E-4
                                            , nfolds=10
                                            , seed=2020
                                            , keep_cross_validation_models=False
                                            , keep_cross_validation_predictions=False
                                            , keep_cross_validation_fold_assignment=False
                                            , missing_values_handling='mean_imputation'
                                       )
    print('Estimating GLM model...')  #notification of progress when running function
    model.train(x=features, 
                y='target',
                training_frame=h2o_df_train[idx_h2o_design, :])

    ### NOTE: This model is run using hard-coded values of alpha and lambda.
    ### These are the ones corresponding to the best model found via grid
    ### search above. Computation (wall) time: 3min 3s





    ### Save the model
    dirPData = '../PData/'
    dirPOutput = '../POutput/'
    best_glm = model
    best_glm_path = h2o.save_model(model=best_glm, path=dirPData, force=True)
    print(best_glm_path)


    ## MAKE PREDICTIONS ON TEST DATASET
    temp_preds = best_glm.predict(h2o_df_test)


    ### Export predictions to kaggle-required format
    df_test['Predicted'] = np.round(temp_preds[2].as_data_frame(), 5)
    df_preds = df_test[['unique_id', 'Predicted']].copy()
    df_test[['unique_id', 'Predicted']].to_csv(dirPOutput + 'best_glm_250k.csv', index=False)

    #### KAGGLE AUCROC PUBLIC LEADERBOARD SCORE: 0.80162

    ### SHUT DOWN H2O CLUSTER
    # h2o.cluster().shutdown()  #not shutting down cluster as not sure if this will cause issues when returning the handle to the h2o object

    ############################### END OF FUNCTION, RETURN ###########################
    # - trained H2OGeneralizedLinearEstimator object
    # - Test data fed to object when making predictions: handle to H2OFrame object
    # - Kaggle public leaderboard score, hardcoded as 3 dp
    return [best_glm, h2o_df_test, 0.802]
