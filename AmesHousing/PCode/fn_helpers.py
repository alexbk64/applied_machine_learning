#prints the model equation
def fn_print_coefficients(model):
    import numpy as np
    coef_skl_intercept = model.intercept_
    coef_skl_other = list(model.coef_.flatten())
    toPrint=""
    exp = 0
    for i in coef_skl_other:
        exp+=1
        toPrint += " + {:6.4f}x^{}".format(i,exp) if exp>1 else " + {:6.4f}x".format(i)
        
    str1 = str(coef_skl_intercept)
    print('Learned polynomial for degree {}:'.format(exp))
    print('{:6.4f}'.format(coef_skl_intercept) + toPrint)
    
    
#plots predictions
def fn_plot_predictions(data, model):
    
    import pandas as pd
    import numpy as np
    
    from matplotlib import pyplot as plt
#     %matplotlib inline 
    
    def plot_data(df1,df2):
        fig, axes = plt.subplots(2, 1, figsize=[16,24])
#         ticks=np.arange(-1.5, 3.0,0.5)
#         labels=ticks.astype(str)
        
        ## Plot data
#         axes[0].set_yticks(ticks, minor=False)
#         axes[0].set_yticklabels(labels, fontdict=None, minor=False)
        axes[0].set_title('Training data')
        axes[0].plot(df1['x_1'],df1['y'],'k.')
        axes[0].set_xlabel('x_1',fontsize=20)
        axes[0].set_ylabel('y',fontsize=20)
        axes[0].set_xlim([0.0,1.0])
        axes[1].set_ylim([-1.5,2.0])

        
#         axes[1].set_yticks(ticks, minor=False)
#         axes[1].set_yticklabels(labels, fontdict=None, minor=False)
        axes[1].set_title('Test data')
        axes[1].plot(df2['x_1'],df2['y'],'k.')
        axes[1].set_xlabel('x_1',fontsize=20)
        axes[1].set_ylabel('y',fontsize=20)
        axes[1].set_xlim([0.0,1.0])
        axes[1].set_ylim([-1.5,2.0])
        
        ## Plot predictions
        
        axes[0].plot(df1['x_1'],df1['y_hat'],'r.')
        axes[1].plot(df2['x_1'],df2['y_hat'],'r.')
        
        
     
    # Get degree of polynomial model was fitted on
#     degree = model.rank_ #can only use rank for LinearRegression, as Ride doesn't have attribute rank_
    #instead get shape of array of coefficients, its size indicates degree of polynomial
    degree = model.coef_.shape[0] 
    
    
    # Prepare data
    vars_x = ['x_1']
    if degree >=2:
        vars_x.extend(['x_%d'%i for i in range(2,degree+1)])

    X = data[vars_x].values
    y = data['y'].values
    
    idx_train = np.arange(0,30)
    idx_test = np.arange(30,len(data))
    X_train = X[idx_train]
    X_test = X[idx_test]
    
    df_train = data.iloc[idx_train]
    df_test = data.iloc[idx_test]
    
    
    ## Make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    ## Append predictions to train and test data
    #creates an extra column in each data set
    train_data = df_train.copy()
    test_data = df_test.copy()
    
    colname = 'y_hat'      #new var will be x_power
    train_data[colname] = train_preds
    test_data[colname] = test_preds


    
    plot_data(train_data, test_data)

    




