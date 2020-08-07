import pandas as pd                                                                                

import numpy as np                                                                                 

from n_190011697 import fn_titanic                                                                 

def fn_accuracy(actuals, predictions): 
    return np.mean(actuals == predictions) 
                                                                                                    

path_train = '../input/train.csv'                                                                  

df_train = pd.read_csv(path_train)                                                                 

# run function 
lm_, X_train, y_train, th, acc = fn_titanic(df_train)                                              

# test results 
lm__pred_train = lm_.predict(X_train) 
bln_pass = np.round(fn_accuracy(lm__pred_train > th, y_train), 3) == acc                           

bln_pass                                                                                           


acc                                                                                               


th                                                                                                

