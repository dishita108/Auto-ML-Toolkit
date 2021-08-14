import pandas as pd
import sys
#import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pathlib import Path

#from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV


def auto_ml(df, cv, iters, corr,task):
    
    inputs = df.iloc[:, :-1]
    target = df.iloc[:,-1]
    
    if task == 'regression':    
        model_params = {
            'LinearRegression':{
                'model': LinearRegression(),
                'params':{
                    'fit_intercept': [False, True],
                    'normalize': [False, True],
                    'copy_X': [False, True]
                }
            },
            'Lasso':{
                'model': Lasso(),
                'params':{
                    'alpha': [1,2,3,4,5,10],
                    'tol': [1e-2,1e-4,1e-6,1e-8,1e-9,1e-10],
                    'max_iter': [100,500,1000],
                    'selection':['cyclic', 'random']
                }
            },
            'DecisionTreeRegressor': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse', 'friedman_mse', 'mae'],
                'splitter': ['best', 'random'],
                'max_features': ['auto', 'sqrt', 'log2']
                }
            },
            'RandomForestRegressor': {
                'model': RandomForestRegressor(),
                'params': {
                    'n_estimators': [10,20,50,100,150,200],
                    'criterion': ['mse', 'mae'],
                    'max_features': ['auto', 'sqrt', 'log2']
                }
            },
            'Ridge':{
                'model': Ridge(),
                'params':{
                    'alpha': [1,2,3,4,5,10],
                    'tol': [1e-2,1e-4,1e-6,1e-8,1e-9,1e-10],
                    'max_iter': [100,500,1000]
                }
            },
        }
    
    elif task == 'classification':
        model_params = {
            'LogisticRegression':{
                'model': LogisticRegression(),
                'params':{
                    'C' : [1,2,5,10],
                    'max_iter' : [30, 50, 100,800,1000],
                    'tol': [1e-3,1e-4,1e-6]
                }
            },
            'DecisionTreeClassifier':{
                'model': DecisionTreeClassifier(),
                'params': {
                    'criterion': ['gini', 'entropy'],
                    'max_features': ['auto', 'sqrt', 'log2']
                }
            },
            'RandomForestClassifier': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [10,20,50,100,150,200],
                'criterion': ['gini', 'entropy'],
                'max_features': ['auto', 'sqrt', 'log2'],
                'class_weight': ['balanced', 'balanced_subsample']
                }
            },
            'SVC':{
            'model': SVC(),
            'params': {
                'C' : [1,2,3,4,5,10],
                'kernel' : ['rbf','linear', 'poly', 'sigmoid'],
                'degree' : [1,2,3,4,5,10],
                'gamma': ['auto', 'scale'],
                'decision_function_shape': ['ovo','ovr'],
                'tol': [1e-3,1e-4,1e-5,1e-6]
                }
            },
            'GausssianNB': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-09, 1e-10, 1e-11, 1e-12]
                }
            },
            'MultinomialNB': {
                'model': MultinomialNB(),
                'params': {
                    'alpha': [1,2,3,4,5,10],
                    'fit_prior': ['false', 'true']
                }
            }
        }
    else:
        pass

    
    
    #RSCV Template
    #df = pd.read_csv('Breast_cancer_data.csv')
    #describing data
    #df.describe(include='all')
    #finding no. of NaN values in each feature
    temp1 = df.head(n=10)
    temp2 = df.describe(include='all')
    data_null =  df.isnull().sum()
    #Finding Total NaN values
    total_nan = data_null.sum()
    total = len(df)
    #Calculating %
    percent_nan = (total_nan/total)*100
    #print("Feature\t\tNo. of NaN values")
    #Replacing with median
    if(percent_nan> 15):
        df.fillna(df.median(), inplace=True)
    else:
        df.dropna()

    #print(len(df))
    
    scores_rscv =[]
    for model_name, mp in model_params.items():
        if task == 'regression':
            rscv_clf = RandomizedSearchCV(mp['model'], mp['params'], cv = cv,  n_iter = iters, return_train_score = False)
        else:
            rscv_clf = RandomizedSearchCV(mp['model'], mp['params'], cv = cv,  n_iter = iters,scoring='roc_auc', return_train_score = False)
        rscv_clf.fit(inputs, target.values.ravel())
        scores_rscv.append({
                    'ModelName': model_name,
                    'BestScore': rscv_clf.best_score_,
                    'Best Parameter': rscv_clf.best_params_,
                    'Best Estimator': rscv_clf.best_estimator_,
                    'Best Re-fit time':  rscv_clf.refit_time_
                })
    
    result_rscv = pd.DataFrame(scores_rscv, columns = ['ModelName', 'BestScore', 'Best Parameter', 'Best Estimator', 'Best Re-fit time'])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    #print(result_rscv) #Table of results...
    
    
    for i in range(0, result_rscv.shape[0]):
            if result_rscv.iloc[i,1] == result_rscv.iloc[:,1].max():
                model = result_rscv.iloc[i,3]
            else:
                pass
    #Plotting the performance of the models...
    for i in range(0, result_rscv.shape[0]):
        result_rscv.iloc[i,1] = round(result_rscv.iloc[i,1] * 100, 2)


    
    plt.figure(figsize = (20,10))
    performance1 = sns.barplot(x = 'ModelName', y = 'BestScore', data = result_rscv)
    for index, row in result_rscv.iterrows():
        performance1.text(row.name,row.BestScore, round(row.BestScore,2), ha="center", color='black')
    plt.savefig('static/images/graph.png', bbox_inches = 'tight', pad_inches = 0)
    #print("")
    
    print(result_rscv) #Table of results to be displayed..



    plt.figure(figsize = (20,20))
    ax = sns.heatmap(df.corr(), annot=True, linewidths=.5) #notation: "annot" and NOT "annote"
    bottom, top = ax.get_ylim()
    heat_map = ax.set_ylim(bottom + 0.5, top - 0.5)
    #print(f'heat map has been generated...\n{heat_map}')
    plt.savefig('static/images/heatmap.png', bbox_inches = 'tight', pad_inches = 0)
    #The best model is to be selected and then trained...
    
    
    print()
    print(f'The best model has been auto-selected for you with the configuration as {model}.')
    print('We are now training the model...')
    model = model.fit(inputs,target.values.ravel())
    
    #The accuracy score of the model is being analyzed...
    score = round(model.score(inputs, target.values.ravel()),4)
    #print(f'Model is trained and has an accuracy score of about {round(score,4) * 100} percent. Your model is fit to make predictions now..!!')
    print(f'Model is trained and has an accuracy score of about {round((score*100),2)} percent. Your model is fit to make predictions now..!!')
    #print(data_null)
    print("Total no. of NaN values : ",total_nan)
    print("Total % of NaN values : ",percent_nan)
    
    c = df.corr()
    keeps = c.index[abs(c.iloc[:,-1]) >= corr].tolist()
    df = df[keeps]
    #print('Data-set reduced to selected features based on selected correlation threshold...')
    
    #Thus, we now have the auto-selected features based on thresholds set by the user...
    #We can now define the input and output variable(s)...
    
    target = df.drop(df.iloc[:, :-1], axis = 1)
    inputs = df.iloc[:,:-1]
    #print('The input and output features have been defined...')
    ls = []
    for i in inputs.columns:
        ls.append(i)
    print(f'Selected input features: {ls}')
    lts = []
    for i in target.columns:
        lts.append(i)
    print(f'Output feature(s): {lts}')


    mypath = Path(".") / "static" / "model_pickle"
    with open(mypath, 'wb') as f:
        pickle.dump(model, f)
    return model #You can change this as per your wish...

#Your task would be to display the following on the UI...
#display table of results (result_rscv)
#display the best trained model (model) with best estimators
#a download button that allows the user to download the trained model (pkl file or other equivalent alternative)


cv =sys.argv[1]
iters = sys.argv[2]
corr = sys.argv[3]
df = sys.argv[4]
task = sys.argv[5]
auto_ml(pd.read_csv(df), int(cv), int(iters), float(corr),task)