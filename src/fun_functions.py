from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from yellowbrick.regressor import PredictionError, ResidualsPlot
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.datasets import load_diabetes
from numpy import mean
from numpy import std
from numpy import absolute
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from yellowbrick.features import JointPlotVisualizer
sns.set_theme()


'''
IMPORT INSTRUCTIONS

--import fun_functions as fun--

call as fun.(*fun*ction name here)

Have fun!

'''


def logistic_model(X, y):# Lasso with 5 fold cross-validation
    '''
    to get both variables returned from this function, call it this way:

    model, lasso_best = fun.lasso_model(X.reshape(-1, 1), y)

    you can then use this model output in the mse_on_fold_lasso_plot 
    below for a fun time.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=40)
    model = LogisticRegressionCV(cv=10, random_state=40, max_iter=10000)
    # Fit model
    model.fit(X_train, y_train)

    model.alphas_

    lasso_best = Lasso(alpha=model.alphas_)
    lasso_best.fit(X_train, y_train)
    print('Best coeficient - X value:', list(zip(lasso_best.coef_, X)))
    print('R squared training set', round(lasso_best.score(X_train, y_train)*100, 2))
    print('R squared test set', round(lasso_best.score(X_test, y_test)*100, 2))
    print('MSE:', mean_squared_error(y_test, lasso_best.predict(X_test)))
    return model, lasso_best


def lasso_model(X, y):# Lasso with 5 fold cross-validation
    '''
    to get both variables returned from this function, call it this way:

    model, lasso_best = fun.lasso_model(X.reshape(-1, 1), y)

    you can then use this model output in the mse_on_fold_lasso_plot 
    below for a fun time.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=40)
    model = LassoCV(cv=10, random_state=40, max_iter=10000)
    # Fit model
    model.fit(X_train, y_train)

    model.alpha_

    lasso_best = Lasso(alpha=model.alpha_)
    lasso_best.fit(X_train, y_train)
    print('Best coeficient - X value:', list(zip(lasso_best.coef_, X)))
    print('R squared training set', round(lasso_best.score(X_train, y_train)*100, 2))
    print('R squared test set', round(lasso_best.score(X_test, y_test)*100, 2))
    print('MSE:', mean_squared_error(y_test, lasso_best.predict(X_test)))
    return model, lasso_best




def ridge_model(X, y):# Ridge with 5 fold cross-validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=40)
    model1 = RidgeCV(cv=10)

    # Fit model
    model1.fit(X_train, y_train)

    model1.alpha_

    ridge_best = Ridge(alpha=model1.alpha_)
    ridge_best.fit(X_train, y_train)
    print('Best coeficient - X value:', list(zip(ridge_best.coef_, X)))

    print('R squared training set', round(ridge_best.score(X_train, y_train)*100, 2))
    print('R squared test set', round(ridge_best.score(X_test, y_test)*100, 2))
    print(mean_squared_error(y_test, ridge_best.predict(X_test)))
    return model1, ridge_best




def mse_on_fold_lasso_plot(ymin, ymax, model=LassoCV()):
    '''
    
    this only works with a fitted LassoCV model or the model output from the
    lasso_model function above
    
    '''
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.semilogx(model.alphas_, model.mse_path_, ":")
    
    ax1.plot(
        model.alphas_ ,

        model.mse_path_.mean(axis=-1),
        "k",
        label="Average across the folds",
        linewidth=2,
    )
    ax1.axvline(
        model.alpha_, linestyle="--", color="k", label=f"Optimal alpha: CV estimate ({model.alpha_:.4f})"
    )

    ax1.legend()
    ax1.set_xlabel("alphas")
    ax1.set_ylabel("Mean square error")
    ax1.set_title("Mean square error on each fold:\nBanking Features")
    ax1.axis("tight")

    ymin, ymax = 0, 20
    ax1.set_ylim(ymin, ymax);





def coeficients_plot(model, X, y):
    alphas = np.linspace(0.01,500,100)
    coefs = []
    for a in alphas:
        model.set_params(alpha=a)
        model.fit(X, y)
        coefs.append(model.coef_)
    
    ax = plt.gca()
    
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.axis('tight')
    plt.xlabel('alpha')
    plt.ylabel('Standardized Coefficients')
    plt.title('Lasso coefficients as a function of alpha')




def residual_plot_w_qq(model, X, y):
    '''
    if you are doing a plot with a single feature you have to cast 
    it as an array first like this:
    
    X_funded= np.asarray(X['amount_funded_by_investors'])

    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    visualizer = ResidualsPlot(model, hist=False, qqplot=True)
    visualizer.fit(X_train.reshape(-1, 1), y_train)  
    visualizer.score(X_test.reshape(-1, 1), y_test) 
    visualizer.poof()




def prediction_error_plot(model, X, y):
    '''
    if you are doing a plot with a single feature you have to cast 
    it as an array first like this:
    
    X_funded= np.asarray(X['amount_funded_by_investors'])
    
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    visualizer = PredictionError(model)
    visualizer.fit(X_train.reshape(-1, 1), y_train)  
    visualizer.score(X_test.reshape(-1, 1), y_test)  
    visualizer.poof()



def three_way_scatter_plot(data, y_col, x_col, hue_col, plot_title, legend_title, xlabel, ylabel):
    '''
    call like so:

    three_way_scatter_plot(df1
                       ,'interest_rate'
                       , 'amount_requested'
                       , 'loan_length'
                       , 'Interest Rate and Amount Requested\nBy Loan Length'
                       , 'Loan Length'
                       , 'Amount Requested'
                       , 'Interest Rate')
    
    '''
    
    sns.set_theme()
    sns.scatterplot(data, y=y_col, x=x_col, hue=hue_col)
    plt.suptitle(plot_title, fontstyle='italic')
    plt.legend(title= legend_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel);


def occupants(df, col, index):
    onboard_split = []
    for x in df[col]:
        onboard_split.append(x.split('/'))

    occupants = []

    for list in onboard_split:
        occupants.append(list[index])

    list_occupants = []

    for list in occupants:
        list_occupants.append(list.split(':'))

    num_occupants = []

    for item in list_occupants:
        if item[1].isspace() == True:
            num_occupants.append(0)
        else:
            num_occupants.append(int(item[1]))


    return num_occupants


def find_replace(dataset, column,  find, replace):

    result = []
    for x in dataset[column]:
        if x == find:
            result.append(replace)
        else:
            result.append(x)

    return result



def replace_result(df, col):
    new_features = []
    for line in df[col]:
        if ', Result' in line:
            new_line = line.replace(', Result ', '')

            new_features.append('Result of' + new_line)
        else:
            new_features.append(line)
    return new_features


if __name__ == '__main__':
    pass