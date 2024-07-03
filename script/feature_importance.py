import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

os.chdir(r"D:\GitHub\dilemma_Wejo")

# =============================================================================
# load datasets
# =============================================================================

# read FTS, YLR datasets
FTS = pd.read_csv("ignore/Wejo/trips_stop_go/FTS_filtered.txt", sep = '\t') # FTS trips
YLR = pd.read_csv("ignore/Wejo/trips_stop_go/YLR_filtered.txt", sep = '\t') # YLR trips

# read node geometry data and drop redundant columns
ndf = pd.read_csv("ignore/node_geometry.csv")
ndf.drop(['Name', 'Latitude', 'Longitude', 'num_LT_lanes', 'has_shared_LT', 'num_RT_lanes', 'dist_stop_cross'], axis = 1, inplace = True)

# merge node geometry data
FTS = pd.merge(FTS, ndf, on = ['Node', 'Approach'], how = 'left')
YLR = pd.merge(YLR, ndf, on = ['Node', 'Approach'], how = 'left')


# =============================================================================
# generate features
# =============================================================================

# function to classify hours into 4 categories
def classify_hour(xdf):
    xdf['Hour'] = np.select([(xdf.Hour >= 6) & (xdf.Hour < 10), 
                             (xdf.Hour >= 10) & (xdf.Hour < 15),
                             (xdf.Hour >= 15) & (xdf.Hour < 19),
                             (xdf.Hour >= 19) | (xdf.Hour < 6)],
                            ['morning', 'midday', 'evening', 'night'])
    return xdf

# function to generate features
def generate_features(xdf):
    # update localtime to datetime and add hour variable
    xdf.localtime = pd.to_datetime(xdf.localtime)
    xdf['Hour'] = xdf.localtime.dt.hour
    
    # add weekday as binary variable
    xdf['is_weekday'] = xdf.localtime.dt.dayofweek.apply(lambda x: 1 if x < 5 else 0) # Mon = 0, Sun = 6
    
    # # classify hour and perform one-hot encoding
    # xdf = classify_hour(xdf)
    # hour_dummies = pd.get_dummies(xdf.Hour, prefix = 'is_', prefix_sep = '_', drop_first = True, dtype = int)
    # xdf = pd.concat([xdf, hour_dummies], axis = 1)
    
    # add night time as binary variable
    xdf['is_night'] = xdf.localtime.dt.hour.apply(lambda x: 0 if 5 <= x <= 19 else 1)
    
    return xdf

FTS = generate_features(FTS)
YLR = generate_features(YLR)

# drop node 540 from analysis
FTS = FTS[FTS.Node != 540]

# drop redundant features
FTS.drop(['Node', 'Approach', 'ID', 'TripID', 'localtime', 'stop_time', 'stop_dist', 'Hour'], axis = 1, inplace = True)

# Note: 1599 FTS observations with 8 features for predicting Xi

# split data into train/test sets
X = FTS.drop('Xi', axis = 1)
y = FTS['Xi']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# =============================================================================
# feature importance analysis using XGBoost
# =============================================================================

# define parameter grid for GridSearchCV
param_grid = {
    'max_depth': [3, 4, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'min_child_weight': [3, 5, 7]
}

# initialize XGBoost regressor
xgb_model = xgb.XGBRegressor(objective = 'reg:squarederror', random_state = 42)

# initialize GridSearchCV
grid_search = GridSearchCV(
    estimator = xgb_model, 
    param_grid = param_grid, 
    cv = 5, 
    scoring = 'neg_mean_squared_error'
)

# perform grid search to find the best parameter
grid_search.fit(X_train, y_train)

# extract best model from GridSearchCV
best_xgb_model = grid_search.best_estimator_

# model training and prediction
best_xgb_model.fit(X_train, y_train) # train model
y_pred = best_xgb_model.predict(X_test) # predict on test set

# calculate RMSE on test set
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {round(rmse, 4)}")

# plot feature importance with the best model
xgb.plot_importance(best_xgb_model, max_num_features = 8)
plt.show()

# SHAP analysis
explainer = shap.TreeExplainer(best_xgb_model) # create SHAP explainer
shap_values = explainer.shap_values(X_train) # compute SHAP values
shap.summary_plot(shap_values, X_train, plot_type = 'bar') # summarize impact of features


# =============================================================================
# feature importance analysis using Lasso regression
# =============================================================================

# function to perform Lasso Regression
def lasso_regression(xdf):
    # define range of alpha values
    alphas = np.logspace(-4, 0, 50)
    
    # lists to store results
    list_rmse, list_coef = [], []
    
    # loop through each alpha and perform Lasso
    for alpha in alphas:
        lasso = Lasso(alpha = alpha) # initialize Lasso model
        lasso.fit(X_train, y_train) # fit model to training data
        y_pred = lasso.predict(X_test) # make predictions on test set
        rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2) # calculate RMSE
        
        list_rmse.append(rmse)
        list_coef.append(lasso.coef_)

    # convert list of coefficients to dataframe for plotting
    cdf = pd.DataFrame(list_coef, columns = list(X.columns))
    
    # plot coefficients vs alpha to visualize regularization
    plt.figure(figsize = (12, 6))
    for col in cdf.columns:
        plt.plot(alphas, cdf[col], label = col)
        
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Coefficient value')
    plt.title('Coefficient paths for Lasso regression')
    plt.legend(loc = 'lower right')
    plt.grid(True)
    
    # add alpha and corresponding RMSE values to coefficients dataset    
    cdf['Alpha'] = alphas
    cdf['RMSE'] = list_rmse
    
    return cdf

ldf = lasso_regression(FTS)
