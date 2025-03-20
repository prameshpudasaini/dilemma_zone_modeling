import os
import numpy as np
import pandas as pd

import statsmodels.api as sm
from sklearn.model_selection import KFold

os.chdir(r"/Users/prameshpudasaini/Library/CloudStorage/OneDrive-UniversityofArizona/GitHub/dilemma_zone_modeling")

# =============================================================================
# functions
# =============================================================================

# function to process speed, time, and interaction features
def process_predictors(xdf):
    # speed variables
    xdf['speed_fps'] = round(xdf.speed*5280/3600, 1)
    xdf['speed_sq'] = round(xdf.speed_fps**2, 1)
    xdf['PSL_fps'] = round(xdf.Speed_limit*5280/3600, 1)
    
    # speed limit as binary variable
    xdf['is_PSL35'] = (xdf.Speed_limit == 35).astype(int)
    
    # update localtime to datetime and add is_weekend, is_night variables
    xdf.localtime = pd.to_datetime(xdf.localtime)
    xdf['is_peak'] = ((xdf.localtime.dt.hour.between(7, 9, inclusive = 'both')) | (xdf.localtime.dt.hour.between(15, 17, inclusive = 'both'))).astype(int)
    xdf['is_night'] = (~xdf.localtime.dt.hour.between(5, 19, inclusive = 'both')).astype(int)
    xdf['is_weekend'] = (xdf.localtime.dt.dayofweek >= 5).astype(int)
    
    # interaction features
    xdf['int_speed_PSL35'] = xdf.speed_fps * xdf.is_PSL35
    xdf['int_speed_peak'] = xdf.speed_fps * xdf.is_peak
    xdf['int_speed_night'] = xdf.speed_fps * xdf.is_night
    xdf['int_speed_weekend'] = xdf.speed_fps * xdf.is_weekend
    
    return xdf

# function to identify min stopping distance (Xs) and max clearing distance (Xc)
def identify_Xs_Xc(data, num_bins, group):
    xdf = data.copy()
    
    # adaptive binning of data to create speed bins
    xdf['speed_bin'] = pd.qcut(xdf.speed_fps, q = num_bins, duplicates = 'drop')
    
    # minimum/maximum value corresponding to each speed bin gives Xs/Xc
    if group == 'FTS':
        Xo = xdf.groupby('speed_bin', observed = True)['Xi'].transform('min')
    elif group == 'YLR':
        Xo = xdf.groupby('speed_bin', observed = True)['Xi'].transform('max')
    
    # add Xs/Xc to dataset as integer variable
    xdf['Xo'] = (xdf.Xi == Xo).astype(int)
    
    return xdf
    
    
# =============================================================================
# read and prepare datasets
# =============================================================================

# read model datasets
FTS = pd.read_csv("ignore/Wejo/trips_analysis/trips_FTS.txt", sep = '\t')
YLR = pd.read_csv("ignore/Wejo/trips_analysis/trips_YLR.txt", sep = '\t')

# map siteIDs
FTS['NodeID'] = FTS.Node.map({216: 1, 217: 2, 517: 3, 618: 4})
YLR['NodeID'] = YLR.Node.map({216: 1, 217: 2, 517: 3, 618: 4})
FTS.SiteID = FTS.NodeID.astype(str) + FTS.Approach
YLR.SiteID = YLR.NodeID.astype(str) + YLR.Approach

# process predictors
FTS = process_predictors(FTS)
YLR = process_predictors(YLR)

x_FTS = ['speed_fps', 'speed_sq', 'Crossing_length', 'int_speed_peak', 'int_speed_night', 'int_speed_weekend']
x_YLR = ['speed_fps', 'PSL_fps', 'Crossing_length', 'int_speed_peak', 'int_speed_night', 'int_speed_weekend']

# process site data to yield X and y for quantile regression
def process_site_data(xdf, group):
    bin_size = 25 if group == 'FTS' else 35
    predictors = x_FTS if group == 'FTS' else x_YLR
    
    list_site_df = []
    # loop through each site and identify Xs, Xc
    for site in list(xdf.SiteID.unique()):
        site_df = xdf.copy()[xdf.SiteID == site]
        num_bins = int(len(site_df) / bin_size)
        site_df = identify_Xs_Xc(site_df, num_bins, group)
        list_site_df.append(site_df)
    
    sdf = pd.concat(list_site_df, ignore_index = True) # combine data from all sites
    df_X = sdf.copy()[sdf.Xo == 1] # filter Xs or Xc observations
    
    df_X['constant'] = 1 # constant term for regression
    
    X = df_X[predictors]
    y = df_X['Xi']
    
    return {'X': X, 'y': y}

df_FTS = process_site_data(FTS, 'FTS')
df_YLR = process_site_data(YLR, 'YLR')


# =============================================================================
# quantile regression with cross-validation
# =============================================================================

# cross-validation parameters
k_folds = 5
kf = KFold(n_splits = k_folds, shuffle = True, random_state = 42)

# define pinball loss function
def pinball_loss(y_true, y_pred, tau):
    loss = np.where(y_true >= y_pred, tau * (y_true - y_pred), (1 - tau) * (y_pred - y_true))
    return np.mean(loss)

def quantile_regression_CV(group):
    # specify datasets and quantiles by group
    if group == 'FTS':
        X, y = df_FTS['X'], df_FTS['y']
        quantiles = [0.15, 0.5]
    elif group == 'YLR':
        X, y = df_YLR['X'], df_YLR['y']
        quantiles = [0.5, 0.85]
        
    # dictionaries to store pinball loss results
    results = {tau: [] for tau in quantiles}

    # perform k-fold cross-validation for quantile regression
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        for tau in quantiles:
            # fit quantile regression model
            model = sm.QuantReg(y_train, X_train).fit(q = tau)
            y_pred = model.predict(X_test)
            
            # compute pinball loss for test fold
            loss = pinball_loss(y_test, y_pred, tau)
            results[tau].append(loss)
            
    # compute average pinball loss across_folds
    avg_pinball_loss = {tau: np.mean(results[tau]) for tau in quantiles}
    
    print(f"Average pinball loss for {group}: {avg_pinball_loss}")
    
quantile_regression_CV('FTS')
quantile_regression_CV('YLR')
