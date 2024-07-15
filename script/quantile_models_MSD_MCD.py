import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\dilemma_Wejo")

# specify parameters for quantile binning
num_bins, quantile_Xs, quantile_Xc = 40, 0.00, 1.00

# =============================================================================
# functions
# =============================================================================

# function to create bins based on adaptive binning of quantiles
def process_speed_bin(xdf, group):
    # assign variable name and quantile to compute based on FTS/YLR group
    if group == 'FTS':
        var_name, q = 'Xs', quantile_Xs
    else:
        var_name, q = 'Xc', quantile_Xc
    
    # process speed data
    xdf['speed'] = xdf.speed * 5280/3600 # mph to ft/s
    xdf['speed_sq'] = xdf.speed ** 2 # square of speed
    
    # compute mean and st dev; filter out observations beyond 2 st dev
    mean, std = xdf.speed.mean(), xdf.speed.std()
    xdf = xdf.copy()[xdf.speed.between(mean - 2*std, mean + 2*std, inclusive = 'both')]
    
    # adaptive binning of data by speed variable
    xdf['speed_bin'] = pd.qcut(xdf.speed, q = num_bins, duplicates = 'drop')
    
    # group by speed bin and compute quantile
    gdf = xdf.groupby('speed_bin')['Xi'].quantile(q).reset_index(name = var_name)

    # merge quantile based Xs/Xc data to original data
    xdf = pd.merge(xdf, gdf, on = 'speed_bin', how = 'left')
    
    return xdf

# function to process intersection geometry features
def geometry_features(xdf):
    # read node geometry data and select relevant columns
    ndf = pd.read_csv("ignore/node_geometry_v2.csv")
    ndf = ndf[['Node', 'Approach', 'Speed_limit', 'int_cross_length', 'num_TH_lanes', 'has_shared_RT', 'has_median']]

    # merge node geometry data
    xdf = pd.merge(xdf, ndf, on = ['Node', 'Approach'], how = 'left')
    
    # update localtime to datetime and add is_weekend, is_night variables
    xdf.localtime = pd.to_datetime(xdf.localtime)
    xdf['is_weekend'] = (xdf.localtime.dt.dayofweek >= 5).astype(int)
    xdf['is_night'] = (~xdf.localtime.dt.hour.between(5, 19, inclusive = 'both')).astype(int)
    
    # add duration of yellow indication
    xdf['yellow_time'] = xdf.Node.apply(lambda x: 3.5 if x == 540 else 4)
    
    # unit conversion for speed limit from mph to ft/s
    xdf.Speed_limit = round(xdf.Speed_limit * 5280/3600, 1)
    
    # generate features as interaction terms
    xdf['int_speed_RT'] = xdf.speed * xdf.has_shared_RT
    xdf['int_speed_median'] = xdf.speed * xdf.has_median
    xdf['int_speed_cross'] = xdf.speed * xdf.int_cross_length
    xdf['int_speed_night'] = xdf.speed * xdf.is_night
    xdf['int_speed_yellow'] = xdf.speed * xdf.yellow_time

    # add difference of speed from speed limit as predictor
    xdf['speed_diff'] = xdf.speed - xdf.Speed_limit

    return xdf

# function to model median Xs or Xc using quantile regression
def quantile_regression(X, q = 0.5):
    model = sm.QuantReg(y, X).fit(q = q)
    print(model.summary())
    return model


# =============================================================================
# load and prepare datasets
# =============================================================================

# read datasets with stop/go trips
GLR = pd.read_csv("ignore/Wejo/trips_stop_go/GLR_filtered.txt", sep = '\t') # GLR trips
FTS = pd.read_csv("ignore/Wejo/trips_stop_go/FTS_filtered.txt", sep = '\t') # FTS trips
YLR = pd.read_csv("ignore/Wejo/trips_stop_go/YLR_filtered.txt", sep = '\t') # YLR trips
RLR = pd.read_csv("ignore/Wejo/trips_stop_go/RLR_filtered.txt", sep = '\t') # RLR trips

# combine GLR with FTS data
FTS = pd.concat([FTS, GLR], ignore_index = True)

# filter out FTS trips stopping beyond stop line
# filter out YLR trips with yellow onset beyond stop line
FTS = FTS[FTS.stop_dist >= -10]
YLR = YLR[YLR.Xi >= -10]

# process speed bins for FTS/YLR datasets
fdf = process_speed_bin(FTS, 'FTS')
ydf = process_speed_bin(YLR, 'YLR')

# get features for modeling
fdf = geometry_features(fdf)
ydf = geometry_features(ydf)

# px.scatter(fdf, x = 'Xs', y = 'speed').update_traces(marker = dict(size = 7, symbol = 'circle')).show()
# px.scatter(ydf, x = 'Xc', y = 'speed').update_traces(marker = dict(size = 7, symbol = 'circle')).show()

# px.scatter(fdf, x = 'Xi', y = 'Xs').update_traces(marker = dict(size = 7, symbol = 'circle')).show()
# px.scatter(ydf, x = 'Xi', y = 'Xc').update_traces(marker = dict(size = 7, symbol = 'circle')).show()


# =============================================================================
# quantile regression models for Xs
# =============================================================================

# target and predictor variables
X = fdf.copy()
y = fdf['Xs']

# # testing different models for Xs
# quantile_regression(X[['speed']]) # 0.4064
# quantile_regression(X[['Speed_limit']]) # 0.04413
# quantile_regression(X[['speed', 'speed_diff']]) # 0.4417
# quantile_regression(X[['speed', 'speed_sq']]) # 0.4803 (**best model**)
# quantile_regression(X[['speed', 'speed_sq', 'speed_diff']]) # 0.4838 (slight improvement)
# quantile_regression(X[['speed', 'speed_sq', 'Speed_limit']]) # (strong multicollinearity)

# quantile_regression(X[['speed', 'speed_sq', 'int_speed_RT']]) # (not significant)
# quantile_regression(X[['speed', 'speed_sq', 'int_speed_median']]) # (not significant)
# quantile_regression(X[['speed', 'speed_sq', 'int_speed_cross']]) # (strong multicollinearity)
# quantile_regression(X[['speed', 'speed_sq', 'int_speed_night']]) # (not significant)
# quantile_regression(X[['speed', 'speed_sq', 'int_speed_yellow']]) # strong multicollinearity
# quantile_regression(X[['speed_sq', 'int_speed_yellow']]) # 0.4778

# final model for Xs
model_Xs = quantile_regression(X[['speed', 'speed_sq']])
df_Xs = model_Xs.summary2().tables[1]
df_Xs.to_csv("output/quantile_regression_model_Xs.csv")

# create dataset for fitting quantile regression line
y_pred = np.linspace(FTS.speed.min(), FTS.speed.max(), 100)
X_pred = pd.DataFrame({'speed': y_pred, 'speed_sq': y_pred**2})
predictions = model_Xs.predict(X_pred)

# plot FTS with computed Xs and fit quantile regression line
plt.scatter(FTS.Xi, FTS.speed, alpha = 0.5, label = 'Data')
plt.plot(predictions, y_pred, color = 'red', label = 'Median regression')
plt.xlabel('Minimum stopping distance (ft)')
plt.ylabel('Speed (ft/s)')
plt.legend()
plt.show()


# =============================================================================
# quantile regression models for Xc
# =============================================================================

# add intercept with value 1
ydf['intercept'] = 1

# target and predictor variables
X = ydf.copy()
y = ydf['Xc']

# # testing different models for Xc
# quantile_regression(X[['speed']]) # 0.4667
# quantile_regression(X[['Speed_limit']]) # -0.03915
# quantile_regression(X[['intercept', 'speed']]) # 0.5375 (**best model**)
# quantile_regression(X[['intercept', 'speed', 'speed_diff']]) # (not significant)

# quantile_regression(X[['intercept', 'speed', 'int_speed_RT']]) # (not significant)
# quantile_regression(X[['intercept', 'speed', 'int_speed_median']]) # (not significant)
# quantile_regression(X[['intercept', 'speed', 'int_speed_cross']]) # (strong multicollinearity)
# quantile_regression(X[['intercept', 'speed', 'int_speed_night']]) # (not significant)

# quantile_regression(X[['speed', 'int_cross_length']]) # 0.4751
# quantile_regression(X[['intercept', 'speed', 'int_cross_length']]) # (strong multicollinearity)
# quantile_regression(X[['speed', 'int_speed_yellow']]) # 0.4678
# quantile_regression(X[['speed', 'int_cross_length', 'int_speed_yellow']]) # 0.4804
# quantile_regression(X[['intercept', 'speed', 'int_speed_yellow']]) # (strong multicollinearity)

# final model for Xc
model_Xc = quantile_regression(X[['intercept', 'speed']])
df_Xc = model_Xc.summary2().tables[1]
df_Xc.to_csv("output/quantile_regression_model_Xc.csv")

# create dataset for fitting quantile regression line
y_pred = np.linspace(YLR.speed.min(), YLR.speed.max(), 100)
X_pred = pd.DataFrame({'intercept': 1, 'speed': y_pred})
predictions = model_Xc.predict(X_pred)

# plot FTS with computed Xs and fit quantile regression line
plt.scatter(YLR.Xi, YLR.speed, alpha = 0.5, label = 'Data')
plt.plot(predictions, y_pred, color = 'red', label = 'Median regression')
plt.xlabel('Maximum clearing distance (ft)')
plt.ylabel('Speed (ft/s)')
plt.legend()
plt.show()


# =============================================================================
# prepare data for DZ analysis
# =============================================================================

# function to obtain predicted Xs and Xc
def get_predicted_quantile_distance(xdf, group):
    # add variables
    xdf['Group'] = group # group of FTS, YLR, RLR
    xdf['speed_sq'] = xdf.speed**2 # square of speed
    xdf['intercept'] = 1
    
    # add predicted Xs and Xc from quantile models
    xdf['Xs'] = round(model_Xs.predict(xdf[['speed', 'speed_sq']]), 0)
    xdf['Xc'] = round(model_Xc.predict(xdf[['intercept', 'speed']]), 0)
    
    # select relevant columns
    xdf = xdf[['Node', 'Approach', 'ID', 'TripID', 'localtime', 'speed', 'Xi', 'Xs', 'Xc', 'Group']]
    return xdf

# add predicted Xs and Xc to FTS, YLR, RLR datasets
FTS = get_predicted_quantile_distance(FTS, 'FTS')
YLR = get_predicted_quantile_distance(YLR, 'YLR')
RLR = get_predicted_quantile_distance(RLR, 'RLR')

# add stop/go decision to dataset
FTS['Decision'] = 0
YLR['Decision'] = 1
RLR['Decision'] = 1

# combine datasets and save file
ddf = pd.concat([FTS, YLR, RLR], ignore_index = True)
ddf.to_csv("ignore/Wejo/trips_stop_go/data_Xs_Xc.txt", sep = '\t', index = False)
