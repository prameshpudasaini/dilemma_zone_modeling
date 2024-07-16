import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\dilemma_Wejo")

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

# =============================================================================
# speed binning and processing features
# =============================================================================

# specify parameters for quantile binning
num_bins, quantile_Xs, quantile_Xc = 40, 0.00, 1.00

# function to create bins based on adaptive binning of quantiles
def process_speed_bin(zdf, group):
    xdf = zdf.copy()

    # process speed data
    xdf['speed'] = xdf.speed * 5280/3600 # mph to ft/s
    xdf['speed_sq'] = xdf.speed ** 2 # square of speed
    
    # compute mean and st dev; filter out observations beyond 2 st dev
    mean, std = xdf.speed.mean(), xdf.speed.std()
    xdf = xdf[xdf.speed.between(mean - 2*std, mean + 2*std, inclusive = 'both')]
    
    # adaptive binning of data by speed variable
    xdf['speed_bin'] = pd.qcut(xdf.speed, q = num_bins, duplicates = 'drop')
    
    # assign variable name and quantile to compute based on FTS/YLR group
    # add limiting values by speed bin as variable
    if group == 'FTS':
        var_name, q = 'Xs', quantile_Xs
        limit_values = xdf.groupby('speed_bin').Xi.transform('min')
    else:
        var_name, q = 'Xc', quantile_Xc
        limit_values = xdf.groupby('speed_bin').Xi.transform('max')
    
    # create a df of limit values
    xdf['limit_Xi'] = (xdf.Xi == limit_values).astype(int)

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

# process speed bins for FTS/YLR datasets
fdf = process_speed_bin(FTS, 'FTS')
ydf = process_speed_bin(YLR, 'YLR')

# get features for modeling
fdf = geometry_features(fdf)
ydf = geometry_features(ydf)

# px.scatter(fdf, x = 'Xs', y = 'speed').update_traces(marker = dict(size = 7, symbol = 'circle')).show()
# px.scatter(ydf, x = 'Xc', y = 'speed').update_traces(marker = dict(size = 7, symbol = 'circle')).show()

plot_Xs = fdf.copy()[fdf.limit_Xi == 1][['Xi', 'speed']]
plot_Xs.speed = round(plot_Xs.speed*3600/5280, 1)

plot_Xc = ydf.copy()[ydf.limit_Xi == 1][['Xi', 'speed']]
plot_Xc.speed = round(plot_Xc.speed*3600/5280, 1)

# create subplots of Xi vs speed for FTS and YLR
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))

ax1.scatter(FTS.Xi, FTS.speed, color = 'black', marker = 'o', facecolors = 'none', alpha = 0.6, label = 'FTS vehicles')
ax1.scatter(plot_Xs.Xi, plot_Xs.speed, color = 'black', marker = 'o', label = 'FTS vehicles with $X_s$')
ax1.set_title('(a) First-to-stop (FTS) vehicles', fontsize = 14, fontweight = 'bold')
ax1.set_xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
ax1.set_ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax1.legend(loc = 'lower right')

ax2.scatter(YLR.Xi, YLR.speed, color = 'black', marker = 's', facecolors = 'none', alpha = 0.6, label = 'YLR')
ax2.scatter(plot_Xc.Xi, plot_Xc.speed, color = 'black', marker = 's', label = 'YLR vehicles with $X_c$')
ax2.set_title('(b) Yellow light running (YLR) vehicles', fontsize = 14, fontweight = 'bold')
ax2.set_xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
ax2.set_ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax2.legend(loc = 'lower right')

plt.tight_layout(pad = 1)
plt.savefig('output/plot_Xs_Xc.png', dpi = 600)
plt.show()


# =============================================================================
# quantile regression models for Xs
# =============================================================================

# function to model median Xs or Xc using quantile regression
def quantile_regression(X, q = 0.5):
    model = sm.QuantReg(y, X).fit(q = q)
    print(model.summary())
    return model

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

# 85th quantile model for Xs
model_Xs_q15 = quantile_regression(X[['speed', 'speed_sq']], q = 0.15)
df_Xs_q15 = model_Xs_q15.summary2().tables[1]
df_Xs_q15.to_csv("output/quantile_regression_model_Xs_q15.csv")    
        
# create dataset for fitting quantile regression line
y_val = np.linspace(FTS.speed.min()*5280/3600, FTS.speed.max()*5280/3600, 100)
X_val = pd.DataFrame({'speed': y_val, 'speed_sq': y_val**2})
X_pred_Xs = model_Xs.predict(X_val)
y_val_Xs = np.round(y_val*3600/5280, decimals = 1)

# # plot FTS with computed Xs and fit quantile regression line
# plt.figure(figsize = (8, 6))
# plt.scatter(FTS.Xi, FTS.speed, color = 'black', alpha = 0.6, label = 'First-to-stop vehicles')
# plt.scatter(plot_Xs.Xi, plot_Xs.speed, color = 'blue', label = 'First-to-stop vehicles with $X_s$')
# plt.plot(X_pred_Xs, y_val_Xs, color = 'red', label = 'Quantile regression model for $X_s$')
# plt.xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
# plt.ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
# plt.xticks(fontsize = 12)
# plt.yticks(fontsize = 12)

# legend = plt.legend(loc = 'upper center', fontsize = 12, bbox_to_anchor = (0.5, 1.13), ncol = 2)
# frame = legend.get_frame()
# frame.set_facecolor('white')
# frame.set_edgecolor('white')

# plt.tight_layout(pad = 0)
# plt.savefig('output/quantile_regression_model_Xs.png', dpi = 600)
# plt.show()


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

# 85th quantile model for Xc
model_Xc_q85 = quantile_regression(X[['intercept', 'speed']], q = 0.85)
df_Xc_q85 = model_Xc_q85.summary2().tables[1]
df_Xc_q85.to_csv("output/quantile_regression_model_Xc_q85.csv")

# create dataset for fitting quantile regression line
y_val = np.linspace(YLR.speed.min()*5280/3600, YLR.speed.max()*5280/3600, 100)
X_val = pd.DataFrame({'intercept': 1, 'speed': y_val})
X_pred_Xc = model_Xc.predict(X_val)
y_val_Xc = np.round(y_val*3600/5280, decimals = 1)

# # plot FTS with computed Xs and fit quantile regression line
# plt.figure(figsize = (8, 6))
# plt.scatter(YLR.Xi, YLR.speed, color = 'black', marker = 's', alpha = 0.6, label = 'Yellow light running vehicles')
# plt.scatter(plot_Xc.Xi, plot_Xc.speed, color = 'blue', marker = 's', label = 'Yellow light running vehicles with $X_c$')
# plt.plot(X_pred_Xc, y_val_Xc, color = 'red', label = 'Quantile regression model for $X_c$')
# plt.xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
# plt.ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
# plt.xticks(fontsize = 12)
# plt.yticks(fontsize = 12)

# legend = plt.legend(loc = 'upper center', fontsize = 12, bbox_to_anchor = (0.5, 1.13), ncol = 2)
# frame = legend.get_frame()
# frame.set_facecolor('white')
# frame.set_edgecolor('white')

# plt.tight_layout(pad = 0)
# plt.savefig('output/quantile_regression_model_Xc.png', dpi = 600)
# plt.show()

# create subplots of Xi vs speed for FTS and YLR
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))

ax1.scatter(FTS.Xi, FTS.speed, color = 'black', marker = 'o', facecolors = 'none', alpha = 0.6, label = 'FTS vehicles')
ax1.scatter(plot_Xs.Xi, plot_Xs.speed, color = 'black', marker = 'o', label = 'FTS vehicles with $X_s$')
ax1.plot(X_pred_Xs, y_val_Xs, color = 'red', linewidth = 2, label = 'Median regression for $X_s$')
ax1.set_title('(a) First-to-stop (FTS) vehicles', fontsize = 14, fontweight = 'bold')
ax1.set_xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
ax1.set_ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax1.legend(loc = 'lower right')

ax2.scatter(YLR.Xi, YLR.speed, color = 'black', marker = 's', facecolors = 'none', alpha = 0.6, label = 'YLR')
ax2.scatter(plot_Xc.Xi, plot_Xc.speed, color = 'black', marker = 's', label = 'YLR vehicles with $X_c$')
ax2.plot(X_pred_Xc, y_val_Xc, color = 'red', linewidth = 2, label = 'Median regression model for $X_c$')
ax2.set_title('(b) Yellow light running (YLR) vehicles', fontsize = 14, fontweight = 'bold')
ax2.set_xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
ax2.set_ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax2.legend(loc = 'lower right')

plt.tight_layout(pad = 1)
plt.savefig('output/model_Xs_Xc.png', dpi = 600)
plt.show()


# =============================================================================
# prepare data for DZ analysis
# =============================================================================

# function to obtain predicted Xs and Xc
def get_predicted_quantile_distance(xdf, group):
    # add variables
    xdf['Group'] = group # group of FTS, YLR, RLR
    xdf['speed_mph'] = xdf.speed
    xdf.speed = round(xdf.speed * 5280/3600, 1)
    xdf['speed_sq'] = xdf.speed**2 # square of speed
    xdf['intercept'] = 1
    
    # add predicted Xs and Xc from quantile models
    xdf['Xs'] = round(model_Xs.predict(xdf[['speed', 'speed_sq']]), 0)
    xdf['Xs_q15'] = round(model_Xs_q15.predict(xdf[['speed', 'speed_sq']]), 0)
    xdf['Xc'] = round(model_Xc.predict(xdf[['intercept', 'speed']]), 0)
    xdf['Xc_q85'] = round(model_Xc_q85.predict(xdf[['intercept', 'speed']]), 0)
    
    # select relevant columns
    xdf = xdf[['Node', 'Approach', 'ID', 'TripID', 'localtime', 'speed_mph', 'speed', 'Xi', 
               'Xs', 'Xc', 'Xs_q15', 'Xc_q85', 'Group']]
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
