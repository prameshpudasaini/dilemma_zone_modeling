import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\dilemma_zone_modeling")

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

# function to plot yellow onset speed vs distance
def plot_yellow_onset_speed_distance(xdf, group):
    ydf = xdf.copy()[xdf.Xo == 1] # observations with Xs or Xc
    
    plt.figure(figsize = (12, 8))
    marker = 'o' if group == 'FTS' else 's'
    
    # plot first-to-stop or crossing vehicles with xdf
    # plot Xs or Xc observations with ydf
    plt.scatter(xdf['Xi'], xdf['speed'], color = 'black', marker = marker, facecolors = 'none', alpha = 0.6)
    plt.scatter(ydf['Xi'], ydf['speed'], color = 'black', marker = marker)
    
    plt.xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
    plt.ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.show()
    
    
# =============================================================================
# read and prepare datasets
# =============================================================================

# read model datasets
FTS = pd.read_csv("ignore/Wejo/trips_analysis/trips_FTS.txt", sep = '\t')
YLR = pd.read_csv("ignore/Wejo/trips_analysis/trips_YLR.txt", sep = '\t')

# px.scatter(FTS, x = 'Xi', y = 'speed').show()
# px.scatter(YLR, x = 'Xi', y = 'speed').show()

# process predictors
FTS = process_predictors(FTS)
YLR = process_predictors(YLR)


# =============================================================================
# quantile regression tests with different quantiles and bin sizes
# =============================================================================

quantiles_FTS, quantiles_YLR = [0.05, 0.15, 0.5], [0.5, 0.85, 0.95]
bin_sizes = [20, 25, 30, 35]

def quantile_regression_test(xdf, bin_size, q, group):
    list_site_df = []
    for site in list(xdf.SiteID.unique()):
        site_df = xdf.copy()[xdf.SiteID == site]
        num_bins = int(len(site_df) / bin_size)
        site_df = identify_Xs_Xc(site_df, num_bins, group)
        list_site_df.append(site_df)
        
    sdf = pd.concat(list_site_df, ignore_index = True)
    sdf = sdf[sdf.Xo == 1] # Xs or Xc observations
    
    plot_yellow_onset_speed_distance(sdf, group)
    
    sdf['constant'] = 1 # constant term for regression
    
    if group == 'FTS':
        predictors = ['speed_fps', 'speed_sq', 'Crossing_length', 'int_speed_peak', 'int_speed_night', 'int_speed_weekend']
    elif group == 'YLR':
        predictors = ['constant', 'speed_fps', 'is_PSL35', 'int_speed_peak', 'int_speed_night', 'int_speed_weekend']
    
    X = sdf[predictors]
    y = sdf['Xi']
    qmodel = sm.QuantReg(y, X).fit(q = q)
    print(qmodel.summary())
        
# quantile regression tests for FTS vehicles
for q in quantiles_FTS:
    for bin_size in bin_sizes:
        print(f"\nQuantile, bin size: {q}, {bin_size}")
        quantile_regression_test(FTS, bin_size, q, 'FTS')
        
# quantile regression tests for YLR vehicles
for q in quantiles_YLR:
    for bin_size in bin_sizes:
        print(f"\nQuantile, bin size: {q}, {bin_size}")
        quantile_regression_test(YLR, bin_size, q, 'YLR')
        
# Observations and conclusions
# Check Xs with q = 0.5, b = [20, 25]
# Check Xc with q = 0.5, b = [20, 25]
