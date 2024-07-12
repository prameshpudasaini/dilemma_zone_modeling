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
# functions
# =============================================================================

# function to perform adaptive binning, ensuring >= N observations in a bin
def adaptive_binning(data, threshold):
    # count unique speed values and sort data
    cdf = data.speed.value_counts().sort_index().reset_index()
    cdf.columns = ['speed', 'sample_count']

    bins = {} # initialize dictionary to store bins or intervals
    i = 1 # dictionary key

    while True:
        # compute cumulative count
        cdf['cum_count'] = cdf.sample_count.cumsum()
        
        # if cumulative count is less than N, assign the remainder to bins and break loop
        if cdf.cum_count.max() < threshold:
            bins[i] = [cdf.speed.min(), cdf.speed.max()]
            break
        
        # minimum index of where the cumulative count is greater than threshold        
        bin_index = np.where(cdf.cum_count >= threshold)[0][0]
        
        # create df up to this bin index and assign min, max of speed as bin interval
        bdf = cdf.iloc[:bin_index + 1]
        bins[i] = [bdf.speed.min(), bdf.speed.max()]
    
        # filter count data to exclude values up to bin index        
        cdf = cdf.iloc[bin_index + 1:]
        cdf.reset_index(drop = True, inplace = True)
        
        i += 1 # for the next key
    
    # get interval values and convert to pd.Interval type        
    bin_values = list(bins.values())
    intervals = [pd.Interval(left, right, closed = 'both') for left, right in bin_values]
    
    return intervals

# function to map speed to bins or intervals
def map_speed_to_interval(speed, intervals):
    for interval in intervals:
        if interval.left <= speed <= interval.right:
            return interval
        
# function to process intersection geometry features
def model_features(xdf):
    # read node geometry data and select relevant columns
    ndf = pd.read_csv("ignore/node_geometry_v2.csv")
    ndf = ndf[['Node', 'Approach', 'Speed_limit', 'int_cross_length', 'num_TH_lanes', 'has_shared_RT', 'has_median']]

    # merge node geometry data
    xdf = pd.merge(xdf, ndf, on = ['Node', 'Approach'], how = 'left')
    
    # update localtime to datetime and add is_weekend, is_night variables
    xdf.localtime = pd.to_datetime(xdf.localtime)
    xdf['is_weekend'] = (xdf.localtime.dt.dayofweek >= 5).astype(int)
    xdf['is_night'] = (~xdf.localtime.dt.hour.between(5, 19, inclusive = 'both')).astype(int)

    # convert speed and speed limit to ft/s
    xdf.speed = round(xdf.speed * 5280/3600, 1)
    xdf.Speed_limit = round(xdf.Speed_limit * 5280/3600, 1)
    
    return xdf

# function to model median Xs or Xc using quantile regression
def quantile_regression(X):
    model = sm.QuantReg(y, X).fit(q = 0.5)
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

# get bin intervals by adaptive binning of speed
bins_fts = adaptive_binning(FTS[['speed', 'Xi']], 30)
bins_ylr = adaptive_binning(YLR[['speed', 'Xi']], 31)

# add speed bin variable
FTS['speed_bin'] = FTS.speed.apply(lambda x: map_speed_to_interval(x, bins_fts))
YLR['speed_bin'] = YLR.speed.apply(lambda x: map_speed_to_interval(x, bins_ylr))

# px.scatter(FTS, x = 'Xi', y = 'speed').update_traces(marker = dict(size = 7, symbol = 'circle')).show()
# px.scatter(YLR, x = 'Xi', y = 'speed').update_traces(marker = dict(size = 7, symbol = 'cross')).show()

bin_count_fts = FTS.speed_bin.value_counts().reset_index(name = 'sample_count')
bin_count_ylr = YLR.speed_bin.value_counts().reset_index(name = 'sample_count')


# =============================================================================
# modeling of Xs/Xc using 5th, 95th quantiles
# =============================================================================

# specify quantiles for Xs/Xc modeling
lower_q, upper_q = 0.05, 0.95

# model dataset FTS: group by speed bin and model Xs as the 5th percentile of Xi
fdf = FTS.groupby('speed_bin')['Xi'].quantile(lower_q).reset_index(name = 'Xs')
fdf['lower_speed'] = fdf.speed_bin.apply(lambda x: x.left).astype(float) # lower speed in bin
fdf['upper_speed'] = fdf.speed_bin.apply(lambda x: x.right).astype(float) # upper speed in bin

# model dataset YLR: group by speed bin and model Xc as the 95th percentile of Xi
ydf = YLR.groupby('speed_bin')['Xi'].quantile(upper_q).reset_index(name = 'Xc')
ydf['lower_speed'] = ydf.speed_bin.apply(lambda x: x.left).astype(float) # lower speed in bin
ydf['upper_speed'] = ydf.speed_bin.apply(lambda x: x.right).astype(float) # upper speed in bin

# pivot wider Xs and Xc datasets
fdf_long = pd.melt(fdf, id_vars = ['Xs'], value_vars = ['lower_speed', 'upper_speed'], var_name = 'range', value_name = 'speed')
ydf_long = pd.melt(ydf, id_vars = ['Xc'], value_vars = ['lower_speed', 'upper_speed'], var_name = 'range', value_name = 'speed')

# # plot of Xs vs speed bin
# px.scatter(
#     fdf_long,
#     x = 'Xs',
#     y = 'speed',
#     color = 'range'
# ).update_traces(marker = dict(size = 7, symbol = 'circle')).show()

# # plot of Xc vs speed bin
# px.scatter(
#     ydf_long,
#     x = 'Xc',
#     y = 'speed',
#     color = 'range'
# ).update_traces(marker = dict(size = 7, symbol = 'cross')).show()

# merge Xs/Xc to FTS/YLR datasets
FTS = pd.merge(FTS, fdf, on = 'speed_bin', how = 'left')
YLR = pd.merge(YLR, ydf, on = 'speed_bin', how = 'left')


# =============================================================================
# quantile regression models for Xs
# =============================================================================

# get features for modeling
fdata = model_features(FTS)

# filter out vehicles stopping beyond the stop line
fdata = fdata[fdata.stop_dist >= -10]

# # plot: Xs vs speed
px.scatter(FTS, x = 'Xs', y = 'speed').update_traces(marker = dict(size = 7, symbol = 'circle')).show()

# exclude first and last speed bins
first_speed_bin, last_speed_bin = fdata.speed_bin.min(), fdata.speed_bin.max()
fdata = fdata[~fdata.speed_bin.isin([first_speed_bin, last_speed_bin])]

# # plot Xs vs speed
# px.scatter(fdata, x = 'Xs', y = 'speed').update_traces(marker = dict(size = 7, symbol = 'circle')).show()

# generate features as interaction terms
fdata['int_speed_RT'] = fdata.speed * fdata.has_shared_RT
fdata['int_speed_median'] = fdata.speed * fdata.has_median
fdata['int_speed_cross'] = fdata.speed * fdata.int_cross_length
fdata['int_speed_night'] = fdata.speed * fdata.is_night

# add square of speed as predictor
fdata['speed_sq'] = fdata.speed ** 2

# add difference of speed from speed limit as predictor
fdata['speed_diff'] = fdata.speed - fdata.Speed_limit

# target and predictor variables
X = fdata.copy()
y = fdata['Xs']

# testing different models for Xs
quantile_regression(X[['speed']]) # 0.4890
quantile_regression(X[['Speed_limit']]) # 0.04872
quantile_regression(X[['speed', 'speed_diff']]) # 0.5623
quantile_regression(X[['speed', 'speed_sq']]) # 0.6108 (**best model**)
quantile_regression(X[['speed', 'speed_sq', 'speed_diff']]) # 0.6115 (slight improvement)
quantile_regression(X[['speed', 'speed_sq', 'Speed_limit']]) # 0.6115 (strong multicollinearity)

quantile_regression(X[['speed', 'speed_sq', 'int_speed_RT']]) # 0.6108 (not significant)
quantile_regression(X[['speed', 'speed_sq', 'int_speed_median']]) # 0.6109 (not significant)
quantile_regression(X[['speed', 'speed_sq', 'int_speed_cross']]) # 0.6109 (strong multicollinearity)
quantile_regression(X[['speed', 'speed_sq', 'int_speed_night']]) # 0.6108 (not significant)

quantile_regression(X[['speed', 'speed_sq', 'speed_diff', 'int_speed_RT']]) # 0.6120 (strong multicollinearity)
quantile_regression(X[['speed', 'speed_sq', 'speed_diff', 'int_speed_median']]) # 0.6117 (strong multicollinearity)
quantile_regression(X[['speed', 'speed_sq', 'speed_diff', 'int_speed_cross']]) # 0.6120 (strong multicollinearity)
quantile_regression(X[['speed', 'speed_sq', 'speed_diff', 'int_speed_night']]) # 0.6116 (not significant)

# final model for Xs
model_Xs = quantile_regression(X[['speed', 'speed_sq']])

# create dataset for fitting quantile regression line
y_pred = np.linspace(fdata.speed.min(), fdata.speed.max(), 100)
X_pred = pd.DataFrame({'speed': y_pred, 'speed_sq': y_pred**2})
predictions = model_Xs.predict(X_pred)

# plot FTS with computed Xs and fit quantile regression line
plt.scatter(fdata.Xi, fdata.speed, alpha = 0.5, label = 'Data')
plt.plot(predictions, y_pred, color = 'red', label = 'Median regression')
plt.xlabel('Minimum stopping distance (ft)')
plt.ylabel('Speed (ft/s)')
plt.legend()
plt.show()


# =============================================================================
# quantile regression models for Xc
# =============================================================================

# get features for modeling
ydata = model_features(YLR)

# filter out vehicles with yellow onset beyond stop line
ydata = ydata[ydata.Xi >= -10]

# # plot: Xc vs speed
# px.scatter(YLR, x = 'Xc', y = 'speed').update_traces(marker = dict(size = 7, symbol = 'cross')).show()

# exclude first and last speed bins
first_speed_bin, last_speed_bin = ydata.speed_bin.min(), ydata.speed_bin.max()
ydata = ydata[~ydata.speed_bin.isin([first_speed_bin, last_speed_bin])]

# # plot Xs vs speed
# px.scatter(ydata, x = 'Xc', y = 'speed').update_traces(marker = dict(size = 7, symbol = 'cross')).show()

# generate features as interaction terms
ydata['int_speed_RT'] = ydata.speed * ydata.has_shared_RT
ydata['int_speed_median'] = ydata.speed * ydata.has_median
ydata['int_speed_cross'] = ydata.speed * ydata.int_cross_length
ydata['int_speed_night'] = ydata.speed * ydata.is_night

# add difference of speed from speed limit as predictor
ydata['speed_diff'] = ydata.speed - ydata.Speed_limit

# add intercept with value 1
ydata['intercept'] = 1

# target and predictor variables
X = ydata.copy()
y = ydata['Xc']
    
# testing different models for Xc
quantile_regression(X[['speed']]) # 0.5829
quantile_regression(X[['Speed_limit']]) # 0.01387
quantile_regression(X[['intercept', 'speed']]) # 0.5963 (**best model**)
quantile_regression(X[['intercept', 'speed', 'speed_diff']]) # 0.5968 (slight improvement)

quantile_regression(X[['intercept', 'speed', 'int_speed_RT']]) # 0.5963 (not significant)
quantile_regression(X[['intercept', 'speed', 'int_speed_median']]) # 0.5963 (not significant)
quantile_regression(X[['intercept', 'speed', 'int_speed_cross']]) # 0.5963 (strong multicollinearity)
quantile_regression(X[['intercept', 'speed', 'int_speed_night']]) # 0.5970 (not significant)

# final model for Xs
model_Xc = quantile_regression(X[['intercept', 'speed']])

# create dataset for fitting quantile regression line
y_pred = np.linspace(ydata.speed.min(), ydata.speed.max(), 100)
X_pred = pd.DataFrame({'intercept': 1, 'speed': y_pred})
predictions = model_Xc.predict(X_pred)

# plot FTS with computed Xs and fit quantile regression line
plt.scatter(ydata.Xi, ydata.speed, alpha = 0.5, label = 'Data')
plt.plot(predictions, y_pred, color = 'red', label = 'Median regression')
plt.xlabel('Maximum clearing distance (ft)')
plt.ylabel('Speed (ft/s)')
plt.legend()
plt.show()

# =============================================================================
# DZ analysis
# =============================================================================

# function to obtain median Xs and Xc
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

# combine datasets
ddf = pd.concat([FTS, YLR, RLR], ignore_index = True)

# compute zone vehicle's position is in
ddf.loc[(((ddf.Xi <= ddf.Xc) & (ddf.Xc <= ddf.Xs)) | ((ddf.Xi <= ddf.Xs) & (ddf.Xs <= ddf.Xc))), 'zone'] = 'Should go'
ddf.loc[(((ddf.Xi >= ddf.Xc) & (ddf.Xc >= ddf.Xs)) | ((ddf.Xi >= ddf.Xs) & (ddf.Xs >= ddf.Xc))), 'zone'] = 'Should stop'
ddf.loc[((ddf.Xc < ddf.Xi) & (ddf.Xi < ddf.Xs)), 'zone'] = 'Dilemma'
ddf.loc[((ddf.Xs < ddf.Xi) & (ddf.Xi < ddf.Xc)), 'zone'] = 'Option'

print(f"{ddf.zone.value_counts()}")
print(f"{ddf.groupby('zone')['Decision'].value_counts()}")
print(f"{ddf.groupby(['Node', 'zone'])['Decision'].value_counts()}")

# =============================================================================
# comparison with FHWA method
# =============================================================================

def modelFHWA(v0, node):
    # specify v85 (speed limit) and yellow interval based on node
    if node == 540:
        v85, y = 30, 3.5
    elif node in [217, 444, 446]:
        v85, y = 35, 4
    elif node in [586, 618]:
        v85, y = 40, 4

    v0, v85 = v0*5280/3600, v85*5280/3600 # mph to ft/s

    prt = 1
    dec = 10
    acc = (16 - 0.213 * v0)
    
    Xs = v0*prt + ((v0**2) / (2*dec))
    Xc = v0*y + 0.5*(acc)*((y - prt)**2)
    
    return {'Xs': Xs, 'Xc': Xc}

ddf['Xs_FHWA'] = ddf.apply(lambda x: modelFHWA(x.speed, x.Node)['Xs'], axis = 1)
ddf['Xc_FHWA'] = ddf.apply(lambda x: modelFHWA(x.speed, x.Node)['Xc'], axis = 1)

print(f"Correlation Xs and Xs_FHWA: {ddf.Xs.corr(ddf.Xs_FHWA)}")
print(f"Correlation Xc and Xc_FHWA: {ddf.Xc.corr(ddf.Xc_FHWA)}")

# =============================================================================
# comparison with Li's method
# =============================================================================

def modelLi2013(v0, node):
    # specify v85 (speed limit) and yellow interval based on node
    if node == 540:
        v85, y = 30, 3.5
    elif node in [217, 444, 446]:
        v85, y = 35, 4
    elif node in [586, 618]:
        v85, y = 40, 4

    v0, v85 = v0*5280/3600, v85*5280/3600 # mph to ft/s

    prt = 0.274 + 30.392/v0
    dec = np.exp(3.572 - 25.013/v0) - 17.855 + 480.558/v85
    acc = -23.513 + 658.948/v0 + 0.223*v85
    
    Xs = v0*prt + ((v0**2) / (2*dec))
    Xc = v0*y + 0.5*(acc)*((y - prt)**2)
    
    return {'Xs': Xs, 'Xc': Xc}

ddf['Xs_Li'] = ddf.apply(lambda x: modelLi2013(x.speed, x.Node)['Xs'], axis = 1)
ddf['Xc_Li'] = ddf.apply(lambda x: modelLi2013(x.speed, x.Node)['Xc'], axis = 1)

print(f"Correlation Xs and Xs_Li: {ddf.Xs.corr(ddf.Xs_Li)}")
print(f"Correlation Xc and Xc_Li: {ddf.Xc.corr(ddf.Xc_Li)}")

# plot of Xs vs Xs_Li
px.scatter(
    ddf,
    x = 'Xs',
    y = 'Xs_Li',
    facet_col = 'Node'
).update_traces(marker = dict(size = 7, symbol = 'circle')).show()

# plot of Xc vs Xc_Li
px.scatter(
    ddf,
    x = 'Xc',
    y = 'Xc_Li',
    facet_col = 'Node'
).update_traces(marker = dict(size = 7, symbol = 'circle')).show()
