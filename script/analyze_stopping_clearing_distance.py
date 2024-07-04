import os
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\dilemma_Wejo")

# =============================================================================
# read & prepare datasets
# =============================================================================

# read datasets with stop/go trips
GLR = pd.read_csv("ignore/Wejo/trips_stop_go/GLR_filtered.txt", sep = '\t') # GLR trips
FTS = pd.read_csv("ignore/Wejo/trips_stop_go/FTS_filtered.txt", sep = '\t') # FTS trips
YLR = pd.read_csv("ignore/Wejo/trips_stop_go/YLR_filtered.txt", sep = '\t') # YLR trips
RLR = pd.read_csv("ignore/Wejo/trips_stop_go/RLR_filtered.txt", sep = '\t') # RLR trips

# combine GLR with FTS data
FTS = pd.concat([FTS, GLR], ignore_index = True)

# # combine RLR with YLR data
# YLR = pd.concat([YLR, RLR], ignore_index = True)

# read node geometry data and select relevant columns
ndf = pd.read_csv("ignore/node_geometry.csv")[['Node', 'Approach', 'Speed_limit']]
# ndf.drop(['Name', 'Latitude', 'Longitude', 'has_shared_LT', 'num_RT_lanes', 'dist_stop_cross'], axis = 1, inplace = True)

# # create datasets for stop/go trips
# GLR['Group'], FTS['Group'], YLR['Group'], RLR['Group'] = 'GLR', 'FTS', 'YLR', 'RLR'
# sdf = pd.concat([FTS, GLR], ignore_index = True) # stop trips dataset
# gdf = pd.concat([YLR, RLR], ignore_index = True) # go trips dataset

# merge node geometry data
FTS = pd.merge(FTS, ndf, on = ['Node', 'Approach'], how = 'left')
YLR = pd.merge(YLR, ndf, on = ['Node', 'Approach'], how = 'left')

# print(f"FTS by speed limit: \n{FTS.Speed_limit.value_counts()}")
# print(f"YLR by speed limit: \n{YLR.Speed_limit.value_counts()}")

# Note: node 540 has small sample size for further analysis
# filter out node 540 from FTS and YLR datasets
FTS = FTS[FTS.Node != 540]
YLR = YLR[YLR.Node != 540]
RLR = RLR[RLR.Node != 540]

# add TTS: travel time to stop line variable
FTS['TSL'] = round(FTS.Xi / (FTS.speed * 5280/3600), 2)
YLR['TSL'] = round(YLR.Xi / (YLR.speed * 5280/3600), 2)

# print(f"FTS by node: \n{FTS.Node.value_counts()}")
# print(f"YLR by node: \n{YLR.Node.value_counts()}")

# round speed to 0 decimals to create speed bin
FTS['speed_bin'] = round(FTS.speed, 0)
YLR['speed_bin'] = round(YLR.speed, 0)

# get sample count by approach
nFTS = FTS[['Node', 'Approach']].value_counts().reset_index()
nYLR = YLR[['Node', 'Approach']].value_counts().reset_index()
nRLR = RLR[['Node', 'Approach']].value_counts().reset_index()

# add group for each sample count dataset
nFTS['Group'] = 'FTS'
nYLR['Group'] = 'YLR'
nRLR['Group'] = 'RLR'

# combine all sample count datasets
sample_node = pd.concat([nFTS, nYLR, nRLR], ignore_index = True)
sample_node.columns = ['Node', 'Approach', 'N', 'Group']

# pivot wider for each group
sample_node = sample_node.pivot(index = ['Node', 'Approach'], columns = 'Group', values = 'N').reset_index()

# =============================================================================
# histogram of speed by speed limit
# =============================================================================

# FTS35 = FTS.copy()[FTS.Speed_limit == 35]
# FTS40 = FTS.copy()[FTS.Speed_limit == 40]

px.histogram(
    FTS,
    x = 'speed_bin',
    facet_col = 'Speed_limit',
    nbins = 30,
    title = 'Histogram of yellow onset speed'
).show()

px.histogram(
    YLR,
    x = 'speed_bin',
    facet_col = 'Speed_limit',
    nbins = 30,
    title = 'Histogram of yellow onset speed'
).show()

# check skewness and kurtosis of speed
skewness = {'FTS': FTS.speed.skew(), 'YLR': YLR.speed.skew()}
kurtosis = {'FTS': FTS.speed.kurt(), 'YLR': YLR.speed.kurt()}

# Notes
# Skewness = 0: data is symmetrically distributed.
# Kurtosis = 3: data has a normal distribution.

# =============================================================================
# visualization by speed limit
# =============================================================================

# FTS: plot yellow onset distance vs speed
px.scatter(
    FTS,
    x = 'Xi',
    y = 'speed',
    facet_col = 'Speed_limit'
).update_traces(marker = dict(size = 7, symbol = 'circle')).show()

# FTS: plot time to stop line vs speed
px.scatter(
    FTS,
    x = 'TSL',
    y = 'speed',
    facet_col = 'Speed_limit'
).update_traces(marker = dict(size = 7, symbol = 'circle')).show()

# YLR: plot yellow onset distance vs speed
px.scatter(
    YLR,
    x = 'Xi',
    y = 'speed',
    facet_col = 'Speed_limit'
).update_traces(marker = dict(size = 7, symbol = 'cross')).show()

# YLR: plot time to stop line vs speed
px.scatter(
    YLR,
    x = 'TSL',
    y = 'speed',
    facet_col = 'Speed_limit'
).update_traces(marker = dict(size = 7, symbol = 'circle')).show()

# =============================================================================
# visualization by node and approach
# =============================================================================

# FTS: plot yellow onset distance vs speed
px.scatter(
    FTS,
    x = 'Xi',
    y = 'speed',
    facet_col = 'Node',
    facet_row = 'Approach'
).update_traces(marker = dict(size = 7, symbol = 'circle')).show()

# YLR: plot yellow onset distance vs speed
px.scatter(
    YLR,
    x = 'Xi',
    y = 'speed',
    facet_col = 'Node',
    facet_row = 'Approach'
).update_traces(marker = dict(size = 7, symbol = 'circle')).show()

# =============================================================================
# compute max deceleration and max acceleration
# =============================================================================

# compute deceleration based on yellow onset speed and distance
def deceleration(v0, Xs):
    v0 = v0 * 5280/3600 # mph to ft/s
    prt = 0.274 + 30.392/v0 # from Li's model (Li & Wei, 2013)
    dec = (v0**2) / (2*(Xs - v0*prt))
    return round(dec, 2)

# compute acceleration based on yellow onset speed, distance, and time to cross stop line
def acceleration(v0, Xs, t):
    v0 = v0 * 5280/3600 # mph to ft/s
    prt = 0.274 + 30.392/v0 # from Li's model (Li & Wei, 2013)
    acc = 2*(Xs - v0*t) / ((t - prt)**2)
    return round(acc, 2)

FTS['dec'] = deceleration(FTS.speed, FTS.Xi)
YLR['acc'] = acceleration(YLR.speed, YLR.Xi, YLR.cross_time)

FTS['dec'] = round((FTS.speed*5280/3600)**2 / (2*(FTS.Xi)), 2)

# =============================================================================
# check consistency of Wei's and Li's Type I DZ parameters
# =============================================================================

# function to compute Type I DZ parameters using Wei's or Li's models
def paramTypeI(v0, v85, model):
    v0, v85 = v0*5280/3600, v85*5280/3600 # mph to ft/s
    
    if model == 'Wei':
        prt = 0.445 + 21.478/v0
        dec = np.exp(3.379 - 36.099/v0) - 9.722 + 429.692/v85
        acc = -27.91 + 760.258/v0 + 0.266*v85
    elif model == 'Li':
        prt = 0.274 + 30.392/v0
        dec = np.exp(3.572 - 25.013/v0) - 17.855 + 480.558/v85
        acc = -23.513 + 658.948/v0 + 0.223*v85
    return {'prt': round(prt, 2), 
            'dec': round(dec, 2),
            'acc': round(acc, 2)}

# function to check consistency of Type I model
def paramConsistencyTypeI(xdf):
    param_Wei = paramTypeI(xdf.speed, xdf.Speed_limit, 'Wei')
    param_Li = paramTypeI(xdf.speed, xdf.Speed_limit, 'Li')

    xdf['prt_Wei'] = param_Wei['prt']
    xdf['max_dec_Wei'] = param_Wei['dec']
    xdf['max_acc_Wei'] = param_Wei['acc']

    xdf['prt_Li'] = param_Li['prt']
    xdf['max_dec_Li'] = param_Li['dec']
    xdf['max_acc_Li'] = param_Li['acc']

    print("Correlation check:")
    print(f"PRT: {xdf.prt_Wei.corr(xdf.prt_Li)}")
    print(f"PRT: {xdf.max_dec_Wei.corr(xdf.max_dec_Li)}")
    print(f"PRT: {xdf.max_acc_Wei.corr(xdf.max_acc_Li)}")
    
    xdf.drop(xdf.filter(like = '_Wei').columns, axis = 1, inplace = True)
    xdf.rename(columns = {col: col.replace('_Li', '') for col in xdf.columns if col.endswith('_Li')}, inplace = True)
    
    return xdf

FTS = paramConsistencyTypeI(FTS)
YLR = paramConsistencyTypeI(YLR)

# Note: Type I parameters estimated from Wei's & Li's models are almost same
# with 0.99 correlation.
# Use Li's model for further analysis.

FTS['Decision'] = 0
YLR['Decision'] = 1

# =============================================================================
# compute Xs and Xc
# =============================================================================

# minimum stopping distance
def minStoppingDistance(v0, prt, dec):
    v0 = v0*5280/3600
    Xs = v0*prt + ((v0**2) / (2*dec))
    return round(Xs, 0)

# maximum clearing distance
def maxClearingDistance(v0, prt, acc):
    v0 = v0*5280/3600
    y = 3 # yellow interval
    Xc = v0*y + 0.5*(acc)*((y - prt)**2)
    return round(Xc, 0)

# columns for dilemma analysis
dz_cols = ['Node', 'Approach', 'speed', 'localtime', 'Xi', 'Speed_limit',
           'prt', 'max_dec', 'max_acc', 'Decision']

# create dataset for DZ analysis by combining FTS and YLR datasets
ddf = pd.concat([FTS[dz_cols], YLR[dz_cols]], ignore_index = True)

ddf['Xs'] = minStoppingDistance(ddf.speed, ddf.prt, ddf.max_dec)
ddf['Xc'] = maxClearingDistance(ddf.speed, ddf.prt, ddf.max_acc)

# compute zone vehicle's position is in
ddf.loc[(((ddf.Xi <= ddf.Xc) & (ddf.Xc <= ddf.Xs)) | ((ddf.Xi <= ddf.Xs) & (ddf.Xs <= ddf.Xc))), 'zone'] = 'Should go'
ddf.loc[(((ddf.Xi >= ddf.Xc) & (ddf.Xc >= ddf.Xs)) | ((ddf.Xi >= ddf.Xs) & (ddf.Xs >= ddf.Xc))), 'zone'] = 'Should stop'
ddf.loc[((ddf.Xc < ddf.Xi) & (ddf.Xi < ddf.Xs)), 'zone'] = 'Dilemma'
ddf.loc[((ddf.Xs < ddf.Xi) & (ddf.Xi < ddf.Xc)), 'zone'] = 'Option'

ddf.zone.value_counts()
ddf.groupby('Decision')['zone'].value_counts()

# # compute deceleration for each speed bin
# FTS['dec_dist'] = round((FTS.speed*5280/3600)**2 / (2*(FTS.Xi - FTS.stop_dist)), 2)
# FTS['dec_time'] = round(FTS.speed * 5280/3600 / FTS.stop_time, 2)

# # maximum deceleration by each study site
# max_dec_node = FTS.groupby(['Node', 'Approach'])['dec'].apply(lambda x: x.max()).reset_index()

# # count number of samples in each speed bin
# print(f"Num of samples in each speed bin: \n{FTS.speed_bin.value_counts()}")
# sample_speed_FTS = FTS.speed_bin.value_counts().reset_index()
# sample_speed_FTS.columns = ['speed_bin', 'sample_count']

# # filter speed bins with statistically enough sample size
# speed_bin_Xs = list(sample_speed_FTS[sample_speed_FTS.sample_count >= 20].speed_bin.unique())

# # compute Xs for each speed bin
# MSD = FTS.copy()[FTS.speed_bin.isin(speed_bin_Xs)]
# max_dec_speed = MSD.groupby('speed_bin')['dec'].apply(lambda x: x.max()).reset_index()

# # add number of observations to max dec speed
# max_dec_speed = pd.merge(max_dec_speed, sample_speed_FTS, on = 'speed_bin', how = 'inner')
