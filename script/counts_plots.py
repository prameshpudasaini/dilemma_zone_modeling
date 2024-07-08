import os
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

# read node geometry data and select relevant columns
ndf = pd.read_csv("ignore/node_geometry.csv")[['Node', 'Approach', 'Speed_limit']]

# merge node geometry data
FTS = pd.merge(FTS, ndf, on = ['Node', 'Approach'], how = 'left')
YLR = pd.merge(YLR, ndf, on = ['Node', 'Approach'], how = 'left')

# round speed to 0 decimals to create speed bin
FTS['speed_bin'] = round(FTS.speed, 0)
YLR['speed_bin'] = round(YLR.speed, 0)

# add TTS: travel time to stop line variable
FTS['TSL'] = round(FTS.Xi / (FTS.speed * 5280/3600), 2)
YLR['TSL'] = round(YLR.Xi / (YLR.speed * 5280/3600), 2)

# =============================================================================
# sample counts
# =============================================================================

print(f"FTS by speed limit: \n{FTS.Speed_limit.value_counts()}")
print(f"YLR by speed limit: \n{YLR.Speed_limit.value_counts()}")

print(f"FTS by node: \n{FTS.Node.value_counts()}")
print(f"YLR by node: \n{YLR.Node.value_counts()}")

# get sample count by approach
nFTS = FTS[['Node', 'Approach']].value_counts().reset_index()
nYLR = YLR[['Node', 'Approach']].value_counts().reset_index()
nRLR = RLR[['Node', 'Approach']].value_counts().reset_index()

# add group for each sample count dataset
nFTS['Group'] = 'FTS'
nYLR['Group'] = 'YLR'
nRLR['Group'] = 'RLR'

# combine all sample count datasets
n_node = pd.concat([nFTS, nYLR, nRLR], ignore_index = True)
n_node.columns = ['Node', 'Approach', 'N', 'Group']

# pivot wider for each group and save file
n_node = n_node.pivot(index = ['Node', 'Approach'], columns = 'Group', values = 'N').reset_index()
n_node.to_csv("ignore/Wejo/trips_stop_go/node_counts_FTS_YLR_RLR.csv", index = False)

# =============================================================================
# histogram of yellow onset speed and distance by speed limit
# =============================================================================

px.histogram(
    FTS,
    x = 'speed_bin',
    facet_col = 'Speed_limit',
    nbins = 30,
    title = 'Histogram of yellow onset speed for FTS'
).show()

px.histogram(
    YLR,
    x = 'speed_bin',
    facet_col = 'Speed_limit',
    nbins = 30,
    title = 'Histogram of yellow onset speed for YLR'
).show()

px.histogram(
    FTS,
    x = 'Xi',
    facet_col = 'Speed_limit',
    nbins = 30,
    title = 'Histogram of yellow onset distance for FTS'
).show()

px.histogram(
    YLR,
    x = 'Xi',
    facet_col = 'Speed_limit',
    nbins = 30,
    title = 'Histogram of yellow onset distance for YLR'
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
# test: model fit and validation sites
# =============================================================================

FTS['Site'] = FTS.Approach.apply(lambda x: 'Fit' if x in ['EB', 'NB'] else 'Validation')
YLR['Site'] = YLR.Approach.apply(lambda x: 'Fit' if x in ['EB', 'NB'] else 'Validation')

FTS['Xs'] = FTS.groupby('speed_bin')['Xi'].transform(min)
YLR['Xc'] = YLR.groupby('speed_bin')['Xi'].transform(max)

FTS.Site.value_counts()
YLR.Site.value_counts()

px.scatter(
    FTS,
    x = 'Xi',
    y = 'speed',
    facet_col = 'Site'
).update_traces(marker = dict(size = 7, symbol = 'circle')).show()

px.scatter(
    YLR,
    x = 'Xi',
    y = 'speed',
    facet_col = 'Site'
).update_traces(marker = dict(size = 7, symbol = 'cross')).show()

# count number of samples in each speed bin by site
n_speed_bin = FTS[['Site', 'speed_bin']].value_counts().reset_index(name = 'N')
