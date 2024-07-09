import os
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\dilemma_Wejo")

# read datasets with stop/go trips
GLR = pd.read_csv("ignore/Wejo/trips_stop_go/GLR_filtered.txt", sep = '\t') # GLR trips
FTS = pd.read_csv("ignore/Wejo/trips_stop_go/FTS_filtered.txt", sep = '\t') # FTS trips
YLR = pd.read_csv("ignore/Wejo/trips_stop_go/YLR_filtered.txt", sep = '\t') # YLR trips
RLR = pd.read_csv("ignore/Wejo/trips_stop_go/RLR_filtered.txt", sep = '\t') # RLR trips

# combine GLR with FTS data
FTS = pd.concat([FTS, GLR], ignore_index = True)

# adaptive binning using quantiles
num_bins = 30
FTS['speed_bin'] = pd.qcut(FTS['speed'], q = num_bins, duplicates = 'drop')
YLR['speed_bin'] = pd.qcut(YLR['speed'], q = num_bins, duplicates = 'drop')

bin_count_fts = FTS.speed_bin.value_counts().reset_index(name = 'sample_count')
bin_count_ylr = YLR.speed_bin.value_counts().reset_index(name = 'sample_count')

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

# convert speed bin keys and Xs/Xc values to a dictionary
dict_Xs = dict(zip(fdf['speed_bin'], fdf['Xs']))
dict_Xc = dict(zip(ydf['speed_bin'], ydf['Xc']))

# function to return Xs for given speed
def get_Xs_for_speed(speed):
    for interval, Xs in dict_Xs.items():
        if interval.left < speed <= interval.right:
            return Xs
    return None

# function to return Xc for given speed
def get_Xc_for_speed(speed):
    for interval, Xc in dict_Xc.items():
        if interval.left < speed <= interval.right:
            return Xc
    return None

# compute maximum clearing distance for FTS arrivals    
FTS['Xc'] = FTS.speed.apply(get_Xc_for_speed)

# compute minimum stopping distance for YLR arrivals
YLR['Xs'] = YLR.speed.apply(get_Xs_for_speed)

# compute Xs and Xc for RLR arrivals
RLR['Xc'] = RLR.speed.apply(get_Xc_for_speed)
RLR['Xs'] = RLR.speed.apply(get_Xs_for_speed)

# add stop/go decision to dataset
FTS['Decision'] = 0
YLR['Decision'] = 1
RLR['Decision'] = 1

# combine both datasets
cols_combine = ['Node', 'Approach', 'localtime', 'speed', 'Xi', 'Xs', 'Xc', 'Decision']
ddf = pd.concat([FTS[cols_combine], YLR[cols_combine], RLR[cols_combine]], ignore_index = True)

# compute zone vehicle's position is in
ddf.loc[(((ddf.Xi <= ddf.Xc) & (ddf.Xc <= ddf.Xs)) | ((ddf.Xi <= ddf.Xs) & (ddf.Xs <= ddf.Xc))), 'zone'] = 'Should go'
ddf.loc[(((ddf.Xi >= ddf.Xc) & (ddf.Xc >= ddf.Xs)) | ((ddf.Xi >= ddf.Xs) & (ddf.Xs >= ddf.Xc))), 'zone'] = 'Should stop'
ddf.loc[((ddf.Xc < ddf.Xi) & (ddf.Xi < ddf.Xs)), 'zone'] = 'Dilemma'
ddf.loc[((ddf.Xs < ddf.Xi) & (ddf.Xi < ddf.Xc)), 'zone'] = 'Option'

print(f"{ddf.zone.value_counts()}")
print(f"{ddf.groupby('zone')['Decision'].value_counts()}")
print(f"{ddf.groupby(['Node', 'zone'])['Decision'].value_counts()}")

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
