import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

# function to compute Xs and Xc from quantile regression
beta_Xs, beta_Xc = [1.0126, 0.0359, -0.1563], [3.4939, 0.7138, -0.1941]
def compute_Xs_Xc(xdf):
    xdf['Xs'] = beta_Xs[0]*xdf['speed_fps'] + beta_Xs[1]*xdf['speed_sq'] + beta_Xs[2]*xdf['Crossing_length']
    xdf['Xc'] = beta_Xc[0]*xdf['speed_fps'] + beta_Xc[1]*xdf['PSL_fps'] + beta_Xc[2]*xdf['Crossing_length']
    xdf = xdf[['SiteID', 'speed', 'speed_fps', 'speed_sq', 'Crossing_length', 'Xi', 'Xs', 'Xc']]
    return xdf


# =============================================================================
# read and prepare datasets
# =============================================================================

# read model datasets
FTS = pd.read_csv("ignore/Wejo/trips_analysis/trips_FTS.txt", sep = '\t')
YLR = pd.read_csv("ignore/Wejo/trips_analysis/trips_YLR.txt", sep = '\t')
RLR = pd.read_csv("ignore/Wejo/trips_analysis/trips_RLR.txt", sep = '\t')

# map siteIDs
FTS['NodeID'] = FTS.Node.map({216: 1, 217: 2, 517: 3, 618: 4})
YLR['NodeID'] = YLR.Node.map({216: 1, 217: 2, 517: 3, 618: 4})
RLR['NodeID'] = RLR.Node.map({216: 1, 217: 2, 517: 3, 618: 4})

FTS.SiteID = FTS.NodeID.astype(str) + FTS.Approach
YLR.SiteID = YLR.NodeID.astype(str) + YLR.Approach
RLR.SiteID = RLR.NodeID.astype(str) + YLR.Approach

# process predictors
FTS = process_predictors(FTS)
YLR = process_predictors(YLR)
RLR = process_predictors(RLR)

# compute Xs and Xc
FTS = compute_Xs_Xc(FTS)
YLR = compute_Xs_Xc(YLR)
RLR = compute_Xs_Xc(RLR)

# combine FTS, YLR, RLR datasets
FTS['Group'] = 'FTS'
YLR['Group'] = 'YLR'
RLR['Group'] = 'RLR'
df = pd.concat([FTS, YLR, RLR], ignore_index = False)

df.loc[df.Group.isin(['FTS']), 'Decision'] = 'Stop'
df.loc[df.Group.isin(['YLR', 'RLR']), 'Decision'] = 'Go'

# compute the zone vehicle's position is in
df.loc[(((df.Xi <= df.Xc) & (df.Xc <= df.Xs)) | ((df.Xi <= df.Xs) & (df.Xs <= df.Xc))), 'zone'] = 'Should go'
df.loc[(((df.Xi >= df.Xc) & (df.Xc >= df.Xs)) | ((df.Xi >= df.Xs) & (df.Xs >= df.Xc))), 'zone'] = 'Should stop'
df.loc[((df.Xc < df.Xi) & (df.Xi < df.Xs)), 'zone'] = 'Dilemma'
df.loc[((df.Xs < df.Xi) & (df.Xi < df.Xc)), 'zone'] = 'Option'

# df.Group.value_counts()
# df.zone.value_counts()

# df.groupby(['Group']).zone.value_counts()
# df.groupby(['zone']).Group.value_counts()
# df.groupby(['Decision']).zone.value_counts()


# =============================================================================
# Type I decision zone vs. actual decision taken
# =============================================================================

zone_colors = {'Should go': 'green', 'Should stop': 'black', 'Option': 'blue', 'Dilemma': 'red'}
group_shapes = {'FTS': 'o', 'YLR': 's', 'RLR': '*'}
df['zone_color'] = df.zone.map(zone_colors)
df['group_shape'] = df.Group.map(group_shapes)

df0 = df.copy()[df.Decision == 'Stop']
df1 = df.copy()[df.Decision == 'Go']
df1_YLR = df1.copy()[df1.Group == 'YLR']
df1_RLR = df1.copy()[df1.Group == 'RLR']

legend_handles = [Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = color, 
                         markersize = 10, label = label) 
                  for label, color in zone_colors.items()]

# create subplots of Xi vs speed for FTS and YLR
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))
ss, s, a, ff, f = 10, 8, 0.5, 14, 12 # alpha, large font, normal font

ax1.scatter(df0.Xi, df0.speed, color = df0.zone_color, s = s, marker = 'o', facecolors = 'none', alpha = a)
ax1.set_title('(a) Actual decision taken: stop', fontsize = ff, fontweight = 'bold')
ax1.set_xlabel('Yellow onset distance from stop line (ft)', fontsize = f, fontweight = 'bold')
ax1.set_ylabel('Yellow onset speed (mph)', fontsize = f, fontweight = 'bold')
ax1.tick_params(axis = 'both', which = 'major', labelsize = f)

ax2.scatter(df1_YLR.Xi, df1_YLR.speed, color = df1_YLR.zone_color, s = s, marker = 's', facecolors = 'none', alpha = a)
ax2.scatter(df1_RLR.Xi, df1_RLR.speed, color = df1_RLR.zone_color, s = ss, marker = '*', alpha = a, label = 'Red light runner')
ax2.set_title('(b) Actual decision taken: go', fontsize = ff, fontweight = 'bold')
ax2.set_xlabel('Yellow onset distance from stop line (ft)', fontsize = f, fontweight = 'bold')
ax2.set_ylabel('Yellow onset speed (mph)', fontsize = f, fontweight = 'bold')
ax2.tick_params(axis = 'both', which = 'major', labelsize = f)
ax2.legend(loc = 'upper left', markerscale = 4, fontsize = f)

fig.legend(handles = legend_handles, loc = 'lower center', bbox_to_anchor = (0.6, -0.05), ncol = 4, fontsize = f)
fig.text(0.165, -0.02, 'Type I Decision Rules:', fontsize = f, fontweight = 'bold', transform = fig.transFigure)
plt.tight_layout(pad = 1)
plt.savefig('output/type_I_actual_decision.png', bbox_inches = 'tight', dpi = 1200)
plt.show()
