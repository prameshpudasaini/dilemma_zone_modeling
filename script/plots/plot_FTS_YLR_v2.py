import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

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

# function to perform quantile regression
def quantile_regression(xdf, bin_size, q, group, return_data = False):
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
    
    predictors = x_FTS if group == 'FTS' else x_YLR
    X = df_X[predictors]
    y = df_X['Xi']
    
    qmodel = sm.QuantReg(y, X).fit(q = q)
    beta = qmodel.summary2().tables[1]['Coef.'].head(3).to_list()
    print(qmodel.summary())
    
    if return_data == True:
        return {'sdf': sdf, 'df_X': df_X, 'beta': beta}

# function to plot Xs/Xc and regression curves
def performance_evaluation(xdf, bin_size, q, group, site):
    qmodel = quantile_regression(xdf, bin_size, q, group, return_data = True)
    sdf, df_X, beta = qmodel['sdf'], qmodel['df_X'], qmodel['beta']
    
    if group == 'FTS':
        df_X['X'] = beta[0]*df_X['speed_fps'] + beta[1]*df_X['speed_sq'] + beta[2]*df_X['Crossing_length']
    elif group == 'YLR':
        df_X['X'] = beta[0]*df_X['speed_fps'] + beta[1]*df_X['PSL_fps'] + beta[2]*df_X['Crossing_length']
        
    site_df = sdf.copy()[sdf.SiteID == site]
    site_df_X = df_X.copy()[df_X.SiteID == site]
    
    PSL_fps = site_df.PSL_fps.values[0]
    Crossing_length = site_df.Crossing_length.values[0]
    
    # create dataset for fitting quantile regression
    y = np.linspace(site_df.speed_fps.min(), site_df.speed_fps.max(), 100) # speed values in fps
    if group == 'FTS':
        x = pd.DataFrame({'speed_fps': y, 'speed_sq': y**2, 'Crossing_length': Crossing_length})
        X = beta[0]*x['speed_fps'] + beta[1]*x['speed_sq'] + beta[2]*x['Crossing_length']
    elif group == 'YLR':
        x = pd.DataFrame({'speed_fps': y, 'PSL_fps': PSL_fps, 'Crossing_length': Crossing_length})
        X = beta[0]*x['speed_fps'] + beta[1]*x['PSL_fps'] + beta[2]*x['Crossing_length']
    
    y = np.round(y*3600/5280, decimals = 1) # fps-mph conversion
        
    return {'site_df': site_df, 'site_df_X': site_df_X, 'X': X, 'y': y}
    
    
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

# =============================================================================
# plot FTS & Xs
# =============================================================================

FTS_1EB = performance_evaluation(FTS, 25, 0.5, 'FTS', '1EB')
FTS_2WB = performance_evaluation(FTS, 25, 0.5, 'FTS', '2WB')
FTS_3NB = performance_evaluation(FTS, 25, 0.5, 'FTS', '3NB')
FTS_4SB = performance_evaluation(FTS, 25, 0.5, 'FTS', '4SB')

# create 4 subplots of Xi vs speed for FTS and YLR
fig, axes = plt.subplots(2, 2, figsize = (12, 10))
ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

site_df = FTS_1EB['site_df']
site_df_X = FTS_1EB['site_df_X']
for v in site_df_X['speed']:
    ax1.axhline(y=v, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
ax1.scatter(site_df['Xi'], site_df['speed'], color = 'black', marker = 'o', facecolors = 'none', alpha = 0.6, label = 'FTS vehicle')
ax1.scatter(site_df_X['Xi'], site_df_X['speed'], color = 'black', marker = 'o', label = 'Observed Xs for different speed')
ax1.set_title('(a) Site: 1EB', fontsize = 14, fontweight = 'bold')
ax1.set_xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
ax1.set_ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)
# ax1.legend(loc = 'lower right')

site_df = FTS_2WB['site_df']
site_df_X = FTS_2WB['site_df_X']
for v in site_df_X['speed']:
    ax2.axhline(y=v, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
ax2.scatter(site_df['Xi'], site_df['speed'], color = 'black', marker = 'o', facecolors = 'none', alpha = 0.6, label = 'FTS vehicle')
ax2.scatter(site_df_X['Xi'], site_df_X['speed'], color = 'black', marker = 'o', label = 'Observed Xs for different speed')
ax2.set_title('(b) Site: 2WB', fontsize = 14, fontweight = 'bold')
ax2.set_xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
ax2.set_ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)
# ax2.legend(loc = 'lower right')

site_df = FTS_3NB['site_df']
site_df_X = FTS_3NB['site_df_X']
for v in site_df_X['speed']:
    ax3.axhline(y=v, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
ax3.scatter(site_df['Xi'], site_df['speed'], color = 'black', marker = 'o', facecolors = 'none', alpha = 0.6, label = 'FTS vehicle')
ax3.scatter(site_df_X['Xi'], site_df_X['speed'], color = 'black', marker = 'o', label = 'Observed Xs for different speed')
ax3.set_title('(c) Site: 3NB', fontsize = 14, fontweight = 'bold')
ax3.set_xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
ax3.set_ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
ax3.tick_params(axis = 'both', which = 'major', labelsize = 12)
# ax3.legend(loc = 'lower right')

site_df = FTS_4SB['site_df']
site_df_X = FTS_4SB['site_df_X']
for v in site_df_X['speed']:
    ax4.axhline(y=v, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
ax4.scatter(site_df['Xi'], site_df['speed'], color = 'black', marker = 'o', facecolors = 'none', alpha = 0.6, label = 'FTS vehicle')
ax4.scatter(site_df_X['Xi'], site_df_X['speed'], color = 'black', marker = 'o', label = 'Observed Xs for different speed')
ax4.set_title('(d) Site: 4SB', fontsize = 14, fontweight = 'bold')
ax4.set_xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
ax4.set_ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
ax4.tick_params(axis = 'both', which = 'major', labelsize = 12)
# ax4.legend(loc = 'lower right')

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc = 'lower center', bbox_to_anchor = (0.5, -0.05), ncol = 2, fontsize = 12)

plt.tight_layout(pad = 1)
plt.savefig('output/scatter_FTS_Xs_v2.png', bbox_inches = 'tight', dpi = 1200)
plt.show()

# =============================================================================
# plot YLR & Xc
# =============================================================================

YLR_1EB = performance_evaluation(YLR, 35, 0.5, 'YLR', '1EB')
YLR_2WB = performance_evaluation(YLR, 35, 0.5, 'YLR', '2WB')
YLR_3NB = performance_evaluation(YLR, 35, 0.5, 'YLR', '3NB')
YLR_4SB = performance_evaluation(YLR, 35, 0.5, 'YLR', '4SB')

# create 4 subplots of Xi vs speed for FTS and YLR
fig, axes = plt.subplots(2, 2, figsize = (12, 10))
ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

site_df = YLR_1EB['site_df']
site_df_X = YLR_1EB['site_df_X']
for v in site_df_X['speed']:
    ax1.axhline(y=v, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
ax1.scatter(site_df['Xi'], site_df['speed'], color = 'black', marker = 's', facecolors = 'none', alpha = 0.6, label = 'YLR vehicle')
ax1.scatter(site_df_X['Xi'], site_df_X['speed'], color = 'black', marker = 's', label = 'Observed Xc for different speed')
ax1.set_title('(a) Site: 1EB', fontsize = 14, fontweight = 'bold')
ax1.set_xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
ax1.set_ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)
# ax1.legend(loc = 'lower right')

site_df = YLR_2WB['site_df']
site_df_X = YLR_2WB['site_df_X']
for v in site_df_X['speed']:
    ax2.axhline(y=v, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
ax2.scatter(site_df['Xi'], site_df['speed'], color = 'black', marker = 's', facecolors = 'none', alpha = 0.6, label = 'YLR vehicle')
ax2.scatter(site_df_X['Xi'], site_df_X['speed'], color = 'black', marker = 's', label = 'Observed Xc for different speed')
ax2.set_title('(b) Site: 2WB', fontsize = 14, fontweight = 'bold')
ax2.set_xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
ax2.set_ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)
# ax2.legend(loc = 'lower right')

site_df = YLR_3NB['site_df']
site_df_X = YLR_3NB['site_df_X']
for v in site_df_X['speed']:
    ax3.axhline(y=v, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
ax3.scatter(site_df['Xi'], site_df['speed'], color = 'black', marker = 's', facecolors = 'none', alpha = 0.6, label = 'YLR vehicle')
ax3.scatter(site_df_X['Xi'], site_df_X['speed'], color = 'black', marker = 's', label = 'Observed Xc for different speed')
ax3.set_title('(c) Site: 3NB', fontsize = 14, fontweight = 'bold')
ax3.set_xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
ax3.set_ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
ax3.tick_params(axis = 'both', which = 'major', labelsize = 12)
# ax3.legend(loc = 'lower right')

site_df = YLR_4SB['site_df']
site_df_X = YLR_4SB['site_df_X']
for v in site_df_X['speed']:
    ax4.axhline(y=v, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
ax4.scatter(site_df['Xi'], site_df['speed'], color = 'black', marker = 's', facecolors = 'none', alpha = 0.6, label = 'YLR vehicle')
ax4.scatter(site_df_X['Xi'], site_df_X['speed'], color = 'black', marker = 's', label = 'Observed Xc for different speed')
ax4.set_title('(d) Site: 4SB', fontsize = 14, fontweight = 'bold')
ax4.set_xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
ax4.set_ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
ax4.tick_params(axis = 'both', which = 'major', labelsize = 12)
# ax4.legend(loc = 'lower right')

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc = 'lower center', bbox_to_anchor = (0.5, -0.05), ncol = 2, fontsize = 12)

plt.tight_layout(pad = 1)
plt.savefig('output/scatter_YLR_Xc_v2.png', bbox_inches = 'tight', dpi = 1200)
plt.show()


# =============================================================================
# test: plot Xs, Xc, and quantile regression curves
# =============================================================================

q15_FTS = quantile_regression(FTS, 25, 0.15, 'FTS', return_data = True)
q50_FTS = quantile_regression(FTS, 25, 0.5, 'FTS', return_data = True)

q50_YLR = quantile_regression(YLR, 35, 0.5, 'YLR', return_data = True)
q85_YLR = quantile_regression(YLR, 35, 0.85, 'YLR', return_data = True)

# px.scatter(q15_FTS['df_X'], x = 'Xi', y = 'speed')
# px.scatter(q50_FTS['df_X'], x = 'Xi', y = 'speed')

plt.figure(figsize = (8, 6))
plt.scatter(q50_FTS['df_X'].Xi, q50_FTS['df_X'].speed, color = 'blue', marker = 'o', label = 'Observed Xs')
plt.scatter(q50_YLR['df_X'].Xi, q50_YLR['df_X'].speed, color = 'green', marker = 's', label = 'Observed Xs')
plt.xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
plt.ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
legend = plt.legend(loc = 'lower right', ncol = 2, fontsize = 12)
plt.tight_layout(pad = 1)
plt.show()


# =============================================================================
# plot Xs, Xc, and quantile regression curves for study sites
# =============================================================================

# create 4 subplots of Xi vs speed for FTS and YLR
# fig, axes = plt.subplots(4, 1, figsize = (12, 20))
# ax1, ax2, ax3, ax4 = axes[0], axes[1], axes[2], axes[3]
fig, axes = plt.subplots(2, 2, figsize = (12, 10))
ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

# 1st site
site_df = FTS_1EB['site_df']
site_df_X = FTS_1EB['site_df_X']
X, y = FTS_1EB['X'], FTS_1EB['y']
ax1.scatter(site_df['Xi'], site_df['speed'], color = 'black', marker = 'o', facecolors = 'none', alpha = 0.4, label = 'FTS vehicle')
ax1.scatter(site_df_X['Xi'], site_df_X['speed'], color = 'black', marker = 'o', label = 'Observed Xs')
ax1.plot(X, y, color = 'red')

site_df = YLR_1EB['site_df']
site_df_X = YLR_1EB['site_df_X']
X, y = YLR_1EB['X'], YLR_1EB['y']
ax1.scatter(site_df['Xi'], site_df['speed'], color = 'black', marker = 's', facecolors = 'none', alpha = 0.4, label = 'YLR vehicle')
ax1.scatter(site_df_X['Xi'], site_df_X['speed'], color = 'black', marker = 's', label = 'Observed Xc')
ax1.plot(X, y, color = 'red', label = 'Type I DZ boundary')

ax1.set_title('(a) Site: 1EB', fontsize = 14, fontweight = 'bold')
ax1.set_xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
ax1.set_ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)

# 2nd site
site_df = FTS_2WB['site_df']
site_df_X = FTS_2WB['site_df_X']
X, y = FTS_2WB['X'], FTS_2WB['y']
ax2.scatter(site_df['Xi'], site_df['speed'], color = 'black', marker = 'o', facecolors = 'none', alpha = 0.4, label = 'FTS vehicle')
ax2.scatter(site_df_X['Xi'], site_df_X['speed'], color = 'black', marker = 'o', label = 'Observed Xs')
ax2.plot(X, y, color = 'red', label = 'Type I DZ boundary')

site_df = YLR_2WB['site_df']
site_df_X = YLR_2WB['site_df_X']
X, y = YLR_2WB['X'], YLR_2WB['y']
ax2.scatter(site_df['Xi'], site_df['speed'], color = 'black', marker = 's', facecolors = 'none', alpha = 0.4, label = 'YLR vehicle')
ax2.scatter(site_df_X['Xi'], site_df_X['speed'], color = 'black', marker = 's', label = 'Observed Xc')
ax2.plot(X, y, color = 'red', label = 'Type I DZ boundary')

ax2.set_title('(b) Site: 2WB', fontsize = 14, fontweight = 'bold')
ax2.set_xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
ax2.set_ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)

# 3rd site
site_df = FTS_3NB['site_df']
site_df_X = FTS_3NB['site_df_X']
X, y = FTS_3NB['X'], FTS_3NB['y']
ax3.scatter(site_df['Xi'], site_df['speed'], color = 'black', marker = 'o', facecolors = 'none', alpha = 0.4, label = 'FTS vehicle')
ax3.scatter(site_df_X['Xi'], site_df_X['speed'], color = 'black', marker = 'o', label = 'Observed Xs')
ax3.plot(X, y, color = 'red', label = 'Type I DZ boundary')

site_df = YLR_3NB['site_df']
site_df_X = YLR_3NB['site_df_X']
X, y = YLR_3NB['X'], YLR_3NB['y']
ax3.scatter(site_df['Xi'], site_df['speed'], color = 'black', marker = 's', facecolors = 'none', alpha = 0.4, label = 'YLR vehicle')
ax3.scatter(site_df_X['Xi'], site_df_X['speed'], color = 'black', marker = 's', label = 'Observed Xc')
ax3.plot(X, y, color = 'red', label = 'Type I DZ boundary')

ax3.set_title('(c) Site: 3NB', fontsize = 14, fontweight = 'bold')
ax3.set_xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
ax3.set_ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
ax3.tick_params(axis = 'both', which = 'major', labelsize = 12)

# 4th site
site_df = FTS_4SB['site_df']
site_df_X = FTS_4SB['site_df_X']
X, y = FTS_4SB['X'], FTS_4SB['y']
ax4.scatter(site_df['Xi'], site_df['speed'], color = 'black', marker = 'o', facecolors = 'none', alpha = 0.4, label = 'FTS vehicle')
ax4.scatter(site_df_X['Xi'], site_df_X['speed'], color = 'black', marker = 'o', label = 'Observed Xs')
ax4.plot(X, y, color = 'red', label = 'Type I DZ boundary')

site_df = YLR_4SB['site_df']
site_df_X = YLR_4SB['site_df_X']
X, y = YLR_4SB['X'], YLR_4SB['y']
ax4.scatter(site_df['Xi'], site_df['speed'], color = 'black', marker = 's', facecolors = 'none', alpha = 0.4, label = 'YLR vehicle')
ax4.scatter(site_df_X['Xi'], site_df_X['speed'], color = 'black', marker = 's', label = 'Observed Xc')
ax4.plot(X, y, color = 'red', label = 'Type I DZ boundary')

ax4.set_title('(d) Site: 4SB', fontsize = 14, fontweight = 'bold')
ax4.set_xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
ax4.set_ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
ax4.tick_params(axis = 'both', which = 'major', labelsize = 12)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=12)

plt.tight_layout(pad = 1)
plt.savefig('output/plot_FTS_YLR_Xs_Xc.png', bbox_inches = 'tight', dpi = 1200)
plt.show()
