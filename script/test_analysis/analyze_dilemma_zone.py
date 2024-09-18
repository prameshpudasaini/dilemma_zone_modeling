import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

os.chdir(r"D:\GitHub\dilemma_Wejo")

# load DZ analysis dataset
ddf = pd.read_csv("ignore/Wejo/trips_stop_go/data_Xs_Xc.txt", sep = '\t')

# compute zone vehicle's position is in
ddf.loc[(((ddf.Xi <= ddf.Xc) & (ddf.Xc <= ddf.Xs)) | ((ddf.Xi <= ddf.Xs) & (ddf.Xs <= ddf.Xc))), 'zone'] = 'Should go'
ddf.loc[(((ddf.Xi >= ddf.Xc) & (ddf.Xc >= ddf.Xs)) | ((ddf.Xi >= ddf.Xs) & (ddf.Xs >= ddf.Xc))), 'zone'] = 'Should stop'
ddf.loc[((ddf.Xc < ddf.Xi) & (ddf.Xi < ddf.Xs)), 'zone'] = 'Dilemma'
ddf.loc[((ddf.Xs < ddf.Xi) & (ddf.Xi < ddf.Xc)), 'zone'] = 'Option'

# # save file
# ddf.to_csv("ignore/Wejo/trips_stop_go/data_DZ_analysis.txt", sep = '\t', index = False)

# print(f"{ddf.zone.value_counts()}")
# print(f"{ddf.groupby('zone')['Decision'].value_counts()}")
# print(f"{ddf.groupby('zone')['Group'].value_counts()}")
# print(f"{ddf.groupby(['Node', 'zone'])['Decision'].value_counts()}")

# =============================================================================
# Type I decision zone vs. actual decision taken
# =============================================================================

zone_colors = {'Should go': 'green', 'Should stop': 'black', 'Option': 'blue'}
ddf['zone_color'] = ddf.zone.map(zone_colors)

ddf0 = ddf.copy()[ddf.Decision == 0]
ddf1 = ddf.copy()[ddf.Decision == 1]

ddf1_YLR = ddf1.copy()[ddf1.Group == 'YLR']
ddf1_RLR = ddf1.copy()[ddf1.Group == 'RLR']

# create subplots of Xi vs speed for FTS and YLR
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))

ax1.scatter(ddf0.Xi, ddf0.speed_mph, color = ddf0.zone_color, marker = 'o', facecolors = 'none', alpha = 0.7)
ax1.set_title('(a) Actual decision taken: stop', fontsize = 14, fontweight = 'bold')
ax1.set_xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
ax1.set_ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)

ax2.scatter(ddf1_YLR.Xi, ddf1_YLR.speed_mph, color = ddf1_YLR.zone_color, marker = 'o', facecolors = 'none', alpha = 0.7)
ax2.scatter(ddf1_RLR.Xi, ddf1_RLR.speed_mph, color = ddf1_RLR.zone_color, marker = '*', alpha = 0.5, label = 'Red light runner')
ax2.set_title('(b) Actual decision taken: go', fontsize = 14, fontweight = 'bold')
ax2.set_xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
ax2.set_ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax2.legend(loc = 'upper right')

plt.tight_layout()
# plt.savefig('output/type_I_actual_decision.png', dpi = 600)
plt.show()


# =============================================================================
# plot dilemma zone comparison: Type I vs Type II
# =============================================================================

ddf.Decision = ddf.Decision.apply(lambda x: 'Stop' if x == 0 else 'Go')

# array of speed values in mph
y = np.linspace(ddf.speed_mph.min(), ddf.speed_mph.max(), 100)

# Type I DZ parameters (Xs and Xc) modeled using quantile regression
Xs_1 = 0.9263 * (y*5280/3600) + 0.025 * ((y*5280/3600)**2)
Xc_1 = 43.5281 + 2.8345 * (y*5280/3600)

# Type II DZ parameters
Xs_2 = y * 5280/3600 * 2.5
Xc_2 = y * 5280/3600 * 5.5

zone_colors = {'Should go': 'green', 'Should stop': 'black', 'Option': 'blue'}
group_markers = {'FTS': 'o', 'YLR': 's', 'RLR': '*'}

bold_font = font_manager.FontProperties(weight = 'bold', size = 12)

plt.figure(figsize = (12, 8))

for (cat1, cat2), group in ddf.groupby(['zone', 'Group']):
    plt.scatter(group['Xi'], group['speed_mph'], label = f'{cat1}-{cat2}',
                color = zone_colors[cat1], marker = group_markers[cat2], 
                facecolors = 'none', alpha = 0.6, s = 20)

# legend for zone colors
color_patches = [plt.Line2D([0], [0], marker = 'o', color = 'w', label = key, markersize = 10, markerfacecolor = color)
                 for key, color in zone_colors.items()]
first_legend = plt.legend(handles = color_patches, title = 'Type I decision zone', fontsize = 12, 
                          title_fontproperties = bold_font, loc = 'lower right', ncol = 3)
plt.gca().add_artist(first_legend)

# legend for group markers
marker_patches = [plt.Line2D([0], [0], marker = marker, color = 'k', label = key, linestyle = 'None')
                  for key, marker in group_markers.items()]
plt.legend(handles = marker_patches, title = 'Actual decision taken', fontsize = 12,
           title_fontproperties = bold_font, loc = 'upper left', ncol = 3)

plt.plot(Xs_1, y, color = 'red', linewidth = 2, label = 'Option zone (Type I)')
plt.plot(Xc_1, y, color = 'red', linewidth = 2, label = 'Option zone (Type I)')
plt.plot(Xs_2, y, color = 'm', linewidth = 2, label = 'Travel time-based dilemma zone (Type II)')
plt.plot(Xc_2, y, color = 'm', linewidth = 2, label = 'Travel time-based dilemma zone (Type II)')

plt.xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
plt.ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

plt.tight_layout(pad = 1)
plt.savefig('output/DZ_comparison_TypeI_TypeII.png', dpi = 600)
plt.show()

# =============================================================================
# count analysis
# =============================================================================

# count and percentage by zone and actual decision taken
cdf = ddf.groupby('zone').Decision.value_counts().reset_index(name = 'total')
cdf.Decision = cdf.Decision.apply(lambda x: 'Stop' if x == 0 else 'Go')
zone_total = cdf.groupby('zone').total.sum().reset_index(name = 'zone_total')
cdf = pd.merge(cdf, zone_total, on = 'zone')
cdf['percent'] = round(cdf.total / cdf.zone_total * 100, 1)
cdf.drop('zone_total', axis = 1, inplace = True)
cdf = cdf.pivot(index = 'Decision', columns = 'zone', values = ['total', 'percent'])
cdf.reset_index(inplace = True)

# count and percentage by zone for FTS, YLR, RLR decisions
cdf = ddf.groupby('zone').Group.value_counts().reset_index(name = 'total')
zone_total = cdf.groupby('zone').total.sum().reset_index(name = 'zone_total')
cdf = pd.merge(cdf, zone_total, on = 'zone')
cdf['percent'] = round(cdf.total / cdf.zone_total * 100, 1)
cdf.drop('zone_total', axis = 1, inplace = True)
cdf = cdf.pivot(index = 'Group', columns = 'zone', values = ['total', 'percent'])
cdf.reset_index(inplace = True)

# count and percentage by zone and node for FTS, YLR, RLR decisions
cdf = ddf.groupby(['Node', 'zone']).Group.value_counts().reset_index(name = 'total')
node_zone_total = cdf.groupby(['Node', 'zone']).total.sum().reset_index(name = 'node_zone_total')
cdf = pd.merge(cdf, node_zone_total, on = ['Node', 'zone'])
cdf['percent'] = round(cdf.total / cdf.node_zone_total, 3)
cdf.to_csv("ignore/Wejo/trips_stop_go/count_node_zone_group.txt", sep = '\t', index = False)

# count by zone and node
cdf = ddf.groupby(['Node', 'zone'])['Decision'].value_counts().reset_index(name = 'N')
cdf = cdf.pivot(index = ['zone', 'Decision'], columns = 'Node', values = 'N')
cdf.reset_index(inplace = True)


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
        
    v85 = v85*5280/3600

    prt = 0.274 + 30.392/v0
    dec = np.exp(3.572 - 25.013/v0) - 17.855 + 480.558/v85
    acc = -23.513 + 658.948/v0 + 0.223*v85
    
    Xs = round(v0*prt + ((v0**2) / (2*dec)), 0)
    Xc = round(v0*y + 0.5*(acc)*((y - prt)**2), 0)
    
    return {'Xs': Xs, 'Xc': Xc}

ddf['Xs_Li'] = ddf.apply(lambda x: modelLi2013(x.speed, x.Node)['Xs'], axis = 1)
ddf['Xc_Li'] = ddf.apply(lambda x: modelLi2013(x.speed, x.Node)['Xc'], axis = 1)

corr_Xs = round(ddf.Xs.corr(ddf.Xs_Li), 4)
corr_Xc = round(ddf.Xc.corr(ddf.Xc_Li), 4)
print(f"Correlation Xs and Xs_Li: {corr_Xs}")
print(f"Correlation Xc and Xc_Li: {corr_Xc}")

# # create subplots of Xi vs speed for FTS and YLR
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))

# ax1.scatter(ddf.Xs, ddf.Xs_Li, color = 'black', alpha = 0.8)
# ax1.set_title('(a) Minimum stopping distance', fontsize = 14, fontweight = 'bold')
# ax1.set_xlabel('$X_s$ from qunatile regression model', fontsize = 12, fontweight = 'bold')
# ax1.set_ylabel('$X_s$ from Li & Wei (2013) model', fontsize = 12, fontweight = 'bold')
# ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)

# ax2.scatter(ddf.Xc, ddf.Xc_Li, color = 'black', alpha = 0.8)
# ax2.set_title('(a) Maximum clearing distance', fontsize = 14, fontweight = 'bold')
# ax2.set_xlabel('$X_c$ from qunatile regression model', fontsize = 12, fontweight = 'bold')
# ax2.set_ylabel('$X_c$ from Li & Wei (2013) model', fontsize = 12, fontweight = 'bold')
# ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)

# plt.tight_layout(pad = 1)
# plt.show()
