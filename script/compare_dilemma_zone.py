import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

os.chdir(r"D:\GitHub\dilemma_zone_modeling")

# function to compute Type II DZ based on probability of stopping
def compute_TypeII_DZ(xdf):
    # get distances and corresponding stop/go decisions
    distances, stops = xdf.Xi.to_numpy(), xdf.stops.to_numpy()
    
    # scale the distance feature for logistic regression
    scaler = StandardScaler()
    distances_scaled = scaler.fit_transform(distances.reshape(-1, 1))
    
    # fit logistic regression model
    log_reg = LogisticRegression()
    log_reg.fit(distances_scaled, stops)
    
    # define probability thresholds and compute corresponding log odds
    probs = np.array([0.1, 0.9])
    log_odds = np.log(probs / (1 - probs))
    
    # find scaled distances for probs threshold using model coefficients
    scaled_dist_10_90 = (log_odds - log_reg.intercept_) / log_reg.coef_[0]
    
    # unscale distances to get original values
    dist_10_90 = scaler.inverse_transform(scaled_dist_10_90.reshape(-1, 1))
    
    DZ_end = round(dist_10_90[0][0], 1)
    DZ_start = round(dist_10_90[1][0], 1)
    
    return {'start': DZ_start, 'end': DZ_end}

# read model datasets
FTS = pd.read_csv("ignore/Wejo/trips_analysis/trips_FTS.txt", sep = '\t')
YLR = pd.read_csv("ignore/Wejo/trips_analysis/trips_YLR.txt", sep = '\t')

# add stops as a binary decision variable
FTS['stops'] = 1
YLR['stops'] = 0

# combine two datasets and filter for sites 517 & 618 (speed limit of 40 mph)
df = pd.concat([FTS, YLR], ignore_index = True)
df = df[df.SiteID.str.contains('517|618', regex = True)]

Xs, Xc = {}, {}
N = 200

# array of speed values in mph
y = np.linspace(df.speed.min()*5280/3600, df.speed.max()*5280/3600, N)

# Type I DZ based on quantile regression
mean_cross_length = np.mean(list(df.Crossing_length.unique()))
mean_PSL = 40*5280/3600 # mph to fps
beta_Xs, beta_Xc = [1.0126, 0.0359, -0.1563], [3.4939, 0.7138, -0.1941]
Xs['QR'] = beta_Xs[0]*y + beta_Xs[1]*y**2 + beta_Xs[2]*mean_cross_length
Xc['QR'] = beta_Xc[0]*y + beta_Xc[1]*mean_PSL + beta_Xc[2]*mean_cross_length

# Type I DZ based on ITE guidelines
prt, dec_max, yellow_int = 1, 10, 4
Xs['ITE'] = prt * y + (1/(2*dec_max))*y**2
Xc['ITE'] = yellow_int * y

# Type II DZ based on travel time to stop
Xs['TT'] = 2.5*y
Xc['TT'] = 5.5*y

y = np.round(y*3600/5280, decimals = 1) # fps-mph conversion

# Type II DZ based on probability of stopping
prob_DZ = compute_TypeII_DZ(df)
Xs['Prob'] = np.full(N, prob_DZ['end'])
Xc['Prob'] = np.full(N, prob_DZ['start'])

plt.figure(figsize = (12, 8))
lw, f = 2, 12

plt.plot(Xc['QR'], y, color = 'blue', linewidth = lw, label = 'Type I DZ based on proposed method')
plt.plot(Xs['QR'], y, color = 'blue', linewidth = lw, linestyle = '--')
plt.plot(Xc['ITE'], y, color = 'red', linewidth = lw, label = 'Type I DZ based on ITE parameters')
plt.plot(Xs['ITE'], y, color = 'red', linewidth = lw, linestyle = '--')
plt.plot(Xc['TT'], y, color = 'green', linewidth = lw, label = 'Type II DZ based on travel time to stop')
plt.plot(Xs['TT'], y, color = 'green', linewidth = lw, linestyle = '--')
plt.plot(Xc['Prob'], y, color = 'magenta', linewidth = lw, label = 'Type II DZ based on stopping probability')
plt.plot(Xs['Prob'], y, color = 'magenta', linewidth = lw, linestyle = '--')

plt.plot([350, 375], [27, 27], color = 'black', linewidth = lw)
plt.plot([350, 375], [23.7, 23.7], color = 'black', linewidth = lw, linestyle = '--')
plt.text(0.6, 0.30, 'Start of dilemma zone', fontsize = f, ha = 'left', va = 'center', transform = plt.gcf().transFigure)
plt.text(0.6, 0.25, 'End of dilemma zone', fontsize = f, ha = 'left', va = 'center', transform = plt.gcf().transFigure)

plt.xlabel('Yellow onset distance from stop line (ft)', fontsize = f, fontweight = 'bold')
plt.ylabel('Yellow onset speed (mph)', fontsize = f, fontweight = 'bold')
plt.xticks(fontsize = f)
plt.yticks(fontsize = f)
legend = plt.legend(loc = 'lower center', bbox_to_anchor = (0.5, -0.2), ncol = 2, fontsize = f)
plt.tight_layout(pad = 1)
plt.savefig('output/DZ_boundary_comparison.png', bbox_inches = 'tight', dpi = 1200)
plt.show()
