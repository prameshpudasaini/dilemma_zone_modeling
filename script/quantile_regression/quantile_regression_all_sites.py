import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

os.chdir(r"/Users/prameshpudasaini/Library/CloudStorage/OneDrive-UniversityofArizona/GitHub/dilemma_zone_modeling")

# read model dataset
FTS = pd.read_csv("ignore/Wejo/trips_analysis/trips_FTS.txt", sep = '\t')

# add speed variables
FTS['speed_fps'] = round(FTS.speed*5280/3600, 1)
FTS['speed_sq'] = round(FTS.speed_fps**2, 1)

# select relevant columns for quantile regression
FTS = FTS[['SiteID', 'Xi', 'speed', 'speed_fps', 'speed_sq']]

def quantile_regression_model_Xs(sdf, site):
    # target and predictors
    X = sdf[['speed_fps', 'speed_sq']]
    y = sdf['Xi']
    
    # fit quantile regression models for two extreme percentiles
    qmodel1 = sm.QuantReg(y, X).fit(q = 0.001) # 0.1th percentile
    qmodel2 = sm.QuantReg(y, X).fit(q = 0.01) # 1st percentile
    qmodel3 = sm.QuantReg(y, X).fit(q = 0.05) # 5th percentile
    
    # create x, y values for fitting quantile regression line
    y_val = np.linspace(sdf.speed_fps.min(), sdf.speed_fps.max(), 100) # y values in fps
    x_val = pd.DataFrame({'speed': y_val, 'speed_sq': y_val**2})
    y_val = np.round(y_val*3600/5280, decimals = 1) # fps-mph conversion
    
    # predict Xs values from quantile regression models
    Xs1 = qmodel1.predict(x_val)
    Xs2 = qmodel2.predict(x_val)
    Xs3 = qmodel3.predict(x_val)
    
    # plot FTS scatter points and quantile regression curves
    plt.figure(figsize = (12, 8))
    
    plt.scatter(sdf['Xi'], sdf['speed'], color = 'black', marker = 'o', facecolors = 'none', alpha = 0.6, label = 'FTS vehicles')
    plt.plot(Xs1, y_val, color = 'red', label = 'tau = 0.1')
    plt.plot(Xs2, y_val, color = 'blue', label = 'tau = 1')
    plt.plot(Xs3, y_val, color = 'green', label = 'tau = 5')
    
    plt.title(f'FTS vehicles for site: {site}', fontsize = 14, fontweight = 'bold')
    plt.xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
    plt.ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    plt.legend(loc = 'lower right')
    
    plt.tight_layout(pad = 1)
    plt.savefig(f'ignore/quantile_regression_plots/Xs_{site}.png', dpi = 600)
    plt.show()

# loop through all sites and plot quantile regression curves for Xs
for site in list(FTS.SiteID.unique()):
    sdf = FTS.copy()[FTS.SiteID == site]
    quantile_regression_model_Xs(sdf, site)

# quantile regression curves for full FTS dataset
quantile_regression_model_Xs(FTS, 'all')
