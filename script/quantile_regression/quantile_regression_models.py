import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
    
    marker = 'o' if group == 'FTS' else 's'
    plt.figure(figsize = (12, 8))
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

# map siteIDs
FTS['NodeID'] = FTS.Node.map({216: 1, 217: 2, 517: 3, 618: 4})
YLR['NodeID'] = YLR.Node.map({216: 1, 217: 2, 517: 3, 618: 4})
FTS.SiteID = FTS.NodeID.astype(str) + FTS.Approach
YLR.SiteID = YLR.NodeID.astype(str) + YLR.Approach

# px.scatter(FTS, x = 'Xi', y = 'speed').show()
# px.scatter(YLR, x = 'Xi', y = 'speed').show()

# process predictors
FTS = process_predictors(FTS)
YLR = process_predictors(YLR)

# =============================================================================
# quantile regression tests with different quantiles and bin sizes
# =============================================================================

x_FTS = ['speed_fps', 'speed_sq', 'Crossing_length', 'int_speed_peak', 'int_speed_night', 'int_speed_weekend']
x_YLR = ['speed_fps', 'PSL_fps', 'Crossing_length', 'int_speed_peak', 'int_speed_night', 'int_speed_weekend']

quantiles_FTS, quantiles_YLR = [0.05, 0.15, 0.5], [0.5, 0.85, 0.95]
bin_sizes = [20, 25, 30, 35]

def quantile_regression(xdf, bin_size, q, group, plot_data = False, return_data = False):
    list_site_df = []
    # loop through each site and identify Xs, Xc
    for site in list(xdf.SiteID.unique()):
        site_df = xdf.copy()[xdf.SiteID == site]
        num_bins = int(len(site_df) / bin_size)
        site_df = identify_Xs_Xc(site_df, num_bins, group)
        list_site_df.append(site_df)
    
    sdf = pd.concat(list_site_df, ignore_index = True) # combine data from all sites
    df_X = sdf.copy()[sdf.Xo == 1] # filter Xs or Xc observations
    
    if plot_data == True:
        plot_yellow_onset_speed_distance(df_X, group)
    
    df_X['constant'] = 1 # constant term for regression
    
    predictors = x_FTS if group == 'FTS' else x_YLR
    X = df_X[predictors]
    y = df_X['Xi']
    
    qmodel = sm.QuantReg(y, X).fit(q = q)
    beta = qmodel.summary2().tables[1]['Coef.'].head(3).to_list()
    print(qmodel.summary())
    
    if return_data == True:
        return {'sdf': sdf, 'df_X': df_X, 'beta': beta}
        
# quantile regression tests for FTS vehicles
for q in quantiles_FTS:
    for bin_size in bin_sizes:
        print(f"\nQuantile, bin size: {q}, {bin_size}")
        quantile_regression(FTS, bin_size, q, 'FTS')
        
# quantile regression tests for YLR vehicles
for q in quantiles_YLR:
    for bin_size in bin_sizes:
        print(f"\nQuantile, bin size: {q}, {bin_size}")
        quantile_regression(YLR, bin_size, q, 'YLR')
        
# Observations and conclusions
# Check Xs with q = 0.5, b = [20, 25]
# Check Xc with q = 0.5, b = [30, 35]


# =============================================================================
# evaluate estimation of Xs and Xc at each site
# =============================================================================

def performance_evaluation(xdf, bin_size, q, group):
    qmodel = quantile_regression(xdf, bin_size, q, group, plot_data = False, return_data = True)
    sdf, df_X, beta = qmodel['sdf'], qmodel['df_X'], qmodel['beta']
    
    if group == 'FTS':
        df_X['X'] = beta[0]*df_X['speed_fps'] + beta[1]*df_X['speed_sq'] + beta[2]*df_X['Crossing_length']
    elif group == 'YLR':
        df_X['X'] = beta[0]*df_X['speed_fps'] + beta[1]*df_X['PSL_fps'] + beta[2]*df_X['Crossing_length']
    
    error_list = []
    for site in list(sdf.SiteID.unique()):
        site_df = sdf.copy()[sdf.SiteID == site]
        site_df_X = df_X.copy()[df_X.SiteID == site]
        
        MAE = round(np.mean(abs(site_df_X['Xi'] - site_df_X['X'])), 2)
        RMSE = round(np.sqrt(np.mean((site_df_X['Xi'] - site_df_X['X'])**2)), 2)
        error_list.append({'Site': site, 'MAE': MAE, 'RMSE': RMSE, 'Group': group})
        print(f"MAE, RMSE for {site}: {MAE}, {RMSE}")
        
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
        
        plt.figure(figsize = (12, 8))
        marker = 'o' if group == 'FTS' else 's'
        # plot first-to-stop or crossing vehicles with xdf
        # plot Xs or Xc observations with ydf
        # plot quantile regression curve
        plt.scatter(site_df['Xi'], site_df['speed'], color = 'black', marker = marker, facecolors = 'none', alpha = 0.6)
        plt.scatter(site_df_X['Xi'], site_df_X['speed'], color = 'black', marker = marker)
        plt.plot(X, y, color = 'red', label = 'tau = 1')
        
        plt.xlabel('Yellow onset distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
        plt.ylabel('Yellow onset speed (mph)', fontsize = 12, fontweight = 'bold')
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.show()
        
    # performance dataset
    pdf = pd.DataFrame(error_list)
    return pdf

FTS_pdf_20 = performance_evaluation(FTS, 20, 0.5, 'FTS')
FTS_pdf_25 = performance_evaluation(FTS, 25, 0.5, 'FTS') # best results for Xs
YLR_pdf_30 = performance_evaluation(YLR, 30, 0.5, 'YLR')
YLR_pdf_35 = performance_evaluation(YLR, 35, 0.5, 'YLR') # best results for Xc

FTS_pdf_25_q15 = performance_evaluation(FTS, 25, 0.15, 'FTS') # Xs for 15th percentile
YLR_pdf_35_q85 = performance_evaluation(YLR, 35, 0.85, 'YLR') # Xc for 85th percentile

print(f"Average MAE, RMSE for bin size 20: {round(FTS_pdf_20.MAE.mean(), 2)}, {round(FTS_pdf_20.RMSE.mean(), 2)}")
print(f"Average MAE, RMSE for bin size 25: {round(FTS_pdf_25.MAE.mean(), 2)}, {round(FTS_pdf_25.RMSE.mean(), 2)}")
print(f"Average MAE, RMSE for bin size 30: {round(YLR_pdf_30.MAE.mean(), 2)}, {round(YLR_pdf_30.RMSE.mean(), 2)}")
print(f"Average MAE, RMSE for bin size 35: {round(YLR_pdf_35.MAE.mean(), 2)}, {round(YLR_pdf_35.RMSE.mean(), 2)}")

FTS_pdf_25['Group'] = 'Xs'
FTS_pdf_25['Method'] = 'QR'
YLR_pdf_35['Group'] = 'Xc'
YLR_pdf_35['Method'] = 'QR'
pdf0 = pd.concat([FTS_pdf_25, YLR_pdf_35], ignore_index = True)


# =============================================================================
# performance comparison with ITE
# =============================================================================

FTS_results = quantile_regression(FTS, 25, 0.5, 'FTS', plot_data = False, return_data = True)
YLR_results = quantile_regression(YLR, 35, 0.5, 'YLR', plot_data = False, return_data = True)

FTS_Xs = FTS_results['df_X']
YLR_Xc = YLR_results['df_X']

prt, dec_max = 1, 10
FTS_Xs['ITE_X'] = prt * FTS_Xs.speed_fps + (1/(2*dec_max))*FTS_Xs.speed_sq
YLR_Xc['ITE_X'] = YLR_Xc.speed_fps * 4 # yellow interval

def performance_comparison_ITE(xdf, group):
    error_list = []
    for site in list(xdf.SiteID.unique()):
        site_df_X = xdf.copy()[xdf.SiteID == site]
        
        MAE = round(np.mean(abs(site_df_X['Xi'] - site_df_X['ITE_X'])), 2)
        RMSE = round(np.sqrt(np.mean((site_df_X['Xi'] - site_df_X['ITE_X'])**2)), 2)
        error_list.append({'Site': site, 'MAE': MAE, 'RMSE': RMSE, 'Group': group})
        print(f"MAE, RMSE for {site}: {MAE}, {RMSE}")
        
    # performance dataset
    pdf = pd.DataFrame(error_list)
    return pdf

FTS_pdf_ITE = performance_comparison_ITE(FTS_Xs, 'FTS')
YLR_pdf_ITE = performance_comparison_ITE(YLR_Xc, 'YLR')

FTS_pdf_ITE['Group'] = 'Xs'
FTS_pdf_ITE['Method'] = 'ITE'
YLR_pdf_ITE['Group'] = 'Xc'
YLR_pdf_ITE['Method'] = 'ITE'
pdf1 = pd.concat([pdf0, FTS_pdf_ITE, YLR_pdf_ITE], ignore_index = True)


# =============================================================================
# performance comparison with Type II travel time-based method
# =============================================================================

FTS_Xs['TT_X'] = 2.5 * FTS_Xs.speed_fps
YLR_Xc['TT_X'] = 5.5 * YLR_Xc.speed_fps

def performance_comparison_TT(xdf, group):
    error_list = []
    for site in list(xdf.SiteID.unique()):
        site_df_X = xdf.copy()[xdf.SiteID == site]
        
        MAE = round(np.mean(abs(site_df_X['Xi'] - site_df_X['TT_X'])), 2)
        RMSE = round(np.sqrt(np.mean((site_df_X['Xi'] - site_df_X['TT_X'])**2)), 2)
        error_list.append({'Site': site, 'MAE': MAE, 'RMSE': RMSE, 'Group': group})
        print(f"MAE, RMSE for {site}: {MAE}, {RMSE}")
        
    # performance dataset
    pdf = pd.DataFrame(error_list)
    return pdf

FTS_pdf_TT = performance_comparison_TT(FTS_Xs, 'FTS')
YLR_pdf_TT = performance_comparison_TT(YLR_Xc, 'YLR')

FTS_pdf_TT['Group'] = 'Xs'
FTS_pdf_TT['Method'] = 'TT'
YLR_pdf_TT['Group'] = 'Xc'
YLR_pdf_TT['Method'] = 'TT'
pdf2 = pd.concat([pdf1, FTS_pdf_TT, YLR_pdf_TT], ignore_index = True)


# =============================================================================
# performance comparison with Type II probabilistic method
# =============================================================================

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

FTS = FTS_results['sdf']
YLR = YLR_results['sdf']

# add stops as a binary decision variable
FTS['stops'] = 1
YLR['stops'] = 0

# combine two datasets
prob_df = pd.concat([FTS, YLR], ignore_index = True)

# identify Type II DZ boundary for each site
site_prob_DZ = []
for site in list(prob_df.SiteID.unique()):
    site_df = prob_df.copy()[prob_df.SiteID == site]
    prob_DZ = compute_TypeII_DZ(site_df)
    site_prob_DZ.append([site, prob_DZ['start'], prob_DZ['end']])
    
site_prob_DZ = pd.DataFrame(site_prob_DZ, columns = ['SiteID', 'Xc', 'Xs'])

def performance_comparison_prob(xdf, group):
    error_list = []
    for site in list(xdf.SiteID.unique()):
        site_df = xdf.copy()[xdf.SiteID == site]
        prob_DZ = site_prob_DZ[site_prob_DZ.SiteID == site]
        
        if group == 'FTS':
            site_df['Prob_X'] = prob_DZ['Xs'].values[0]
        elif group == 'YLR':
            site_df['Prob_X'] = prob_DZ['Xc'].values[0]
        
        site_df_X = site_df[site_df.Xo == 1]
        
        MAE = round(np.mean(abs(site_df_X['Xi'] - site_df_X['Prob_X'])), 2)
        RMSE = round(np.sqrt(np.mean((site_df_X['Xi'] - site_df_X['Prob_X'])**2)), 2)
        error_list.append({'Site': site, 'MAE': MAE, 'RMSE': RMSE, 'Group': group})
        print(f"MAE, RMSE for {site}: {MAE}, {RMSE}")
        
    # performance dataset
    pdf = pd.DataFrame(error_list)
    return pdf

FTS_pdf_Prob = performance_comparison_prob(FTS, 'FTS')
YLR_pdf_Prob = performance_comparison_prob(YLR, 'YLR')

FTS_pdf_Prob['Group'] = 'Xs'
FTS_pdf_Prob['Method'] = 'Prob'
YLR_pdf_Prob['Group'] = 'Xc'
YLR_pdf_Prob['Method'] = 'Prob'
pdf3 = pd.concat([pdf2, FTS_pdf_Prob, YLR_pdf_Prob], ignore_index = True)


# =============================================================================
# accuracy evaluation across study sites
# =============================================================================

mean_mae = pdf3.groupby(['Group', 'Method'])['MAE'].mean().reset_index()
mean_rmse = pdf3.groupby(['Group', 'Method'])['RMSE'].mean().reset_index()

mean_mae_rmse = pd.merge(mean_mae, mean_rmse, on = ['Group', 'Method'], how = 'left')
mean_mae_rmse['Site'] = 'All Sites'

pdf4 = pd.concat([pdf3, mean_mae_rmse], ignore_index = True)

px.bar(pdf4, x = 'Site', y = 'MAE', color = 'Method', facet_col = 'Group', barmode = 'group')
px.bar(pdf4, x = 'Site', y = 'RMSE', color = 'Method', facet_row = 'Group', barmode = 'group')

pdf4.Group = pdf4.Group.map({'Xs': 'End of dilemma zone (Xs)', 'Xc': 'Start of dilemma zone (Xc)'})
pdf4.Method = pdf4.Method.map({
    'QR': 'Type I DZ based on proposed method',
    'ITE': 'Type I DZ based on ITE parameters',
    'TT': 'Type II DZ based on travel time to stop',
    'Prob': 'Type II DZ based on stopping probability'
})

pdf4.to_csv("ignore/DZ_estimation_accuracy.txt", sep = '\t', index = False)
