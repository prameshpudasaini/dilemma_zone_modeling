import os
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, mean_pinball_loss

os.chdir(r"D:\GitHub\dilemma_Wejo")

# =============================================================================
# load dataset and prepare X, y
# =============================================================================

# read FTS dataset for modeling
FTS = pd.read_csv("ignore/Wejo/trips_stop_go/model_data_FTS.txt", sep = '\t')

# target and predictor variables
X = FTS.drop(['Node', 'Approach', 'Xi'], axis = 1)
y = FTS['Xi']

# split into train, test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)

# =============================================================================
# implementation of XGBoost quantile regression
# =============================================================================

q = 0.05

# Define the custom quantile loss function
def quantile_loss(q, y_true, y_pred):
    errors = y_true - y_pred
    loss = np.where(errors < 0, (1 - q) * np.abs(errors), q * np.abs(errors))
    return np.mean(loss)

def quantile_objective(y_true, y_pred):
    errors = y_true - y_pred
    grad = np.where(errors < 0, -(1 - q), q)  # Gradient
    hess = np.ones_like(grad)  # Hessian
    return grad, hess

# Create DMatrix objects for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define a grid of hyperparameters
param_grid = {
    'max_depth': [2, 3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.5, 0.7, 0.8],
    'colsample_bytree': [0.5, 0.7, 0.8],
    'n_estimators': [50, 100, 200, 300]
}

# Set up the XGBoost model with a custom objective
model = xgb.XGBRegressor(objective=quantile_objective, base_score=np.median(y_train))

# Define the custom scorer using make_scorer
def pinball_scorer(y_true, y_pred):
    return mean_pinball_loss(y_true, y_pred, alpha=q)

custom_scorer = make_scorer(pinball_scorer, greater_is_better=False)

# Perform grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, scoring=custom_scorer, cv=3, verbose=1, n_jobs=-1, error_score='raise')
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Make predictions using DMatrix for compatibility
predictions = best_model.predict(X_test)

# Evaluate the model using mean_pinball_loss
loss = mean_pinball_loss(y_test, predictions, alpha=q)
print(f"Mean Pinball Loss (5th percentile): {loss}")

# Feature importance
importances = best_model.feature_importances_
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
feature_importance = dict(zip(feature_names, importances))

# Sort and print feature importance
sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
print("Feature importance:")
for feature, importance in sorted_importance:
    print(f"{feature}: {importance:.4f}")
# =============================================================================
# existing model
# =============================================================================

# read datasets with stop/go trips
GLR = pd.read_csv("ignore/Wejo/trips_stop_go/GLR_filtered.txt", sep = '\t') # GLR trips
FTS = pd.read_csv("ignore/Wejo/trips_stop_go/FTS_filtered.txt", sep = '\t') # FTS trips
YLR = pd.read_csv("ignore/Wejo/trips_stop_go/YLR_filtered.txt", sep = '\t') # YLR trips
RLR = pd.read_csv("ignore/Wejo/trips_stop_go/RLR_filtered.txt", sep = '\t') # RLR trips

# combine GLR with FTS data
FTS = pd.concat([FTS, GLR], ignore_index = True)

# specify num bins and quantile for Xs modeling
num_bins, lower_q, upper_q = 30, 0.1, 0.9

# adaptive binning using quantiles
FTS['speed_bin'] = pd.qcut(FTS['speed'], q = num_bins, duplicates = 'drop')
YLR['speed_bin'] = pd.qcut(YLR['speed'], q = num_bins, duplicates = 'drop')

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

Xs_xgb = best_model.predict(X)
FTS['Xs_xgb'] = Xs_xgb

px.scatter(FTS, x = 'Xs_xgb', y = 'speed').show()

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