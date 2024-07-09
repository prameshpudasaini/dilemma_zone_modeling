import os
import pandas as pd
import statsmodels.api as sm

from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\dilemma_Wejo")

# read model dataset
FTS = pd.read_csv("ignore/Wejo/trips_stop_go/model_data_FTS.txt", sep = '\t')

px.scatter(FTS, x = 'Xi', y = 'speed').show()

# target and predictor variables
X = FTS.drop(['Node', 'Approach', 'Xi'], axis = 1)
y = FTS['Xi']

# =============================================================================
# features: permutation importance using Gradient Boosting Regressor
# =============================================================================

# fit gradient boosting regressor for 5th percentile
quantile_gb = GradientBoostingRegressor(loss = 'quantile', alpha = 0.05, random_state = 42)
quantile_gb.fit(X, y)

# compute permutation importance for 5th percentile
result = permutation_importance(quantile_gb, X, y, scoring = 'neg_mean_squared_error', n_repeats = 10, random_state = 42)
feature_importance = pd.DataFrame({
    'feature': list(X.columns),
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
})

# =============================================================================
# test quantile regression at different percentiles
# =============================================================================

X = FTS[['speed', '']]

model_q1 = sm.QuantReg(y, X).fit(q = 0.01)
print(f"\nQuantile = 1 \n{model_q1.summary()}")

model_q5 = sm.QuantReg(y, X).fit(q = 0.05)
print(f"\nQuantile = 5 \n{model_q5.summary()}")

model_q10 = sm.QuantReg(y, X).fit(q = 0.10)
print(f"\nQuantile = 10 \n{model_q10.summary()}")

# add square of speed as predictor
X['speed_sq'] = X.speed**2

model_q5 = sm.QuantReg(y, X).fit(q = 0.05)
print(f"\nQuantile = 5 \n{model_q5.summary()}")

