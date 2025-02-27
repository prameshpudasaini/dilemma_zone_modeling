import os
import pandas as pd

os.chdir("/Users/prameshpudasaini/Library/CloudStorage/OneDrive-UniversityofArizona/GitHub/dilemma_zone_modeling")

Xs = pd.read_csv("ignore/quantile_regression_data/Xs_data.txt", sep = '\t')
Xc = pd.read_csv("ignore/quantile_regression_data/Xc_data.txt", sep = '\t')

Xs.is_peak.value_counts()
Xc.is_peak.value_counts()

Xs.is_night.value_counts()
Xc.is_night.value_counts()

Xs.is_weekend.value_counts()
Xc.is_weekend.value_counts()
