import os
import pandas as pd

os.chdir(r"D:\GitHub\dilemma_Wejo")

# read datasets with stop/go trips
GLR = pd.read_csv("ignore/Wejo/trips_stop_go/GLR_filtered.txt", sep = '\t') # GLR trips
FTS = pd.read_csv("ignore/Wejo/trips_stop_go/FTS_filtered.txt", sep = '\t') # FTS trips

# combine GLR with FTS data
FTS = pd.concat([FTS, GLR], ignore_index = True)

# read node geometry data and select relevant columns
ndf = pd.read_csv("ignore/node_geometry_v2.csv")

# # add binary variable for dual left lanes
# ndf['dual_LT_lanes'] = ndf.num_LT_lanes.apply(lambda x: 1 if x == 2 else 0)

# select relevant node geometry variables
ndf = ndf[['Node', 'Approach', 'Speed_limit', 'int_cross_length', 'num_TH_lanes', 'has_shared_RT', 'has_median']]

# compute correlation between node geometry variables
node_geo_corr = ndf.copy().drop(['Node', 'Approach'], axis = 1).corr()

# merge node geometry data
FTS = pd.merge(FTS, ndf, on = ['Node', 'Approach'], how = 'left')

# update localtime to datetime and add is_weekend, is_night variables
FTS.localtime = pd.to_datetime(FTS.localtime)
FTS['is_weekend'] = (FTS.localtime.dt.dayofweek >= 5).astype(int)
FTS['is_night'] = (~FTS.localtime.dt.hour.between(5, 19, inclusive = 'both')).astype(int)

# convert speed and speed limit to ft/s
FTS.speed = round(FTS.speed * 5280/3600, 1)
FTS.Speed_limit = round(FTS.Speed_limit * 5280/3600, 1)

# drop redundant columns and save file
FTS.drop(['ID', 'TripID', 'localtime', 'stop_dist', 'stop_time'], axis = 1, inplace = True)
FTS.to_csv("ignore/Wejo/trips_stop_go/model_data_FTS.txt", sep = '\t', index = False)
