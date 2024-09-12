import os
import pandas as pd

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\dilemma_zone_modeling")

# read dataset with stop/go trips
df = pd.read_csv("ignore/Wejo/trips_analysis/processed_trips.txt", sep = '\t')

# read node geometry data and select relevant columns
ndf = pd.read_csv("ignore/node_geometry.csv")
ndf = ndf[['Node', 'Approach', 'Speed_limit', 'Crossing_length']]

# merge node geometry data with FTS dataset
df = pd.merge(df, ndf, on = ['Node', 'Approach'], how = 'left')

# update localtime to pandas datetime and add day variable
df.localtime = pd.to_datetime(df.localtime)
df['Month'] = df.localtime.dt.month
df['Day'] = df.localtime.dt.day

# add site ID variable for node and approach
df['SiteID'] = df.Node.astype(str) + df.Approach

speed_diff_threshold = 20 # threshold for filtering out yellow onset speed below speed limit
group_cols = ['Node', 'Approach', 'Month', 'Day', 'TripID']


# =============================================================================
# functions
# =============================================================================

# function to plot trajectory and signal information
def plotTrajectorySignal(node, dirc, month, day, trip_id):
    # filter df with trajectory and signal info for trip id
    trip_df = df.copy()[(df.Node == node) & (df.Approach == dirc) & (df.Month == month) & (df.Day == day) & (df.TripID == trip_id)]

    fig = px.scatter(
        trip_df,
        x = 'Xi',
        y = 'localtime',
        color = 'Signal',
        hover_data = ['Node', 'Approach', 'TripID', 'speed', 'TUY', 'TAY'],
        color_discrete_map = {'G': 'green', 'Y': 'orange', 'R': 'red'}
    )
    fig.update_traces(marker = dict(size = 10))
    fig.update_layout(title = str(node) + ', ' + dirc + ', ' + str(month) + ', ' + str(day) + ', ' + str(int(trip_id)))
    
    # add vertical line at stop line
    fig.add_shape(
        type = 'line',
        x0 = 0,
        y0 = trip_df['localtime'].min(),
        x1 = 0,
        y1 = trip_df['localtime'].max(),
        line = dict(color = 'black', width = 2, dash = 'dash')
    )
    fig.show()

# test trajectories
# plotTrajectorySignal(618, 'NB', 8, 28, 380) # stopping beyond stop line
# plotTrajectorySignal(517, 'WB', 8, 17, 287) # stopping beyond stop line


# =============================================================================
# process FTS (first-to-stop) trips
# =============================================================================

# filter FTS trips
FTS = df.copy()[(df.Group == 'FTS')]

# find the longest stopping position of all FTS trips
fts_stop_dist = FTS.groupby(group_cols)['Xi'].agg(lambda x: x.mode().iloc[0]).reset_index()
fts_stop_dist.rename(columns = {'Xi': 'stop_dist'}, inplace = True) # rename Xi column

# find time after yellow when the vehicle comes to stopping position
fts_stop_time = FTS.groupby(group_cols).apply(lambda x: x.loc[x.speed == 0].iloc[0]['TAY'], include_groups = False).reset_index(name = 'stop_time')

# filter yellow onset FTS trips
FTS = FTS[(FTS.TUY == 0) | (FTS.TAY == 0)]

# filter out yellow onset speed below speed limit by specified threshold
FTS['Speed_diff'] = FTS.speed - FTS.Speed_limit # difference between yellow onset speed and speed limit
FTS = FTS[FTS.Speed_diff >= -(speed_diff_threshold)] # filter out lower speeds

# add stopping position and time after yellow to GLR data
FTS = pd.merge(FTS, fts_stop_dist, on = group_cols, how = 'left')
FTS = pd.merge(FTS, fts_stop_time, on = group_cols, how = 'left')

# check deceleration based on observed time to stop and speed
FTS['Dec'] = -round(FTS.speed*(5280/3600) / FTS.stop_time, 2)

# filter out outlier observations
FTS = FTS[~((FTS.SiteID == '216EB') & (FTS.Month == 10) & (FTS.Day == 25) &(FTS.TripID == 511))]
FTS = FTS[~((FTS.SiteID == '618NB') & (FTS.Month == 9) & (FTS.Day == 11) & (FTS.ID == 107210))]
FTS = FTS[~((FTS.SiteID == '618EB') & (FTS.Month == 8) & (FTS.Day == 17) & (FTS.TripID == 252))]

# drop redundant columns and save file
FTS.drop(['ID', 'Signal', 'TUY', 'TAY', 'Group', 'Month', 'Day', 'Speed_diff'], axis = 1, inplace = True)
FTS.to_csv("ignore/Wejo/trips_analysis/trips_FTS.txt", sep = '\t', index = False)


# =============================================================================
# process YLR trips
# =============================================================================

# filter YLR trips
YLR = df.copy()[(df.Group == 'YLR')]

# filter out short YLR trips not reaching intersection stop line
ylr_short_trips = YLR.groupby(group_cols).apply(lambda x: (x.Xi < 0).any()).reset_index(name = 'crosses_stop_line')

# merge information on short trips to YLR dataset and filter out short trips
YLR = pd.merge(YLR, ylr_short_trips, on = group_cols, how = 'left')
YLR = YLR[YLR.crosses_stop_line == True]

# find the time after yellow when vehicle crosses the stop line for a trip
ylr_cross_time = YLR.groupby(group_cols).apply(lambda x: x.loc[x.Xi < 0]['TAY'].values[0]).reset_index(name = 'cross_time')

# add cross time info to YLR dataset and filter out cross times longer than 4 sec
YLR = pd.merge(YLR, ylr_cross_time, on = group_cols, how = 'left')
YLR = YLR[YLR.cross_time <= 4] # yellow interval of 4 sec is the maximum at all sites

# filter yellow onset YLR trips 
YLR = YLR[(YLR.TUY == 0) | (YLR.TAY == 0)]

# filter out yellow onset speed below speed limit by specified threshold
YLR['Speed_diff'] = YLR.speed - YLR.Speed_limit # difference between yellow onset speed and speed limit
YLR = YLR[YLR.Speed_diff >= -(speed_diff_threshold)] # filter out lower speeds

# filter out outlier observations
YLR = YLR[~((YLR.SiteID == '216WB') & (YLR.Month == 10) & (YLR.Day == 18) &(YLR.TripID == 588))]

# drop redundant columns and save file
YLR.drop(['ID', 'Signal', 'TUY', 'TAY', 'Group', 'Month', 'Day', 'Speed_diff', 'crosses_stop_line'], axis = 1, inplace = True)
YLR.to_csv("ignore/Wejo/trips_analysis/trips_YLR.txt", sep = '\t', index = False)


# =============================================================================
# process RLR trips
# =============================================================================

# filter RLR trips
RLR = df.copy()[(df.Group == 'RLR')]

# filter out short RLR trips not reaching intersection stop line
rlr_short_trips = RLR.groupby(group_cols).apply(lambda x: (x.Xi < 0).any()).reset_index(name = 'crosses_stop_line')

# merge information on short trips to RLR dataset and filter out short trips
RLR = pd.merge(RLR, rlr_short_trips, on = group_cols, how = 'left')
RLR = RLR[RLR.crosses_stop_line == True]

# find the time after yellow when vehicle crosses the stop line for a trip
rlr_cross_time = RLR.groupby(group_cols).apply(lambda x: x.loc[x.Xi < 0]['TAY'].values[0]).reset_index(name = 'cross_time')

# add cross time info to RLR dataset and filter out cross times longer than 4 sec
RLR = pd.merge(RLR, rlr_cross_time, on = group_cols, how = 'left')

# filter yellow onset RLR trips 
RLR = RLR[(RLR.TUY == 0) | (RLR.TAY == 0)]

# add speed limit data and compute speed difference
RLR = pd.merge(RLR, ndf[['Node', 'Approach', 'Speed_limit']], on = ['Node', 'Approach'], how = 'left')
RLR['Speed_diff'] = RLR.speed - RLR.Speed_limit # difference between yellow onset speed and speed limit

# filter out yellow onset speed less than -15 mph and greater than 25 mph
RLR = RLR[RLR.Speed_diff >= -15]

# drop redundant columns and save file
RLR.drop(['Signal', 'TUY', 'TAY', 'Group', 'Day', 'Speed_limit', 'Speed_diff', 'crosses_stop_line'], axis = 1, inplace = True)
RLR.to_csv("ignore/Wejo/trips_stop_go/RLR_filtered.txt", sep = '\t', index = False)
