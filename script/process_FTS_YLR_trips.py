import os
import pandas as pd

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\dilemma_Wejo")

# read dataset with stop/go trips
df = pd.read_csv("ignore/Wejo/trips_stop_go/trips_stop_go.txt", sep = '\t')

# read node geometry data
ndf = pd.read_csv("ignore/node_geometry.csv")

# =============================================================================
# count trips and create list of trips by group
# =============================================================================

# create dataset for counting number of trips
cdf = df.copy()[['Node', 'Approach', 'Day', 'TripID', 'Group']]
cdf.drop_duplicates(inplace = True) # drop all duplicates

# count number of trips by group and node
count_trips_group = cdf['Group'].value_counts().reset_index()
count_trips_node = cdf['Node'].value_counts().reset_index()

group_cols = ['Node', 'Approach', 'Day', 'TripID']

# create list of trips by group with details on: node, approach, day, trip ID
def list_trips_by_group(group):
    xdf = cdf.copy()[cdf.Group == group]
    trips = [xdf[group_cols].iloc[i].tolist() for i in range(len(xdf))]
    return trips

trips_GLR = list_trips_by_group('GLR')
trips_FTS = list_trips_by_group('FTS')
trips_YLR = list_trips_by_group('YLR')
trips_RLR = list_trips_by_group('RLR')

# =============================================================================
# functions
# =============================================================================

# function to plot trajectory and signal information
def plotTrajectorySignal(node, dirc, day, trip_id):
    # filter df with trajectory and signal info for trip id
    trip_df = df.copy()[(df.Node == node) & (df.Approach == dirc) & (df.Day == day) & (df.TripID == trip_id)]

    fig = px.scatter(
        trip_df,
        x = 'Xi',
        y = 'localtime',
        color = 'Signal',
        hover_data = ['Node', 'Approach', 'TripID', 'speed', 'TUY', 'TAY'],
        color_discrete_map = {'G': 'green', 'Y': 'orange', 'R': 'red'}
    )
    fig.update_traces(marker = dict(size = 10))
    fig.update_layout(title = str(node) + ', ' + dirc + ', ' + str(day) + ', ' + str(int(trip_id)))
    
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


# =============================================================================
# process GLR trips (trips with stopping position beyond intersection stop line)
# =============================================================================

# filter GLR trips
GLR = df.copy()[df.Group == 'GLR']

# find the longest stopping position of all GLR trips
glr_stop_dist = GLR.groupby(group_cols)['Xi'].agg(lambda x: x.mode().iloc[0]).reset_index()
glr_stop_dist.rename(columns = {'Xi': 'stop_dist'}, inplace = True) # rename Xi column

# # plot stopping position by node and approach
# fig_stop_dist = px.scatter(
#     glr_stop_dist,
#     x = 'stop_dist',
#     y = 'Node',
#     color = 'Approach'
# )
# fig_stop_dist.update_traces(marker = dict(size = 20))
# fig_stop_dist.show()

# # check trajectory for GLR data
# for i in range(0, len(trips_GLR)):
#     node, dirc, day, trip_id = glr_list[i]
#     plotTrajectorySignal(node, dirc, day, trip_id)

# plotTrajectorySignal(*trips_GLR[1])

# Note: analysis of plots show that GLR were FTS vehicles that stopped beyond
# the intersection stop line.

# find time after yellow when the vehicle comes to stopping position
glr_stop_time = GLR.groupby(group_cols).apply(lambda x: x.loc[x.speed == 0].iloc[0]['TAY']).reset_index(name = 'stop_time')

# filter yellow onset GLR trips
GLR = GLR[(GLR.TUY == 0) | (GLR.TAY == 0)]

# add speed limit data and filter out yellow onset speed 15 miles below speed limit
GLR = pd.merge(GLR, ndf[['Node', 'Approach', 'Speed_limit']], on = ['Node', 'Approach'], how = 'left')
GLR['Speed_diff'] = GLR.speed - GLR.Speed_limit # difference between yellow onset speed and speed limit
GLR = GLR[GLR.Speed_diff >= -15] # filter out lower speeds

# add stopping position and time after yellow to GLR data
GLR = pd.merge(GLR, glr_stop_dist, on = group_cols, how = 'left')
GLR = pd.merge(GLR, glr_stop_time, on = group_cols, how = 'left')

# check deceleration based on observed time to stop and speed
GLR['Dec'] = round(GLR.speed*(5280/3600) / GLR.stop_time, 2)

# filter out data with stopping position beyond 20 ft of intersection stop line
# filter out data with yellow onset distance less than 100 ft
# this considers GLR data as FTS, given the spatial accuracy of trajectory data
GLR = GLR[(GLR.stop_dist >= -20) & (GLR.Xi >= 100)]

# drop redundant columns and save file
GLR.drop(['Signal', 'TUY', 'TAY', 'Group', 'Day', 'Speed_limit', 'Speed_diff', 'Dec'], axis = 1, inplace = True)
GLR.to_csv("ignore/Wejo/trips_stop_go/GLR_filtered.txt", sep = '\t', index = False)


# =============================================================================
# process FTS (first-to-stop) trips
# =============================================================================

# filter FTS trips
FTS = df.copy()[(df.Group == 'FTS')]

# find the longest stopping position of all GLR trips
fts_stop_dist = FTS.groupby(group_cols)['Xi'].agg(lambda x: x.mode().iloc[0]).reset_index()
fts_stop_dist.rename(columns = {'Xi': 'stop_dist'}, inplace = True) # rename Xi column

# # test to check IndexError: single positional indexer is out-of-bounds
# xdf = FTS.copy()[(FTS.Node == 540) & (FTS.Day == 5) & (FTS.Approach == 'EB') & (FTS.TripID == 215)]
# fts_stop_time = xdf.groupby(group_cols).apply(lambda x: x.loc[x.speed == 0].iloc[0]['TAY']).reset_index(name = 'stop_time')

# filter out data with node, dirc, day, trip_id = 540, 'EB', 5, 215
# reason: IndexError: single positional indexer is out-of-bounds
# while computing time after yellow when the vehicle comes to stopping position
FTS = FTS[~((FTS.Node == 540) & (FTS.Day == 5) & (FTS.Approach == 'EB') & (FTS.TripID == 215))]

# find time after yellow when the vehicle comes to stopping position
fts_stop_time = FTS.groupby(group_cols).apply(lambda x: x.loc[x.speed == 0].iloc[0]['TAY']).reset_index(name = 'stop_time')

# filter yellow onset FTS trips
FTS = FTS[(FTS.TUY == 0) | (FTS.TAY == 0)] 

# add speed limit data and filter out yellow onset speed 15 miles below speed limit
FTS = pd.merge(FTS, ndf[['Node', 'Approach', 'Speed_limit']], on = ['Node', 'Approach'], how = 'left')
FTS['Speed_diff'] = FTS.speed - FTS.Speed_limit # difference between yellow onset speed and speed limit
FTS = FTS[FTS.Speed_diff >= -15] # filter out lower speeds

# add stopping position and time after yellow to GLR data
FTS = pd.merge(FTS, fts_stop_dist, on = group_cols, how = 'left')
FTS = pd.merge(FTS, fts_stop_time, on = group_cols, how = 'left')

# check deceleration based on observed time to stop and speed
FTS['Dec'] = round(FTS.speed*(5280/3600) / FTS.stop_time, 2)

# filter out data with time until stop > 25 sec
# filter out data with stopping position beyond 20 ft
FTS = FTS[(FTS.stop_time < 25) & (FTS.stop_dist.between(-20, 20, inclusive = 'both'))]

# filter out data with speed difference > -5 and yellow onset distance less than 80
FTS = FTS[~((FTS.Xi < 80) & (FTS.Speed_diff <= 0))]

# drop redundant columns and save file
FTS.drop(['Signal', 'TUY', 'TAY', 'Group', 'Day', 'Speed_limit', 'Speed_diff', 'Dec'], axis = 1, inplace = True)
FTS.to_csv("ignore/Wejo/trips_stop_go/FTS_filtered.txt", sep = '\t', index = False)


# =============================================================================
# process YLR trips
# =============================================================================

# filter YLR trips
YLR = df.copy()[(df.Group == 'YLR')]

# # test to check IndexError: single positional indexer is out-of-bounds
# ylr_cross_time = YLR.groupby(group_cols).apply(lambda x: x.loc[x.Xi < 0]['TAY'].values).reset_index(name = 'cross_time')
# plotTrajectorySignal(217, 'EB', 6, 620)
# plotTrajectorySignal(217, 'EB', 23, 570)
# # Note: analysis showed that some trips processed as YLR did not reach intersection stop line.

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

# add speed limit data and compute speed difference
YLR = pd.merge(YLR, ndf[['Node', 'Approach', 'Speed_limit']], on = ['Node', 'Approach'], how = 'left')
YLR['Speed_diff'] = YLR.speed - YLR.Speed_limit # difference between yellow onset speed and speed limit

# filter out yellow onset speed less than -15 mph and greater than 25 mph
YLR = YLR[YLR.Speed_diff.between(-15, 25, inclusive = 'both')]

# drop redundant columns and save file
YLR.drop(['Signal', 'TUY', 'TAY', 'Group', 'Day', 'Speed_limit', 'Speed_diff', 'crosses_stop_line'], axis = 1, inplace = True)
YLR.to_csv("ignore/Wejo/trips_stop_go/YLR_filtered.txt", sep = '\t', index = False)


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
