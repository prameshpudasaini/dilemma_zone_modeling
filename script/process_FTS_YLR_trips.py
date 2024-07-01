import os
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\dilemma_Wejo")

# function to plot trajectory and signal information
def plotTrajectorySignal(node, dirc, day, trip_id):
    # filter df with trajectory and signal info for trip id
    trip_df = df.copy()[(df.Node == node) & (df.Approach == dirc) & (df.Day == day) & (df.TripID == trip_id)]

    fig = px.scatter(
        trip_df,
        x = 'Xi',
        y = 'localtime',
        color = 'Signal',
        hover_data = ['TripID', 'LocationID'],
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
# read trajectory data combined with signal info
# =============================================================================

# path and list of files
file_path = "ignore/Wejo/processed_data"
file_list = os.listdir(file_path)

list_df = [] # initialize empty list to store 
# loop through each file and append to list
for file in file_list:
    file_df = pd.read_csv(os.path.join(file_path, file), sep = '\t')
    
    # add node and approach as variables
    node, dirc = file[:3], file[4:6]
    file_df['Node'] = node
    file_df['Approach'] = dirc

    list_df.append(file_df)

# combine list into a single df    
df = pd.concat(list_df, ignore_index = True)

# process variables
df.localtime = pd.to_datetime(df.localtime) # update localtime to datetime
df['Day'] = df.localtime.dt.day # add day variable
df.Node = df.Node.astype(int) # convert string to integer
df.TripID = df.TripID.astype(int)

# count number of trips by group
count_df = df.copy()[['Node', 'Approach', 'Day', 'TripID', 'Group']]
count_df.drop_duplicates(inplace = True)
count_trips_group = count_df['Group'].value_counts().reset_index()
count_trips_node = count_df['Node'].value_counts().reset_index()

# =============================================================================
# check: trips labelled GLR
# =============================================================================

# find the longest stopping position of all GLR data
glr_mode_stop = df[df.Group == 'GLR'].groupby(['Node', 'Approach', 'Day', 'TripID'])['Xi'].agg(lambda x: x.mode().iloc[0]).reset_index()
glr_mode_stop.Node = glr_mode_stop.Node.astype(str)

# plot stopping position by node and approach
fig_mode_stop = px.scatter(
    glr_mode_stop,
    x = 'Xi',
    y = 'Node',
    color = 'Approach'
)
fig_mode_stop.update_traces(marker = dict(size = 20))
fig_mode_stop.show()

# count GLR data by node, dirc, day and convert each row to list
glr_df = count_df[count_df.Group == 'GLR']
glr_list = [glr_df[['Node', 'Approach', 'Day', 'TripID']].iloc[i].tolist() for i in range(len(glr_df))]

# # check trajectory for GLR data
# for i in range(60, len(glr_list)):
#     node, dirc, day, trip_id = glr_list[i]
#     plotTrajectorySignal(node, dirc, day, trip_id)

# Note: analysis of plots show that GLR were FTS vehicles that stopped beyond
# the intersection stop line. These data will not be considered as FTS in following analysis.

# =============================================================================
# analyze FTS trips
# =============================================================================

# filter trips for FTS trips and select relevant cols
FTS = df.copy()[(df.Group == 'FTS') & ((df.TUY == 0) | (df.TAY == 0))]
FTS = FTS[['Node', 'Approach', 'TripID', 'Xi', 'speed', 'localtime']]

# merge speed limit info into FTS data
ndf = pd.read_csv("ignore/node_geometry.csv") # node geometry data
FTS = pd.merge(FTS, ndf[['Node', 'Approach', 'Speed_limit']], on = ['Node', 'Approach'], how = 'left')

# check for low speed for each speed limit range
FTS.loc[FTS.Speed_limit.isin([30, 35]), 'low_speed'] = FTS.speed < 20
FTS.loc[FTS.Speed_limit == 40, 'low_speed'] = FTS.speed < 25

# filter out trips with low speed at the yellow onset
FTS = FTS[FTS.low_speed == False]
print(f"Num of FTS trips: {len(FTS)}")

# add day of week and hour variables
FTS['Day'] = FTS.localtime.dt.dayofweek # Monday = 0, Sunday = 6
FTS['Hour'] = FTS.localtime.dt.hour

# drop redundant columns and save file
FTS.drop(['localtime', 'low_speed'], axis = 1, inplace = True)

# =============================================================================
# analyze YLR trips
# =============================================================================

YLR_trips = df[df.Group == 'YLR'][['Node', 'Approach', 'Day', 'TripID']].drop_duplicates()
YLR_trips = [YLR_trips.iloc[i].tolist() for i in range(len(YLR_trips))]

list_YLR = []
for trip in YLR_trips:
    # filter df for node, dirc, day and trip id
    node, dirc, day, trip_id = trip
    ydf = df.copy()[(df.Node == node) & (df.Approach == dirc) & (df.Day == day) & (df.TripID == trip_id)]
    
    # find row index where vehicle crosses intersection and find corresponding time after yellow
    cross_time_index = ydf.Xi.idxmin()
    cross_time = df.iloc[cross_time_index, ydf.columns.get_loc('TAY')]
    
    # add cross time as variable and append data to list
    ydf = ydf[((ydf.TUY == 0) | (ydf.TAY == 0))] # filter row with yellow onset
    ydf['Cross_time'] = cross_time
    list_YLR.append(ydf)

# combine files from list    
YLR = pd.concat(list_YLR, ignore_index = True)

# # check YLR with TUY = 0
# plotTrajectorySignal(618, 'EB', 6, 325) # yellow onset while entering intersection
# plotTrajectorySignal(618, 'EB', 13, 514) # yellow onset while entering intersection

# add speed limit info
YLR = pd.merge(YLR, ndf[['Node', 'Approach', 'Speed_limit']], on = ['Node', 'Approach'], how = 'left')

# check for low speed for each speed limit range
YLR.loc[YLR.Speed_limit.isin([30, 35]), 'low_speed'] = YLR.speed < 20
YLR.loc[YLR.Speed_limit == 40, 'low_speed'] = YLR.speed < 25

# filter out trips with low speed at the yellow onset
YLR = YLR[YLR.low_speed == False]
YLR.drop('low_speed', axis = 1, inplace = True)
print(f"Num of YLR trips: {len(YLR)}")

# save FTS and YLR data
FTS.to_csv("ignore/Wejo/FTS_YLR_data/FTS.txt", sep = '\t', index = False)
YLR.to_csv("ignore/Wejo/FTS_YLR_data/YLR.txt", sep = '\t', index = False)
