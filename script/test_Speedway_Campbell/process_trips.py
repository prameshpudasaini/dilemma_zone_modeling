# test carried out at Speedway Blvd & Campbell Ave

import os
import numpy as np
import pandas as pd
import time

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\dilemma_Wejo")

# =============================================================================
# parameters
# =============================================================================

# coordinates of intersection cross line, stop line, and advance location (500 ft u/s from stop line)
coord = {'EB': {'stop': [32.236004699647296, -110.94427309956106], 
                'cross': [32.23599934372346, -110.94422060896643], 
                'adv': [32.23596695607203, -110.94589300691366]},
         'WB': {'stop': [32.23615663766954, -110.94368894895453], 
                'cross': [32.2361527696248, -110.94374039888217], 
                'adv': [32.23615922544614, -110.94206785480095]}}

# dictionaries for mapping values
approach_direction = {0: 'NB', 90: 'EB', 180: 'SB', 270: 'WB', 360: 'NB'} # approaching direction
signal_event = {1: 'G', 8: 'Y', 10: 'R'} # signal phase change events
phase_parameter = {2: 'EB', 4: 'SB', 6: 'WB', 8: 'NB'} # phase parameters

stop_speed_threshold = 5 # speed threshold to determine speeding
FTS_threshold = 20 # queueing distance from intersection stop line where a vehicle is first-to-stop

yellow_interval = 4
PRT = 1
# PSL = 35 # posted speed limit
# v85 = round((PSL + 7) * 5280/3600, 2) # 85th percentile speed

# =============================================================================
# read trajectory and signal files
# =============================================================================

# file paths
input_path_wejo = "script/test_Speedway_Campbell/data/Wejo/Speedway_Campbell_08.txt"
input_path_signal = "script/test_Speedway_Campbell/data/MaxView/MaxView_08_217.txt"

# read wejo and maxview data
wdf = pd.read_csv(input_path_wejo, sep = '\t')
mdf = pd.read_csv(input_path_signal, sep = '\t')

# update signal data with event and parameter values
mdf.rename(columns = {'TimeStamp': 'localtime', 'EventId': 'Signal', 'Parameter': 'Approach'}, inplace = True)
mdf.Signal = mdf.Signal.map(signal_event)
mdf.Approach = mdf.Approach.map(phase_parameter)

# convert timestamps to datetime format
wdf.localtime = pd.to_datetime(wdf.localtime)
mdf.localtime = pd.to_datetime(mdf.localtime)

# =============================================================================
# functions
# =============================================================================

# function to seggregate multiple sub trips across intersection within a trip
def segregate_sub_trips(day_df):
    trip_dfs = [] # list to hold resulting segregated trip dfs
    
    # loop through each trip and add to trip dfs
    for trip in list(day_df.TripID.unique()):
        # filter for trip
        xdf = day_df.copy()[day_df.TripID == trip]
        xdf.reset_index(drop = True, inplace = True)
        
        # add trip change indicator variable
        xdf['trip_change'] = xdf.LocationID - xdf.LocationID.shift(1)
        trip_change_index = xdf[xdf['trip_change'] > 1].index.to_list()
        
        # check if there are multiple sub trips for the selected trip
        if len(trip_change_index) >= 1:
            # add 0 as starting point of trip and length of df as the ending point
            trip_split_index = [0] + trip_change_index + [len(xdf)]
            
            # loop through the split points and create sub dfs
            for i in range(len(trip_split_index) - 1):
                start, end = trip_split_index[i], trip_split_index[i + 1]
                sub_df = xdf.copy().iloc[start:end]
                
                # append i to the end of trip id
                sub_df.TripID = (sub_df.TripID.astype(int).astype(str) + str(i)).astype(int)
                
                # drop trip change column and append sub df
                sub_df.drop('trip_change', axis = 1, inplace = True)
                trip_dfs.append(sub_df)
        else:
            xdf.drop('trip_change', axis = 1, inplace = True)
            trip_dfs.append(xdf)
            
    result = pd.concat(trip_dfs, ignore_index = True)
    
    return result


# function to round trip direction
def round_direction(direction):
    target_values = np.array([0, 90, 180, 270, 360])
    
    # find the closest target value for each direction
    return target_values[np.abs(target_values - direction).argmin()]

# function to filter thru trips based on direction statistic
def process_trip_direction(xdf):
    # compute mean and standard deviation statistic of each trip's direction
    adf = xdf.groupby('TripID')['direction'].agg(dirc_mean = 'mean', dirc_std = 'std').reset_index()
    
    # high st dev of trip direction indicates moving to driveways or taking U-turns
    # filter trips with st dev of trip direction < 2
    adf = adf[adf.dirc_std <= 2]
    
    # round average direction and update direction
    adf.dirc_mean = adf.dirc_mean.apply(round_direction)
    
    # categorize approach based on average direction
    adf['approach'] = adf.dirc_mean.map(approach_direction)
    adf.drop(['dirc_mean', 'dirc_std'], axis = 1, inplace = True)
    
    # create a dictionary of filtered trip IDs as keys and approaching direction as values
    records = adf.to_dict(orient = 'records')
    thru_trips = {record['TripID']: record['approach'] for record in records}
    
    return thru_trips


# function to interpolate values for each trip ID
def interpolate_trip(trip_df):
    # create a new date range with millisecond intervals
    new_index = pd.date_range(start = trip_df.index.min(), end = trip_df.index.max(), freq='100L')
    
    # reindex the dataframe to include the new date range
    trip_df = trip_df.reindex(new_index)
    
    # interpolate intermediate values
    trip_df['TripID'] = trip_df['TripID'].interpolate(method = 'nearest')
    trip_df['LocationID'] = trip_df['LocationID'].interpolate(method = 'linear')
    trip_df['Latitude'] = trip_df['Latitude'].interpolate(method = 'linear')
    trip_df['Longitude'] = trip_df['Longitude'].interpolate(method = 'linear')
    trip_df['speed'] = trip_df['speed'].interpolate(method = 'linear')
    
    # round locationID to the nearest 2 digits
    trip_df['LocationID'] = trip_df['LocationID'].round(2)
    
    # round speed to the nearest 0.1 mph
    trip_df['speed'] = trip_df['speed'].round(1)
    
    return trip_df


# Haversine formula to compute the distance between two points given their latitude and longitude
def haversine(lat, lon, lat_ref, lon_ref):
    R = 6371.0 # radius of Earth in km
    
    dlon = np.radians(lon - lon_ref)
    dlat = np.radians(lat - lat_ref)
    
    # compute square of half the chord length between the two points
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat_ref)) * np.cos(np.radians(lat)) * np.sin(dlon / 2)**2
    
    # compute angular distance in radians
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    dist = R * c * 1000 * 3.28084 # conversion to meters and feet
    return dist

# # check haversine formula
# haversine(32.23615283527675, -110.9432987728307, 32.23615474098094, -110.94370221489316)
# haversine(32.236154206339286, -110.94375308222527, 32.23615474098094, -110.94370221489316)


# function to round up distance to the nearest 50 ft
def round_distance(x):
    return np.ceil(x / 50) * 50

# function to check whether a trip is FTS, YLR, RLR, queued, or stopped multiple times
def check_FTS_YLR_RLR(cdf, trip_id):
    # filter for trip id and select relevant columns
    xdf = cdf.copy()[(cdf.TripID == trip_id)][['speed', 'Signal', 'Xi_cross', 'Xi_stop']]
    xdf.reset_index(inplace = True, drop = True)
    
    # create bins of distance from intersection stop line
    xdf['Xi_bin'] = xdf.Xi_stop.apply(round_distance)
    
    # compute minimum speed in each bin
    bdf = xdf[xdf.Xi_stop >= 0].groupby('Xi_bin')['speed'].agg(min_speed = 'min').reset_index()
    
    # check for stops in each bin and count number of stops    
    bdf['is_stop'] = bdf.min_speed <= stop_speed_threshold
    num_stops = bdf.is_stop.sum()
    
    # classify the trip as FTS, YLR, RLR   
    if num_stops == 0:
        # for 0 stops, find the signal status corresponding to intersection cross line
        group = str(xdf.loc[xdf.Xi_cross.idxmin(), 'Signal']) + 'LR'
    
    elif num_stops <= 2:
        # filter data with zero speed u/s of intersection stop line
        xdf0 = xdf.copy()[(xdf.speed == 0) & (xdf.Xi_stop >= 0)]
        # round up vehicle position to nearest 10
        xdf0['Xi_10'] = np.ceil(xdf0.Xi_stop / 10) * 10
        # mode gives the position of longest stop
        mode_stop = int(xdf0.Xi_10.mode())

        # check if the vehicle is within the first-to-stop distance threshold
        if mode_stop <= FTS_threshold:
            group = 'FTS' # first to stop
        else:
            group = 'Queued' # other positions in queue
    
    else:
        # vehicle made many stops approaching the intersection
        group = 'Stopped'
        
    return group


# function to process trajectory and signal data
def process_trajectory_signal_data(day, direction):
    # filter trips for day
    print(f"\nDay, direction: {day}, {direction}")
    day_df = wdf.copy()[wdf.localtime.dt.day == day]
    
    # seggregate multiple sub trips across intersection within a trip
    seg_df = segregate_sub_trips(day_df)
    
    # check length of day df and seg df are equal
    if len(day_df) == len(seg_df):
        print("Trips segregation OK!")
    else:
        print("Trips segregation ERROR!")
    
    # process trip direction and filter thru trips in given direction
    thru_trips = process_trip_direction(seg_df)
    thru_trips_filtered = [key for key, value in thru_trips.items() if value == direction]
    
    print(f"Num of trips, sub trips, thru trips: {len(day_df.TripID.unique())}, {len(seg_df.TripID.unique())}, {len(thru_trips_filtered)}")
    
    # filter trips for direction
    ddf = seg_df.copy()[seg_df.TripID.isin(thru_trips_filtered)]
    
    # drop direction column
    ddf.drop('direction', axis = 1, inplace = True)
    
    # set localtime as index for interpolating trip attributes
    ddf.set_index('localtime', inplace = True)
    
    # apply the interpolation function to each tripID
    idf = ddf.groupby('TripID').apply(interpolate_trip)
    
    # groupby adds an extra level to the index, so reset index
    idf.index = idf.index.get_level_values(1)
    
    # reset localtime as column
    idf.reset_index(inplace = True)
    idf.rename(columns = {'index': 'localtime'}, inplace = True)
    
    # filter trips between intersection cross line and adv location
    coord_dirc = coord[direction] # specify coordinates for input direction
    cross, stop, adv = coord_dirc['cross'], coord_dirc['stop'], coord_dirc['adv']
    idf = idf[idf.Longitude.between(min(cross[1], adv[1]), max(cross[1], adv[1]), inclusive = 'both')]
    
    # get min, max of each trip's time and compute duration
    trip_times = idf.groupby(['TripID']).agg({'localtime': ['min', 'max']}).reset_index()
    trip_times.columns = ['TripID', 'start_time', 'end_time']
    trip_times['duration'] = (trip_times.end_time - trip_times.start_time).dt.total_seconds()
    print(f"Min, max of trips times: {trip_times.duration.min()}, {trip_times.duration.max()}")
    
    # # filter out trips with trip duration > 300 sec and < 5 sec
    # trips_5_to_300 = list(trip_times[trip_times.duration.between(5, 300, inclusive = 'both')].TripID)
    # idf = idf[idf.TripID.isin(trips_5_to_300)]
    
    # filter signal data for day and direction
    sdf = mdf.copy()[(mdf.localtime.dt.day == day) & (mdf.Approach == direction)]
    sdf.drop('Approach', axis = 1, inplace = True)
    sdf.sort_values(by = 'localtime', inplace = True) # sort data by timestamp
    
    # append signal info to trajectory data
    cdf = pd.concat([idf, sdf], ignore_index = True)
    cdf.sort_values(by = 'localtime', inplace = True)
    
    # compute wait time until yellow
    cdf.loc[cdf.Signal == 'Y', 'yellowtime'] = cdf.localtime
    
    # backward fill yellow time for time until yellow
    cdf['TUY'] = cdf.yellowtime.bfill()
    cdf.TUY = round((cdf.TUY - cdf.localtime).dt.total_seconds(), 1)
    
    # forward fill yellow time for time after yellow
    cdf['TAY'] = cdf.yellowtime.ffill()
    cdf.TAY = round((cdf.localtime - cdf.TAY).dt.total_seconds(), 1)
    
    # forward fill signal info
    cdf.Signal.ffill(inplace = True)
    
    # filter trips facing yellow indication before the intersection stop line
    yellow_onset_trips = list(cdf[(cdf.Signal == 'Y') & ((cdf.TUY == 0) | (cdf.TAY == 0))]['TripID'].unique())
    cdf = cdf[cdf.TripID.isin(yellow_onset_trips)]
    print(f"Num of yellow onset trips: {len(cdf.TripID.unique())}")
    
    # remove redundant cols and rows with nan values
    cdf.drop('yellowtime', axis = 1, inplace = True)
    cdf.dropna(subset = ['TripID', 'Signal'], inplace = True)
    
    # compute distance from intersection cross line using Haversine formula
    cdf['Xi_cross'] = cdf.apply(lambda row: haversine(row['Latitude'], row['Longitude'], cross[0], cross[1]), axis = 1)
    cdf.Xi_cross = cdf.Xi_cross.round(0)
    
    # update Xi for distance from intersection stop line
    dist_cross_stop = round(haversine(cross[0], cross[1], stop[0], stop[1]), 0)
    cdf['Xi_stop'] = cdf.Xi_cross - dist_cross_stop
    
    # initiate dictionary to store trip id and corresponding FTS, YLR, RLR
    group = {}
    
    # loop through each trip to compute FTS, YLR, RLR status
    for trip in list(cdf.TripID.unique()):
        group[trip] = check_FTS_YLR_RLR(cdf, trip)
        
    # update each trip into FTS, YLR, RLR, Queued, Stopped status
    cdf['Group'] = cdf.TripID.map(group)
    
    # filter for FTS, YLR, RLR groups
    cdf = cdf[cdf.Group.isin(['FTS', 'YLR', 'RLR'])]
    print(f"Num of FTS, YLR, RLR trips: {len(cdf.TripID.unique())}")
    
    # add approach direction as variable
    cdf['Approach'] = direction
    
    # filter first-to-stop trips
    df_FTS = cdf.copy()[(cdf.Group == 'FTS') & ((cdf.TUY == 0) | (cdf.TAY == 0))]
    
    # filter yellow and red light running trips
    df_YLR = cdf.copy()[(cdf.Group.isin(['YLR', 'RLR']))]
    df_YLR.reset_index(drop = True, inplace = True)
    
    # for YLR & RLR trips, find the time when vehicle crosses the stop line
    df_YLR['Time_Cross'] = df_YLR.iloc[df_YLR.groupby('TripID').Xi_cross.idxmin()].TAY
    df_YLR.Time_Cross.bfill(inplace = True)
    
    # filter YLR observations at yellow onset
    df_YLR = df_YLR[(df_YLR.TUY == 0) | (df_YLR.TAY == 0)]
    
    return {
        'trip_times': trip_times,
        'group': group,
        'df_FTS': df_FTS,
        'df_YLR': df_YLR
    }

# =============================================================================
# process Wejo trips for each day and direction
# =============================================================================

# common days for which trajectory and signal data are available
days = set(wdf.localtime.dt.day.unique()).intersection(mdf.localtime.dt.day.unique())

list_FTS, list_YLR = [], [] # store processed data for each day

# loop through each day and combine trajectory-signal data
for day in days:
    for direction in ['WB', 'EB']:        
        # process trajectory & signal info
        result = process_trajectory_signal_data(day, direction)
        list_FTS.append(result['df_FTS'])
        list_YLR.append(result['df_YLR'])
        
df_FTS = pd.concat(list_FTS, ignore_index = True)
df_YLR = pd.concat(list_YLR, ignore_index = True)
