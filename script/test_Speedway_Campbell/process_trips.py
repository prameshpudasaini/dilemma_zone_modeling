# test carried out at Speedway Blvd & Campbell Ave

import os
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\dilemma_Wejo")

# =============================================================================
# parameters
# =============================================================================

# coordinates of intersection stop line and longitude value at advance location
# Note: advance location is considered 500 ft upstream of intersection stop line based on
# Type II travel time-based DZ starting at 5.5 sec upstream of intersection stop line
# 3 sec was added to 5.5 sec to account for the resolution of Wejo trajectory data
coord = {'EB': {'lat_stop': 32.23599710503406, 'lon_stop': -110.94425278147006, 'lon_adv': -110.94587509416841},
         'WB': {'lat_stop': 32.236153104983266, 'lon_stop': -110.94370075761138, 'lon_adv': -110.94207789594019}}

# dictionaries for mapping values
approach_direction = {0: 'NB', 90: 'EB', 180: 'SB', 270: 'WB', 360: 'NB'} # approaching direction
signal_event = {1: 'G', 8: 'Y', 10: 'R'} # signal phase change events
phase_parameter = {2: 'EB', 4: 'SB', 6: 'WB', 8: 'NB'} # phase parameters

yellow_interval = 4
PSL = 35 # posted speed limit
v85 = round((PSL + 7) * 5280/3600, 2) # 85th percentile speed

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

# function to round trip direction
def round_direction(direction):
    target_values = np.array([0, 90, 180, 270, 360])
    
    # find the closest target value for each direction
    return target_values[np.abs(target_values - direction).argmin()]

# function to round direction to nearest 50
def round_distance(x):
    return 'dist_' + str(int(np.ceil(x / 50) * 50)).zfill(3)

# Haversine formula to compute the distance between two points given their latitude and longitude
def haversine(lat, lon, lat_stop, lon_stop):
    R = 6371.0 # radius of Earth in km
    
    dlon = np.radians(lon - lon_stop)
    dlat = np.radians(lat - lat_stop)
    
    # compute square of half the chord length between the two points
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat_stop)) * np.cos(np.radians(lat)) * np.sin(dlon / 2)**2
    
    # compute angular distance in radians
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    dist = R * c * 1000 * 3.28084 # conversion to meters and feet
    return dist    

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

# function to estimate Type I DZ parameters
def estimate_TypeI_zone(xdf):
    # minimum perception reaction time
    def minPerceptionReactionTime(v0):
        prt = round(0.445 + 21.478/v0, 2)
        return prt
    
    # maximum deceleration rate    
    def maxDecelerationRate(v0):
        dec = round(np.exp(3.379 - 36.099/v0) - 9.722 + 429.692/v85, 2)
        return -dec
    
    # maximum acceleration rate
    def maxAccelerationRate(v0):
        acc = round(-27.91 + 760.258/v0 + 0.266*v85, 2)
        return acc
    
    # minimum stopping distance
    def minStoppingDistance(v0, prt, dec):
        Xs = round(v0*prt + (v0**2) / (2*(abs(dec))), 0)
        return Xs
    
    # maximum clearing distance
    def maxClearingDistance(v0, prt, acc):
        Xc = round(v0*yellow_interval + 0.5*(acc)*((yellow_interval - prt)**2), 0)
        return Xc
    
    # IDs of stopped trips
    zdf = xdf.copy()
    stopped_trips = list(zdf[zdf.speed <= 5].TripID.unique())
    
    # add stop/run decision
    zdf.loc[zdf.TripID.isin(stopped_trips), 'decision'] = 0
    zdf.loc[~zdf.TripID.isin(stopped_trips), 'decision'] = 1
    
    # filter observations at the yellow onset
    zdf = zdf[(xdf.TUY == 0) | (xdf.TAY == 0)]
    
    # filter observations with speed greater than threshold
    zdf = zdf[zdf.speed >= 15]
    
    # convert velocity from mph to ft/s
    zdf.speed = round(zdf.speed * 5280/3600, 1)
    
    # compute perception-reaction time, max deceleration, max acceleration
    zdf['PRT'] = minPerceptionReactionTime(zdf.speed)
    zdf['deceleration'] = maxDecelerationRate(zdf.speed)
    zdf['acceleration'] = maxAccelerationRate(zdf.speed)
    
    # compute zone vehicle's position is in
    zdf.loc[(((zdf.Xi <= zdf.Xc) & (zdf.Xc <= zdf.Xs)) | ((zdf.Xi <= zdf.Xs) & (zdf.Xs <= zdf.Xc))), 'zone'] = 'should-go'
    zdf.loc[(((zdf.Xi >= zdf.Xc) & (zdf.Xc >= zdf.Xs)) | ((zdf.Xi >= zdf.Xs) & (zdf.Xs >= zdf.Xc))), 'zone'] = 'should-stop'
    zdf.loc[((zdf.Xc < zdf.Xi) & (zdf.Xi < zdf.Xs)), 'zone'] = 'dilemma'
    zdf.loc[((zdf.Xs < zdf.Xi) & (zdf.Xi < zdf.Xc)), 'zone'] = 'option'
    
    # covert velocity from ft/s to mph
    zdf.speed = round(zdf.speed * 3600/5280, 1)
    
    return zdf

# function to process trajectory and signal data
def process_trajectory_signal_data(day, direction):
    # filter trips for day
    ddf = wdf.copy()[wdf.localtime.dt.day == day]
    
    # specify coordinates for direction
    if direction == 'WB':
        lon = [coord['WB']['lon_stop'], coord['WB']['lon_adv']] # stop and adv longitudes
        lat_stop, lon_stop = coord['WB']['lat_stop'], coord['WB']['lon_stop'] # inputs to Haversine function
    elif direction == 'EB':
        lon = [coord['EB']['lon_stop'], coord['EB']['lon_adv']] # stop and adv longitudes
        lat_stop, lon_stop = coord['EB']['lat_stop'], coord['EB']['lon_stop'] # inputs to Haversine function
    
    # process trip direction and filter thru trips in given direction
    thru_trips = process_trip_direction(ddf)
    thru_trips_filtered = [key for key, value in thru_trips.items() if value == direction]
    
    # filter trips for direction
    fdf = ddf.copy()[ddf.TripID.isin(thru_trips_filtered)]
    
    # drop direction column
    fdf.drop('direction', axis = 1, inplace = True)
    
    # set localtime as index for interpolating trip attributes
    fdf.set_index('localtime', inplace = True)
    
    # apply the interpolation function to each tripID
    idf = fdf.groupby('TripID').apply(interpolate_trip)
    
    # groupby adds an extra level to the index, so reset index
    idf.index = idf.index.get_level_values(1)
    
    # reset localtime as column
    idf.reset_index(inplace = True)
    idf.rename(columns = {'index': 'localtime'}, inplace = True)
    
    # filter trips between intersection stop line and advance location
    idf = idf[idf.Longitude.between(min(lon), max(lon), inclusive = 'both')]
    
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
    yellow_trips = list(cdf[cdf.Signal == 'Y']['TripID'].unique())
    cdf = cdf[cdf.TripID.isin(yellow_trips)]
    
    # compute distance from intersection stop line using Haversine formula
    cdf['Xi'] = cdf.apply(lambda row: haversine(row['Latitude'], row['Longitude'], lat_stop, lon_stop), axis = 1)
    cdf.Xi = cdf.Xi.round(0)
    
    # remove redundant cols and rows with nan values
    cdf.drop('yellowtime', axis = 1, inplace = True)
    cdf.dropna(subset = ['TripID', 'Signal'], inplace = True)
    
    # estimate Type I DZ parameters and zones
    zdf = estimate_TypeI_zone(cdf)
    
    return {'cdf': cdf, 'zdf': zdf}

# =============================================================================
# filter Wejo trips and identify approach directions
# =============================================================================

# common days for which trajectory and signal data are available
days = set(wdf.localtime.dt.day.unique()).intersection(mdf.localtime.dt.day.unique())

list_zdf, list_cdf = [], [] # store processed data for each day

# loop through each day and combine trajectory-signal data
for day in days:
    for direction in ['WB', 'EB']:
        print("Day, approach: ", day, direction)
        
        # process trajectory & signal info and estimate Type I DZ parameters
        data = process_trajectory_signal_data(day, direction)
        zdf, cdf = data['zdf'], data['cdf']
        
        # add direction as variables
        zdf['Approach'] = direction
        cdf['Approach'] = direction
        
        list_zdf += [zdf]
        list_cdf += [cdf]

final_zdf = pd.concat(list_zdf, ignore_index = True)
final_cdf = pd.concat(list_cdf, ignore_index = True)

# save file
output_path = os.path.join()
final_cdf.to_csv("script/test_Speedway_Campbell/data/Wejo/Speedway_Campbell_08_trajectory.txt", index = False, sep = '\t')

# # function to plot trajectory and signal information
# def plotTrajectorySignal(trip_id):
#     # filter df with trajectory and signal info for trip id
#     tdf = cdf.copy()[cdf.TripID == trip_id]

#     sig_color = {'G': 'green', 'Y': 'orange', 'R': 'red'}
    
#     fig = px.scatter(
#         tdf,
#         x = 'Xi',
#         y = 'localtime',
#         color = 'Signal',
#         hover_data = ['TripID', 'LocationID'],
#         color_discrete_map = sig_color
#     )
    
#     fig.update_traces(marker = dict(size = 10))
#     fig.show()
