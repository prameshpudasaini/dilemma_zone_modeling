import os
import numpy as np
import pandas as pd
import sys

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\dilemma_Wejo")

# =============================================================================
# parameters
# =============================================================================

# coordinates of intersection stop line for each node
coord = {217: {'EB': [32.2360052791151, -110.94427292789531],
               'WB': [32.23616488753039, -110.943689345455],
               'NB': [32.23584967340982, -110.94384981257787],
               'SB': [32.236304860262344, -110.9441134794737]},
         444: {'EB': [32.27200886783614, -110.96122281809403], 
               'WB': [32.27210773687576, -110.96088159609982], 
               'NB': [32.27190216829961, -110.96099999397345], 
               'SB': [32.27221585004281, -110.96109835944922]},
         446: {'EB': [32.272184798264, -110.94398685251169],
               'WB': [32.27225588397434, -110.94363180689612],
               'NB': [32.272066258294764, -110.94374776045905],
               'SB': [32.27237877326103, -110.94386659241245]},
         540: {'EB': [32.22774008013924, -110.95963601486297],
               'WB': [32.227810755655995, -110.95934289252943],
               'NB': [32.22764494955869, -110.95943486700514],
               'SB': [32.227892584009815, -110.95952593507843]},
         586: {'EB': [32.220945248571674, -110.84131125037088],
               'WB': [32.22111729945157, -110.8406919640169],
               'NB': [32.22078422373597, -110.84090435971096],
               'SB': [32.22129888620325, -110.84109548600746]},
         618: {'EB': [32.20641724305155, -110.84132687578969],
               'WB': [32.20658334282132, -110.84063775838241],
               'NB': [32.20621700778104, -110.84084609425115],
               'SB': [32.20677888497246, -110.84110007833344]}}

# function with specification of phase parameters and approach direction
def get_phase_parameter(node):
    if node in [217, 540, 586, 618]:
        phase_parameter = {2: 'EB', 4: 'SB', 6: 'WB', 8: 'NB'}
    elif node in [444, 446]:
        phase_parameter = {2: 'SB', 4: 'WB', 6: 'NB', 8: 'EB'}
    return phase_parameter

# dictionaries for mapping values
approach_direction = {0: 'NB', 90: 'EB', 180: 'SB', 270: 'WB', 360: 'NB'} # approaching direction
signal_event = {1: 'G', 8: 'Y', 10: 'R'} # signal phase change events

# define thresholds
threshold = {'stop_speed': 5, # speed threshold to determine stopping
             'first_to_stop': 20} # queueing distance from intersection stop line where a vehicle is first-to-stop
  
# =============================================================================
# functions
# =============================================================================    

# function to seggregate multiple sub trips across intersection within a trip
def segregate_sub_trips(day_df):
    total_trips = len(day_df.TripID.unique())
    trip_dfs = [] # list to hold resulting segregated trip dfs
    
    # loop through each trip and add to trip dfs
    for trip in list(day_df.TripID.unique()):
        # filter for trip
        xdf = day_df.copy()[day_df.TripID == trip]
        xdf.reset_index(drop = True, inplace = True)
        
        # add trip change indicator variable
        xdf['trip_change'] = abs(xdf.LocationID - xdf.LocationID.shift(1))
        trip_change_index = xdf[xdf['trip_change'] > 1].index.to_list()
        
        # check if there are multiple sub trips for the selected trip
        if len(trip_change_index) >= 1:
            # add 0 as starting point of trip and length of df as the ending point
            trip_split_index = [0] + trip_change_index + [len(xdf)]
            
            # loop through the split points and create sub dfs
            for i in range(len(trip_split_index) - 1):
                start, end = trip_split_index[i], trip_split_index[i + 1]
                sub_df = xdf.copy().iloc[start:end]
                
                # append i and total_trips to the end of trip id (this avoids duplication of trip id)
                sub_df.TripID = (sub_df.TripID.astype(int).astype(str) + str(i) + str(total_trips)).astype(int)
                
                # drop trip change column and append sub df
                sub_df.drop('trip_change', axis = 1, inplace = True)
                trip_dfs.append(sub_df)
        else:
            xdf.drop('trip_change', axis = 1, inplace = True)
            trip_dfs.append(xdf)
            
    seg_df = pd.concat(trip_dfs, ignore_index = True)
    
    return seg_df


# function to round trip direction
def round_direction(dirc):
    target_values = np.array([0, 90, 180, 270, 360])
    
    # find the closest target value for each direction
    return target_values[np.abs(target_values - dirc).argmin()]

# function to filter thru trips based on deviation in direction
def process_trip_direction(xdf, dirc):
    # compute mean and standard deviation of each trip's direction
    adf = xdf.groupby('TripID')['direction'].agg(dirc_mean = 'mean', dirc_std = 'std').reset_index()
    
    # high st dev of trip direction indicates moving to driveways or taking U-turns
    # filter trips with st dev of trip direction < 2
    adf = adf[adf.dirc_std <= 2]
    
    # round average direction and update direction
    adf.dirc_mean = adf.dirc_mean.apply(round_direction)
    
    # map approach based on average direction and filter for input direction
    adf['approach'] = adf.dirc_mean.map(approach_direction)
    adf = adf[adf.approach == dirc]
    
    return list(adf.TripID.unique()) # list of thru trip IDs


# function to interpolate values for each trip ID
def interpolate_trip(trip_df):
    # create a new date range with millisecond intervals
    new_index = pd.date_range(start = trip_df.index.min(), end = trip_df.index.max(), freq='100L')
    
    # reindex the dataframe to include the new date range
    trip_df = trip_df.reindex(new_index)
    
    # interpolate intermediate values
    trip_df['ID'] = trip_df['ID'].interpolate(method = 'nearest')
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
def haversine(lat, lon, lat_ref, lon_ref, dirc):
    R = 6371.0 # radius of Earth in km
    
    dlon = np.radians(lon - lon_ref)
    dlat = np.radians(lat - lat_ref)
    
    # compute square of half the chord length between the two points
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat_ref)) * np.cos(np.radians(lat)) * np.sin(dlon / 2)**2
    
    # compute angular distance in radians
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    dist = round(R * c * 1000 * 3.28084, 0) # conversion to meters and feet
    
    if dirc == 'EB':
        return dist if lon <= lon_ref else -dist
    elif dirc == 'WB':
        return dist if lon > lon_ref else -dist
    elif dirc == 'NB':
        return dist if lat <= lat_ref else -dist
    elif dirc == 'SB':
        return dist if lat > lat_ref else -dist

# # check haversine formula
# haversine(32.235987843702404, -110.94550448858705, 32.2360052791151, -110.94427292789531, 'EB')
# haversine(32.23601278195569, -110.9435258168457, 32.2360052791151, -110.94427292789531, 'EB')

# haversine(32.23615745260398, -110.94271571188392, 32.23616488753039, -110.943689345455, 'WB')
# haversine(32.236180602990636, -110.9444902282016, 32.23616488753039, -110.943689345455, 'WB')

# haversine(32.23518886450097, -110.94387849537277, 32.23584967340982, -110.94384981257787, 'NB')
# haversine(32.23655361994955, -110.94383757608976, 32.23584967340982, -110.94384981257787, 'NB')

# haversine(32.23679840929011, -110.94408554979512, 32.236304860262344, -110.9441134794737, 'SB')
# haversine(32.235756188846494, -110.94411082865686, 32.236304860262344, -110.9441134794737, 'SB')


# function to get distance between intersection stop and cross lines for input node
def node_geometry(node, dirc):
    xdf = pd.read_csv("ignore/node_geometry.csv")
    xdf = xdf[(xdf.ID == node) & (xdf.Approach == dirc)]
    dist_stop_cross = xdf.dist_stop_cross.values[0]
    int_cross_length = xdf.int_cross_length.values[0]
    return {'dist_stop_cross': dist_stop_cross, 'int_cross_length': int_cross_length}

# function to process Wejo trajectory data
def process_trajectory_data(day_wdf, node, dirc):
    print(f"Num of trips: {len(day_wdf.TripID.unique())}")
    
    # seggregate multiple sub trips across intersection within a trip
    seg_df = segregate_sub_trips(day_wdf)
    
    # check length of wdf and seg df are equal
    if len(day_wdf) == len(seg_df):
        print(f"Num of trips after segregation: {len(seg_df.TripID.unique())}")
    else:
        print("Trips segregation ERROR!")
    
    # process trip direction and filter thru trips for input direction
    thru_trips = process_trip_direction(seg_df, dirc)
    ddf = seg_df.copy()[seg_df.TripID.isin(thru_trips)]
    print(f"Num of trips after filtering direction: {len(ddf.TripID.unique())}")
    
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
    
    # compute distance of each trajectory point from intersection stop line using Haversine formula
    coord_stop = coord[node][dirc] # coordinates of stop line for input node and direction
    idf['Xi'] = idf.apply(lambda row: haversine(row['Latitude'], row['Longitude'], coord_stop[0], coord_stop[1], dirc), axis = 1)
    
    # get min, max of each trip's distance from intersection stop line
    tddf = idf.groupby(['TripID']).agg({'Xi': ['min', 'max']}).reset_index() # trip dist df
    tddf.columns = ['TripID', 'Xi_cross', 'Xi_adv']
    
    # get node geometries: dist between cross and stop line, intersection crossing length
    node_geo = node_geometry(node, dirc)
    dist_cross, int_length = node_geo['dist_stop_cross'], node_geo['int_cross_length']
    dist_adv = 500
    
    # filter trips with trajectories between [-twice int cross length, adv] ft from intersection stop line
    tddf = tddf[(tddf.Xi_adv >= dist_adv) & (tddf.Xi_cross <= -(2 * int_length))]
    idf = idf[idf.TripID.isin(list(tddf.TripID.unique()))]
    
    # filter trajectories between [cross, adv]
    idf = idf[idf.Xi.between(-dist_cross, dist_adv, inclusive = 'both')]
    
    # get min, max of each trip's time and compute duration
    ttdf = idf.groupby(['TripID']).agg({'localtime': ['min', 'max']}).reset_index() # trip time df
    ttdf.columns = ['TripID', 'start_time', 'end_time']
    ttdf['duration'] = (ttdf.end_time - ttdf.start_time).dt.total_seconds()
    print(f"Min, max of trips times: {ttdf.duration.min()}, {ttdf.duration.max()}")    
    
    return idf


# function to process combined trajectory and signal data
def process_trajectory_signal_data(x_idf, day_mdf):    
    # append signal info to processed trajectory data
    cdf = pd.concat([x_idf, day_mdf], ignore_index = True)
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
    
    return cdf


# function to identify whether a trip is FTS, YLR, RLR, queued, or stopped multiple times
def identify_FTS_YLR_RLR(x_cdf, trip_id):
    # filter for trip id and select relevant columns
    xdf = x_cdf.copy()[(x_cdf.TripID == trip_id)][['speed', 'Signal', 'Xi']]    
    xdf.reset_index(inplace = True, drop = True)
    
    # create bins of distance from intersection stop line
    xdf['bin'] = xdf.Xi.apply(lambda x: np.ceil(x / 50) * 50)
    
    # compute minimum speed in each bin
    bdf = xdf[xdf.Xi >= 0].groupby('bin')['speed'].agg(min_speed = 'min').reset_index()
    
    # check for stops in each bin and count number of stops    
    bdf['is_stop'] = bdf.min_speed <= threshold['stop_speed']
    num_stops = bdf.is_stop.sum()
    
    # classify the trip into groups: FTS, YLR, RLR, Queued, Stopped
    # for 0 stop, find the signal status corresponding to intersection cross line
    # for 1 stop, find whether the vehicle was FTS or queued
    # ignore >= 2 stops
    if num_stops == 0:        
        group = str(xdf.loc[xdf.Xi.idxmin(), 'Signal']) + 'LR'
    elif num_stops == 1:
        xdf0 = xdf.copy()[xdf.speed <= threshold['stop_speed']]
        xdf0['Xi_10'] = np.ceil(xdf0.Xi / 10) * 10 # round up vehicle position to nearest 10
        mode_stop = int(xdf0.Xi_10.mode().max()) # compute the maximum position of longest stop
        # check if vehicle is within the first-to_stop distance threshold
        group = 'FTS' if mode_stop <= threshold['first_to_stop'] else 'Queued'
    else:
        group = 'Stopped'
        
    return group


# =============================================================================
# process Wejo trips for each day and direction
# =============================================================================

nodes = list(coord.keys())
directions = list(coord[nodes[0]].keys())

# loop through each node
for node in nodes:
    # read MaxView signal data for input node
    path_signal = os.path.join("ignore/MaxView/raw_data", "MaxView_08_" + str(node) + ".txt")
    mdf = pd.read_csv(path_signal, sep = '\t') # read MaxView data
    mdf.rename(columns = {'TimeStamp': 'localtime', 'EventId': 'Signal'}, inplace = True) # rename columns
    mdf.localtime = pd.to_datetime(mdf.localtime) # timestamp conversion to datetime
    mdf.Signal = mdf.Signal.map(signal_event) # map signal indication
    
    # save print statements from console to txt file
    path_console_text = os.path.join("ignore/Wejo/console_text", str(node) + '.txt')
    temp_stdout = sys.stdout
    sys.stdout = open(path_console_text, 'w')
    
    # loop through each direction
    for dirc in directions:
        # read Wejo trajectory data for input node and direction
        path_wejo = os.path.join("ignore/Wejo/raw_data_compiled", str(node) + "_" + dirc + ".txt")
        wdf = pd.read_csv(path_wejo, sep = '\t') # read Wejo data
        wdf.localtime = pd.to_datetime(wdf.localtime) # timestamp conversion to datetimek
        
        # Northbound trips have two directions: 0 and 360
        # update direction by argmin of (direction, 360 - direction)
        if dirc == 'NB':
            wdf['direction360'] = 360 - wdf.direction
            wdf.direction = wdf[['direction', 'direction360']].min(axis = 1)
            wdf.drop('direction360', axis = 1, inplace = True)
        
        # filter signal data for input direction
        mdf1 = mdf.copy() # create copy of signal data for input node
        mdf1.Parameter = mdf1.Parameter.map(get_phase_parameter(node)) # map approaching direction
        mdf1 = mdf1[mdf1.Parameter == dirc] # filter input direction
        mdf1.drop(['DeviceId', 'Parameter'], axis = 1, inplace = True) # drop redundant columns
        
        # find common days for which trajectory and signal data are available
        days = set(wdf.localtime.dt.day.unique()).intersection(mdf1.localtime.dt.day.unique())
        
        list_cdf = [] # store processed/combined trajectory+signal data for each node and dirc
        
        # loop through each day
        for day in days:
            print(f"\nNode, direction, day: {node}, {dirc}, {day}")
            
            # filter signal and trajectory data for input day
            wdf1 = wdf.copy()[wdf.localtime.dt.day == day]
            mdf2 = mdf1.copy()[mdf1.localtime.dt.day == day]
            
            # generate new TripIDs
            wdf1['ID'] = wdf1.TripID
            wdf1.TripID = pd.factorize(wdf1.TripID)[0] + 1
            
            # process trajectory and signal data
            idf = process_trajectory_data(wdf1, node, dirc)
            cdf = process_trajectory_signal_data(idf, mdf2)
            
            group = {} # initiate dict to store trip id and corresponding FTS, YLR, RLR status
            # loop through each trip to compute FTS, YLR, RLR status
            for trip_id in list(cdf.TripID.unique()):
                # print(trip_id)
                group[trip_id] = identify_FTS_YLR_RLR(cdf, trip_id)
                
            # update each trip into FTS, YLR, RLR, Queued, Stopped groups
            cdf['Group'] = cdf.TripID.map(group)
            
            # append combined df to list
            list_cdf.append(cdf)

        # concatenate data in list for input node and dirc and save file
        final_df = pd.concat(list_cdf, ignore_index = True)
        path_output = os.path.join("ignore/Wejo/processed_data", str(node) + "_" + dirc + ".txt")
        final_df.to_csv(path_output, index = False, sep = '\t')
        
    sys.stdout.close()
    sys.stdout = temp_stdout


# # function to plot trajectory and signal information
# def plotTrajectorySignal(trip_id):
#     # filter df with trajectory and signal info for trip id
#     tdf = cdf.copy()[cdf.TripID == trip_id]

#     fig = px.scatter(
#         tdf,
#         x = 'Xi',
#         y = 'localtime',
#         color = 'Signal',
#         hover_data = ['TripID', 'LocationID'],
#         color_discrete_map = {'G': 'green', 'Y': 'orange', 'R': 'red'}
#     )
    
#     fig.update_traces(marker = dict(size = 10))
    
#     lon_stop = 0
    
#     # add vertical line at stop line
#     fig.add_shape(
#         type = 'line',
#         x0 = lon_stop,
#         y0 = tdf['localtime'].min(),
#         x1 = lon_stop,
#         y1 = tdf['localtime'].max(),
#         line = dict(color = 'black', width = 2, dash = 'dash')
#     )
#     fig.show()
    
# for trip_id in list(cdf.TripID.unique()):
#     plotTrajectorySignal(trip_id)
