import os
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\dilemma_zone_modeling")

# =============================================================================
# parameters
# =============================================================================

# coordinates of intersection stop line for each node
coord = {216: {'EB': [32.23594848472957, -110.95968391457366],
               'WB': [32.236084658548776, -110.95932248680077],
               'SB': [32.236218147733005, -110.95956250328496]},
         217: {'EB': [32.2360052791151, -110.94427292789531],
               'WB': [32.23616488753039, -110.943689345455],
               'NB': [32.23584967340982, -110.94384981257787],
               'SB': [32.236304860262344, -110.9441134794737]},
         517: {'EB': [32.23548007979899, -110.84137985283755],
               'WB': [32.23565215619724, -110.84066583390516],
               'NB': [32.23527288981551, -110.84090500554865],
               'SB': [32.23587333887579, -110.84111296140715]},
         618: {'EB': [32.20641724305155, -110.84132687578969],
               'WB': [32.20658334282132, -110.84063775838241],
               'NB': [32.20621700778104, -110.84084609425115],
               'SB': [32.20677888497246, -110.84110007833344]}}

# dictionaries for mapping values
signal_event = {1: 'G', 8: 'Y', 10: 'R'} # signal phase change events
phase_parameter = {2: 'EB', 4: 'SB', 6: 'WB', 8: 'NB'} # phase parameters
approach_direction = {0: 'NB', 90: 'EB', 180: 'SB', 270: 'WB', 360: 'NB'} # approaching direction

# define thresholds
direction_sd_threshold = 2 # to filter out trips moving to driveways or taking U-turns
stop_speed_threshold = 5 # speed threshold to determine stopping
FTS_threshold = 20 # queueing distance from intersection stop line where a vehicle is first-to-stop

# distances for filtering trajectory points at intersection approach
dist_adv = 500

# =============================================================================
# functions
# =============================================================================    

# function to seggregate multiple sub trips across intersection within a trip
def segregate_sub_trips(dwdf):
    total_trips = len(dwdf.TripID.unique())
    trip_dfs = [] # list to hold resulting segregated trip dfs
    
    # loop through each trip and add to trip dfs
    for trip in list(dwdf.TripID.unique()):
        # filter for trip
        xdf = dwdf.copy()[dwdf.TripID == trip]
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
            
    seg_wdf = pd.concat(trip_dfs, ignore_index = True)
    
    return seg_wdf


# function to round trip direction
def round_direction(dirc):
    target_values = np.array([0, 90, 180, 270, 360])
    
    # find the closest target value for each direction
    return target_values[np.abs(target_values - dirc).argmin()]

# function to filter thru trips based on deviation in direction
def filter_thru_trips(seg_wdf, dirc):
    # compute mean and standard deviation of each trip's direction
    adf = seg_wdf.groupby('TripID')['direction'].agg(dirc_mean = 'mean', dirc_std = 'std').reset_index()
    
    # drop rows with Nan values for st dev
    adf.dropna(subset = ['dirc_std'], axis = 0, inplace = True)
    
    # high st dev of trip direction indicates moving to driveways or taking U-turns
    # filter trips with st dev of trip direction < threshold
    adf = adf[adf.dirc_std <= direction_sd_threshold]
    
    # round average direction and update direction
    adf.dirc_mean = adf.dirc_mean.apply(round_direction)
    
    # map approach based on average direction and filter for input direction
    adf['approach'] = adf.dirc_mean.map(approach_direction)
    adf = adf[adf.approach == dirc]
    
    thru_trips = list(adf.TripID.unique())
    thru_wdf = seg_wdf.copy()[seg_wdf.TripID.isin(thru_trips)]    
    
    return thru_wdf # segregated trajectory dataset filtered for thru trips


# function to interpolate values for each trip ID
def interpolate_trip(trip_df):
    # create a new date range with millisecond intervals
    new_index = pd.date_range(start = trip_df.index.min(), end = trip_df.index.max(), freq='100ms')
    
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

# function to process Wejo trajectory data
def process_trajectory_data(dwdf, node, dirc):    
    # seggregate multiple sub trips across intersection within a trip
    seg_wdf = segregate_sub_trips(dwdf)
    
    # check length of original and segregated trajectory data are equal
    if len(dwdf) != len(seg_wdf):
        print("Trip segregation ERROR!")
    
    # filter thru trips for input direction
    thru_wdf = filter_thru_trips(seg_wdf, dirc)
    thru_wdf.drop('direction', axis = 1, inplace = True) # drop direction column
    
    # set localtime as index for interpolating trip attributes
    thru_wdf.set_index('localtime', inplace = True)
    
    # apply the interpolation function to each tripID
    idf = thru_wdf.groupby('TripID')[list(thru_wdf.columns)].apply(interpolate_trip)
    
    # groupby adds an extra level to the index, so reset index
    idf.index = idf.index.get_level_values(1)
    
    # reset localtime as column
    idf.reset_index(inplace = True)
    idf.rename(columns = {'index': 'localtime'}, inplace = True)
    
    # compute distance of each trajectory point from intersection stop line using Haversine formula
    coord_stop = coord[node][dirc] # coordinates of stop line for input node and direction
    idf['Xi'] = idf.apply(lambda row: haversine(row['Latitude'], row['Longitude'], coord_stop[0], coord_stop[1], dirc), axis = 1)
    
    # filter out short incompleted trips resulting from trips segregation
    # get min, max of each trip's distance from intersection stop line
    tddf = idf.groupby(['TripID']).agg({'Xi': ['min', 'max']}).reset_index() # trip dist df
    tddf.columns = ['TripID', 'Xi_cross', 'Xi_adv']
    
    # filter complete trips with trajectories between adv and crossing points
    dist_cross = 50 if node in [216, 217] else 40
    tddf = tddf[(tddf.Xi_adv >= dist_adv) & (tddf.Xi_cross <= -(dist_cross))]
    idf = idf[idf.TripID.isin(list(tddf.TripID.unique()))]
    
    # filter interpolated trajectory waypoints between adv and cross
    idf = idf[idf.Xi.between(-dist_cross, dist_adv, inclusive = 'both')]
    
    return idf


# =============================================================================
# process Wejo trips for each day and direction
# =============================================================================

# read and process MaxView signal data
signal_data = pd.read_csv("ignore/MaxView/MaxView_Aug_Sep_Oct_216_217_517_618.txt", sep = '\t')
signal_data.rename(columns = {'TimeStamp': 'localtime', 'EventId': 'Signal'}, inplace = True) # rename columns
signal_data.localtime = pd.to_datetime(signal_data.localtime) # timestamp conversion to datetime
signal_data.Signal = signal_data.Signal.map(signal_event) # map signal indication
signal_data.Parameter = signal_data.Parameter.map(phase_parameter) # map phase parameter

for node in [217]:
    directions = coord[node].keys() # unique directions in node
    
    for dirc in ['EB']:
        # filter MaxView signal data for input node and direction
        mdf = signal_data.copy()[(signal_data.DeviceId == node) & (signal_data.Parameter == dirc)]
        mdf.drop(['DeviceId', 'Parameter'], axis = 1, inplace = True) # drop redundant columns
        
        # read Wejo trajectory data for input node and direction
        path_wejo = os.path.join("ignore/Wejo/raw_data_compiled", str(node) + "_" + dirc + ".txt")
        wdf = pd.read_csv(path_wejo, sep = '\t') # read Wejo data
        wdf.localtime = pd.to_datetime(wdf.localtime) # timestamp conversion to datetime
        
        # Northbound trips have two directions: 0 and 360
        # update direction by argmin of (direction, 360 - direction)
        if dirc == 'NB':
            wdf['direction360'] = 360 - wdf.direction
            wdf.direction = wdf[['direction', 'direction360']].min(axis = 1)
            wdf.drop('direction360', axis = 1, inplace = True)
            
        list_cdf = [] # store processed/combined trajectory+signal data for each node and dirc
            
        months = list(wdf.localtime.dt.month.unique()) # unique months with data available
        for month in [8]:
            print(f"\nNode, direction, month: {node}, {dirc}, {month}")
            
            # filter signal and trajectory data for input month
            mmdf = mdf.copy()[mdf.localtime.dt.month == month] # month mdf
            mwdf = wdf.copy()[wdf.localtime.dt.month == month] # month wdf
            
            # find common days for which trajectory and signal data are available
            days = set(mwdf.localtime.dt.day.unique()).intersection(mmdf.localtime.dt.day.unique())
            
            # loop through each day
            for day in days:
                
                # filter signal and trajectory data for input day
                dmdf = mmdf.copy()[mmdf.localtime.dt.day == day] # day mdf
                dwdf = mwdf.copy()[mwdf.localtime.dt.day == day] # day wdf
                
                # generate new TripIDs
                dwdf['ID'] = dwdf.TripID
                dwdf.TripID = pd.factorize(dwdf.TripID)[0] + 1
                
                # process trajectory and signal data
                idf = process_trajectory_data(dwdf, node, dirc)
                
                # append combined df to list
                list_cdf.append(idf)

        # concatenate data in list for input node and dirc and save file
        final_df = pd.concat(list_cdf, ignore_index = True)
        path_output = os.path.join("ignore/Wejo/processed_trips2", str(node) + "_" + dirc + ".txt")
        final_df.to_csv(path_output, index = False, sep = '\t')