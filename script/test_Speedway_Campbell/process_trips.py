# test carried out at Speedway Blvd & Campbell Ave

import os
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\dilemma_Wejo")

# specify fixed values for latitude and longitude
fixed_lat, fixed_lon = 32.236153104983266, -110.94370075761138

# function to round trip direction
def round_direction(direction):
    target_values = np.array([0, 90, 180, 270, 360])
    
    # find the closest target value for each direction
    return target_values[np.abs(target_values - direction).argmin()]

# Haversine formula to compute the distance between two points given their latitude and longitude
def haversine(lat, lon):
    R = 6371.0 # radius of Earth in km
    
    dlon = np.radians(lon - fixed_lon)
    dlat = np.radians(lat - fixed_lat)
    
    # compute square of half the chord length between the two points
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(fixed_lat)) * np.cos(np.radians(lat)) * np.sin(dlon / 2)**2
    
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
    
    # round speed to the nearest 0.1 ft/s
    trip_df['speed'] = trip_df['speed'].round(1)
    
    return trip_df

# =============================================================================
# functions to estimate Type I DZ parameters (Wei et al., 2011)
# =============================================================================

# parameters
yellow_interval = 4
v85 = 35 # 85th percentile speed computed using INRIX data
v85 = round(v85 * 5280/3600, 2)

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
    # if acc < 0: acc = 0
    # if acc > 6: acc = 6
    return acc

# minimum stopping distance
def minStoppingDistance(v0, prt, acc):
    Xs = round(v0*prt + (v0**2) / (2*(abs(acc))), 0)
    return Xs

# maximum clearing distance
def maxClearingDistance(v0, prt, acc):
    Xc = round(v0*yellow_interval + 0.5*(acc)*((yellow_interval - prt)**2), 0)
    return Xc

# =============================================================================
# filter Wejo trips and identify approach directions
# =============================================================================

# read Wejo trajectory data for August 17, 2021
file_input = "script/test_Speedway_Campbell/data/output_dt=2021-08-17.csv"
df = pd.read_csv(file_input)

# convert localtime to datetime format
df.localtime = pd.to_datetime(df.localtime)

# filter trips on August 17
start_time, end_time = '2021-08-17 06:00:00', '2021-08-17 16:58:00'
df = df[df.localtime.between(start_time, end_time, inclusive = 'left')]

print("Number of unique trips: ", len(list(df.TripID.unique())))

# summarize direction of trips
df.direction.describe()

# compute mean and standard deviation statistic of each trip's direction
adf = df.groupby('TripID')['direction'].agg(dirc_mean = 'mean', dirc_std = 'std').reset_index()

# summarize average and standard deviation of each trip's direction
adf.dirc_mean.describe()
adf.dirc_std.describe()

# high st dev of trip direction indicates moving to driveways or taking U-turns
# filter trips with st dev of trip direction < 2
adf = adf[adf.dirc_std <= 2]

print("Number of unique trips after filtering: ", len(list(adf.TripID.unique())))

# round average direction and update direction
adf.dirc_mean = adf.dirc_mean.apply(round_direction)

# categorize approach based on average direction
direction_approach = {0: 'NB', 90: 'EB', 180: 'SB', 270: 'WB', 360: 'NB'}
adf['approach'] = adf.dirc_mean.map(direction_approach)
adf.drop(['dirc_mean', 'dirc_std'], axis = 1, inplace = True)

# create a dictionary of filtered trip IDs as keys and approaching direction as values
records = adf.to_dict(orient = 'records')
filtered_trips = {record['TripID']: record['approach'] for record in records}

# filter original df for selected trips and add approaching direction
fdf = df.copy()[df.TripID.isin(list(filtered_trips.keys()))]
fdf['approach'] = fdf.TripID.map(filtered_trips)

# filter trips for westbound direction
fdf = fdf[fdf.approach == 'WB']

print("Number of filtered WB trips: ", len(list(fdf.TripID.unique())))

# specify longitude at intersection stop line and filter trips
fdf = fdf[fdf.Longitude > fixed_lon]

# filter trips before 600 ft
lon_600 = -110.94176506971327
fdf = fdf[fdf.Longitude <= lon_600]

# remove redundant columns
fdf.drop(['Altitude', 'direction', 'unixtime', 'accuracy', 'approach'], axis = 1, inplace = True)

# =============================================================================
# interpolation of values for each trip ID
# =============================================================================

# set localtime as index
fdf.set_index('localtime', inplace = True)

# apply the interpolation function to each tripID
idf = fdf.copy()
idf = idf.groupby('TripID').apply(interpolate_trip)

# groupby adds an extra level to the index, so reset index
idf.index = idf.index.get_level_values(1)

# reset localtime as column
idf.reset_index(inplace = True)
idf.rename(columns = {'index': 'localtime'}, inplace = True)

# =============================================================================
# process signal information
# =============================================================================

# read signal phase data
file_sig = r"D:/GitHub/ETT_ITT/ignore/MaxView/2021_signal_data_Aug_Nov_Main_Kolb_raw.txt"
sdf = pd.read_csv(file_sig, sep = ',')

# filter Speedway Blvd & Campbell Ave (intersection ID = 217)
sdf = sdf[sdf.DeviceId == 217]
sdf.drop('DeviceId', axis = 1, inplace = True)

# convert timestamp to datetime 
sdf.TimeStamp = pd.to_datetime(sdf.TimeStamp)
sdf.sort_values(by = 'TimeStamp', inplace = True)

# filter signal events for August 17, 2021 (Tuesday)
start_date, end_date = '2021-08-17', '2021-08-18'
sdf = sdf[sdf.TimeStamp.between(start_date, end_date, inclusive = 'left')]

# filter signal events between start and end times
sdf = sdf[sdf.TimeStamp.between(start_time, end_time, inclusive = 'left')]

# filter signal events (green = 1, yellow = 8, red = 10)
sig = [1, 8, 10]
sdf = sdf[sdf.EventId.isin(sig)]

# map signal events into green, yellow, red
signal = {1: 'G', 8: 'Y', 10: 'R'}
sdf.EventId = sdf.EventId.map(signal)

# map phase parameter into approach
phase = {2: 'EB', 6: 'WB'}
sdf.Parameter = sdf.Parameter.map(phase)

# rename columns
sdf.rename(columns = {'Parameter': 'approach', 'TimeStamp': 'localtime', 'EventId': 'signal'}, inplace = True)

# filter signal data for westbound direction
sdf = sdf[sdf.approach == 'WB']
sdf.drop('approach', axis = 1, inplace = True)

# =============================================================================
# update trajectory with signal information
# =============================================================================

# append signal info to trajectory data
cdf = pd.concat([idf, sdf], ignore_index = True)
cdf.sort_values(by = 'localtime', inplace = True)

# compute wait time until yellow
cdf.loc[cdf.signal == 'Y', 'yellowtime'] = cdf.localtime

# backward fill yellow time for time until yellow
cdf['TUY'] = cdf.yellowtime.bfill()
cdf.TUY = round((cdf.TUY - cdf.localtime).dt.total_seconds(), 1)

# forward fill yellow time for time after yellow
cdf['TAY'] = cdf.yellowtime.ffill()
cdf.TAY = round((cdf.localtime - cdf.TAY).dt.total_seconds(), 1)

# forward fill signal info
cdf.signal.ffill(inplace = True)

# filter trips facing yellow indication before the intersection stop line
yellow_trips = list(cdf[cdf.signal == 'Y']['TripID'].unique())
cdf = cdf[cdf.TripID.isin(yellow_trips)]

# compute distance from intersection stop line using Haversine formula
cdf['Xi'] = cdf.apply(lambda row: haversine(row['Latitude'], row['Longitude']), axis = 1)
cdf.Xi = cdf.Xi.round(0)

# remove redundant cols and rows with nan values
cdf.drop('yellowtime', axis = 1, inplace = True)
cdf.dropna(subset = ['TripID', 'signal'], inplace = True)

print("Number of trips facing yellow: ", len(list(cdf.TripID.unique())))

# =============================================================================
# compute Type I DZ parameters
# =============================================================================

# find stop/run decision
stopped_trips = list(cdf[cdf.speed <= 5].TripID.unique())
cdf.loc[cdf.TripID.isin(stopped_trips), 'decision'] = 0
cdf.loc[~cdf.TripID.isin(stopped_trips), 'decision'] = 1

# # check stop/run decision
# check = list(cdf[cdf.decision == 1].TripID.unique())

# filter observations at the yellow onset
ddf = cdf.copy()[(cdf.TUY == 0) | (cdf.TAY == 0)]

# compute perception-reaction time, max deceleration, max acceleration
ddf['PRT'] = minPerceptionReactionTime(ddf.speed)
ddf.loc[ddf.decision == 0, 'acceleration'] = maxDecelerationRate(ddf.speed) # stopping
ddf.loc[ddf.decision == 1, 'acceleration'] = maxAccelerationRate(ddf.speed) # running

# compute min stopping and max clearing distances
ddf.loc[ddf.decision == 0, 'Xs'] = minStoppingDistance(ddf.speed, ddf.PRT, ddf.acceleration)
ddf.loc[ddf.decision == 1, 'Xs'] = minStoppingDistance(ddf.speed, ddf.PRT, 10)

ddf.loc[ddf.decision == 1, 'Xc'] = maxClearingDistance(ddf.speed, ddf.PRT, ddf.acceleration)
ddf.loc[ddf.decision == 0, 'Xc'] = maxClearingDistance(ddf.speed, ddf.PRT, 6)

# compute zone vehicle's position is in
ddf.loc[(((ddf.Xi <= ddf.Xc) & (ddf.Xc <= ddf.Xs)) | ((ddf.Xi <= ddf.Xs) & (ddf.Xs <= ddf.Xc))), 'zone'] = 'should-go'
ddf.loc[(((ddf.Xi >= ddf.Xc) & (ddf.Xc >= ddf.Xs)) | ((ddf.Xi >= ddf.Xs) & (ddf.Xs >= ddf.Xc))), 'zone'] = 'should-stop'
ddf.loc[((ddf.Xc < ddf.Xi) & (ddf.Xi < ddf.Xs)), 'zone'] = 'dilemma'
ddf.loc[((ddf.Xs < ddf.Xi) & (ddf.Xi < ddf.Xc)), 'zone'] = 'option'

# covert unit of velocity
ddf.speed = round(ddf.speed * 3600/5280, 0)

temp = cdf.copy()[(cdf.TUY == 0) | (cdf.TAY == 0)]

def plotTrajectorySignal(trip_id):
    # filter df with trajectory and signal info for trip id
    tdf = cdf.copy()[cdf.TripID == trip_id]

    sig_color = {'G': 'green', 'Y': 'orange', 'R': 'red'}
    
    fig = px.scatter(
        tdf,
        x = 'Xi',
        y = 'localtime',
        color = 'signal',
        hover_data = ['TripID', 'LocationID'],
        color_discrete_map = sig_color
    )
    
    fig.update_traces(marker = dict(size = 10))
    fig.show()

for i in range(len(stopped_trips)):
    trip_id = stopped_trips[i]
    plotTrajectorySignal(trip_id)

xdf = fdf.reset_index()
xdf = xdf[xdf.TripID == 87868]
px.scatter(xdf, x = 'Longitude', y = 'localtime').update_traces(marker = dict(size = 10)).show()
