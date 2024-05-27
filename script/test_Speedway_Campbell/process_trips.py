# test carried out at Speedway Blvd & Campbell Ave

import os
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\dilemma_Wejo")

# function to round trip direction
def round_direction(direction):
    target_values = np.array([0, 90, 180, 270, 360])
    
    # find the closest target value for each direction
    return target_values[np.abs(target_values - direction).argmin()]

# =============================================================================
# filter Wejo trips and identify approach directions
# =============================================================================

# read Wejo trajectory data for August 17, 2021
file_input = "script/test_Speedway_Campbell/data/output_dt=2021-08-17.csv"
df = pd.read_csv(file_input)

# update local time with unixtime and convert to US/Arizona timezone
df.localtime = pd.to_datetime(df.unixtime, unit = 's', utc = True).dt.tz_convert('US/Arizona').dt.tz_localize(None)

# filter trips between 6 am & 9 am
start_time, end_time = '2021-08-17 06:00:00', '2021-08-17 09:00:00'
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

# specify longitude at intersection stop line and filter trips
lng_stop = -110.94370197552799
fdf = fdf[fdf.Longitude > lng_stop]

# remove redundant columns
fdf.drop(['Altitude', 'direction', 'unixtime', 'accuracy', 'approach'], axis = 1, inplace = True)

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

# filter signal events between 6 am & 9 am
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
sdf.rename(columns = {'Parameter': 'approach', 'TimeStamp': 'localtime'}, inplace = True)

# filter signal data for westbound direction
sdf = sdf[sdf.approach == 'WB']
sdf.drop('approach', axis = 1, inplace = True)

# =============================================================================
# update trajectory with signal information
# =============================================================================

# append signal info to trajectory data
cdf = pd.concat([fdf, sdf], ignore_index = True)
cdf.sort_values(by = 'localtime', inplace = True)

# compute wait time until yellow
cdf.loc[cdf.EventId == 'Y', 'yellowtime'] = cdf.localtime

# backward fill yellow time for time until yellow
cdf['TUY'] = cdf.yellowtime.bfill()
cdf.TUY = round((cdf.TUY - cdf.localtime).dt.total_seconds(), 1)

# forward fill yellow time for time after yellow
cdf['TAY'] = cdf.yellowtime.ffill()
cdf.TAY = round((cdf.localtime - cdf.TAY).dt.total_seconds(), 1)

# forward fill signal info
cdf.EventId.ffill(inplace = True)

# compute distance from intersection stop line
degree_to_feet = 288200
cdf['distance'] = round((cdf.Longitude - lng_stop) * 288200, 1)

# remove redundant cols and rows with nan values
cdf.drop('yellowtime', axis = 1, inplace = True)
cdf.dropna(subset = ['TripID', 'EventId'], inplace = True)

def plotTrajectorySignal(trip_id):
    # filter df with trajectory and signal info for trip id
    tdf = cdf.copy()[cdf.TripID == trip_id]

    sig_color = {'G': 'green', 'Y': 'orange', 'R': 'red'}
    
    fig = px.scatter(
        tdf,
        x = 'distance',
        y = 'LocationID',
        color = 'EventId',
        hover_name = 'LocationID',
        color_discrete_map = sig_color
    )
    
    fig.update_traces(marker = dict(size = 10))
    fig.show()

# filter trips facing yellow indication before the intersection stop line
yellow_trips = list(cdf[cdf.EventId == 'Y']['TripID'].unique())
cdf = cdf[cdf.TripID.isin(yellow_trips)]

trip_id = 158372
plotTrajectorySignal(trip_id)
