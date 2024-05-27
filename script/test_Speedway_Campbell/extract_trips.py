# test carried out at Speedway Blvd & Campbell Ave

import os
import pandas as pd
import dask.dataframe as dd

os.chdir(r"E:\Wejo\Wejo-Aug-Nov 2021")

# read Wejo trajectory data for August 17, 2021
file_input = "output_dt=2021-08-17.csv"
df = dd.read_csv(file_input, assume_missing = True)

# get unique trip IDs
all_trips = set(df.TripID.unique())

# specify coordinate gates for filtering trip trajectories
gate = {'g1': {'c1': [32.23630647990695, -110.93969748930743], 'c2': [32.235866420981864, -110.94095037095649]},
        'g2': {'c1': [32.236272521830756, -110.94338137745342], 'c2': [32.23588077273965, -110.94462299506206]},
        'g3': {'c1': [32.236248338346755, -110.9465362446941], 'c2': [32.235817764157865, -110.9477971426006]}}

# store trip IDs passing through coordinate gates for each approach
gate_trips = []

for k in gate.keys():
    # create list of lat and lng for gate k
    lat = [gate[k]['c1'][0], gate[k]['c2'][0]]
    lng = [gate[k]['c1'][1], gate[k]['c2'][1]]
    
    # filter trip IDs passing over gate k and store the result
    trip_k = set(df[(df.Latitude.between(min(lat), max(lat))) & (df.Longitude.between(min(lng), max(lng)))].TripID.unique())
    gate_trips.append(trip_k)
    
# intersection of sets gives IDs of common trips passing through all gates
trips = sorted(list(set.intersection(all_trips, *gate_trips)))

# filter df for trip IDs passing through all gates
gdf = df.copy()[df.TripID.isin(trips)]

# specify min-max lat and lng bounds
lat = [gate['g1']['c1'][0], gate['g3']['c2'][0]]
lng = [gate['g1']['c1'][1], gate['g3']['c2'][1]]

# filter trips within these bounds
gdf = gdf[(gdf.Latitude.between(min(lat), max(lat))) & (gdf.Longitude.between(min(lng), max(lng)))].compute()

# save file
file_output = os.path.join(r"C:\Users\pramesh\Wejo\DZ_test", file_input)
gdf.to_csv(file_output, index = False)
