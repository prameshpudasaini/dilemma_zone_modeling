# test carried out at Speedway Blvd & Campbell Ave

import os
import pandas as pd
import dask.dataframe as dd

os.chdir(r"E:\Wejo\Wejo-Aug-Nov 2021")

# read Wejo trajectory data for August 17, 2021
file_name = "output_dt=2021-08-17.csv"
df = dd.read_csv(file_name, assume_missing = True)

# get unique trip IDs
all_trips = set(df.TripID.unique())

# specify coordinate gates for filtering trip trajectories at Speedway Blvd & Campbell Ave
gate = {'EB': {'adv': [32.23626426244028, -110.94667104330036], 'ent': [32.23592026475734, -110.94348358193795]},
        'WB': {'adv': [32.236255402912555, -110.94133695642331], 'ent': [32.2358688818948, -110.94449404879498]}}

# store trip IDs passing through coordinate gates for each approach
gate_trips = {}

for i in gate.keys():
    # create list of lat and lng for inbound (adv) and outbound (ent) gates
    lat = [gate[i]['adv'][0], gate[i]['ent'][0]]
    lng = [gate[i]['adv'][1], gate[i]['ent'][1]]
    
    # filter trip IDs passing over inbound (adv) and outbound (ent) gates
    ID = list(df[(df.Latitude.between(min(lat), max(lat))) & (df.Longitude.between(min(lng), max(lng)))].TripID.unique())
    gate_trips[i] = ID

# intersection of sets gives IDs of trips passing through gates in each approach
trips = {'EB': sorted(list(set.intersection(all_trips, gate_trips['EB']))),
         'WB': sorted(list(set.intersection(all_trips, gate_trips['WB'])))}

# specify output paths
output_path = {'EB': os.path.join(r"C:\Users\pramesh\Wejo\DZ_test", file_name[:-4] + '_EB.csv'),
               'WB': os.path.join(r"C:\Users\pramesh\Wejo\DZ_test", file_name[:-4] + '_WB.csv')}

# loop through each approach
for i in trips.keys():
    # filter trips
    gdf = df.copy()[df.TripID.isin(trips[i])].compute()
    
    # specify lat and lng of adv and ent gates
    lat = [gate[i]['adv'][0], gate[i]['ent'][0]]
    lng = [gate[i]['adv'][1], gate[i]['ent'][1]]
    
    # filter trips within the gates for i approach
    gdf = gdf[(gdf.Latitude.between(min(lat), max(lat))) & (gdf.Longitude.between(min(lng), max(lng)))]
    
    # save file
    gdf.to_csv(output_path[i], index = False)
