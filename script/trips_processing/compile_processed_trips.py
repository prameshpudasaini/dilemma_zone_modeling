import os
import pandas as pd

os.chdir(r"D:\GitHub\dilemma_zone_modeling")

# =============================================================================
# extract stop/go trips and compile
# =============================================================================

# path and list of files
file_path = "ignore/Wejo/processed_trips"
file_list = os.listdir(file_path)

list_df = [] # initialize empty list to store 
# loop through each file and append to list
for file in file_list:
    print(f"Processing file: {file}")
    file_df = pd.read_csv(os.path.join(file_path, file), sep = '\t')
    
    # filter out Queued, Stopped trips and drop redundant columns
    file_df = file_df[~file_df.Group.isin(['Queued', 'Stopped'])]
    
    # add node and approach as variables
    node, dirc = file[:3], file[4:6]
    file_df['Node'] = node
    file_df['Approach'] = dirc

    list_df.append(file_df)

# combine list into a single df    
df = pd.concat(list_df, ignore_index = True)

# update columns order and save file
df = df[['Node', 'Approach', 'ID', 'TripID', 'localtime', 'speed', 'Xi', 'Signal', 'TUY', 'TAY', 'Group']]
df.to_csv("ignore/Wejo/trips_analysis/processed_trips.txt", sep = '\t', index = False)

# =============================================================================
# count trips
# =============================================================================

# read dataset with stop/go trips
df = pd.read_csv("ignore/Wejo/trips_analysis/processed_trips.txt", sep = '\t')

# update localtime to pandas datetime and add day variable
df.localtime = pd.to_datetime(df.localtime)
df['Month'] = df.localtime.dt.month
df['Day'] = df.localtime.dt.day

# create dataset for counting number of trips
cdf = df.copy()[['Node', 'Approach', 'Month', 'Day', 'TripID', 'Group']]
cdf.drop_duplicates(inplace = True) # drop all duplicates

# count trips by node, direction, group
count_trips = cdf[['Node', 'Approach', 'Group']].value_counts().reset_index()
count_trips = count_trips.pivot(index = ['Node', 'Approach'], columns = 'Group', values = 'count').reset_index()
count_trips.to_csv("ignore/Wejo/trips_analysis/count_trips.csv", index = False)

# count number of trips by group and node
print(cdf['Group'].value_counts().reset_index())
print(cdf['Node'].value_counts().reset_index())