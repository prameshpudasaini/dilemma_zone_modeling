import os
import pandas as pd

os.chdir(r"D:\GitHub\dilemma_zone_modeling")

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
df.to_csv("ignore/Wejo/trips_analysis/processed_trips_filtered.txt", sep = '\t', index = False)
