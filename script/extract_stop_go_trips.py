import os
import pandas as pd

os.chdir(r"D:\GitHub\dilemma_Wejo")

# path and list of files
file_path = "ignore/Wejo/processed_data"
file_list = os.listdir(file_path)

list_df = [] # initialize empty list to store 
# loop through each file and append to list
for file in file_list:
    file_df = pd.read_csv(os.path.join(file_path, file), sep = '\t')
    
    # add node and approach as variables
    node, dirc = file[:3], file[4:6]
    file_df['Node'] = node
    file_df['Approach'] = dirc

    list_df.append(file_df)

# combine list into a single df    
df = pd.concat(list_df, ignore_index = True)

# update localtime to pandas datetime and add day variable
df.localtime = pd.to_datetime(df.localtime)
df['Day'] = df.localtime.dt.day

# filter out Queued, Stopped trips and drop redundant columns
df = df[~df.Group.isin(['Queued', 'Stopped'])]

# update columns order and save file
df = df[['Node', 'Approach', 'ID', 'TripID', 'localtime', 'speed', 'Xi', 'Signal', 'TUY', 'TAY', 'Group', 'Day']]
df.to_csv("ignore/Wejo/trips_stop_go/trips_stop_go.txt", sep = '\t', index = False)
