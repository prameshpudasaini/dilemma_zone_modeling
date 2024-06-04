# test carried out at Speedway Blvd & Campbell Ave

import os
import pandas as pd

os.chdir(r"D:\GitHub\dilemma_Wejo")

# list of Wejo trajectory data
wejo_path = os.path.join("script/test_Speedway_Campbell/data/Wejo_Aug_Nov")
files_wejo = os.listdir(wejo_path)

month = 8 # month to filter data for
ldf = [] # store filtered df

# loop through each file and filter data start & end times
for file in files_wejo:
    xdf = pd.read_csv(os.path.join(wejo_path, file))
    
    # convert localtime to datetime format
    xdf.localtime = pd.to_datetime(xdf.localtime)

    # filter trips between start and end times
    xdf = xdf[xdf.localtime.dt.month == month]

    # add df to list
    ldf += [xdf]

# combine list of df
df = pd.concat(ldf, ignore_index = True)

# remove redundant columns
df.drop(['Altitude', 'unixtime', 'accuracy'], axis = 1, inplace = True)

# save file
output_file = "Speedway_Campbell_" + str(month).zfill(2) + ".txt"
output_path = os.path.join("script/test_Speedway_Campbell/data/Wejo", output_file)
df.to_csv(output_path, index = False, sep = '\t')
