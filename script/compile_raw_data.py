import os
import pandas as pd

os.chdir(r"D:\GitHub\dilemma_Wejo")

# path and folders with raw Wejo trajectory data
wejo_path = os.path.join("ignore/Wejo/raw_data")
wejo_folders = os.listdir(wejo_path)

# loop through each folder, compile data for each day and store
for folder in wejo_folders:
    print(f"Processing folder: {folder}")
    
    # list of files with Wejo trajectory data
    wejo_files = os.listdir(os.path.join(wejo_path, folder))
    
    month = 8 # month to filter data for
    ldf = [] # store filtered df

    # loop through each file and filter data for month
    for file in wejo_files:
        xdf = pd.read_csv(os.path.join(wejo_path, folder, file))
        xdf.localtime = pd.to_datetime(xdf.localtime) # convert localtime to datetime format
        xdf = xdf[xdf.localtime.dt.month == month]
        ldf.append(xdf) # add df to list

    # combine list of df
    df = pd.concat(ldf, ignore_index = True)

    # save file
    output_path = os.path.join("ignore/Wejo/raw_data_compiled", folder + '.txt')
    df.to_csv(output_path, index = False, sep = '\t')