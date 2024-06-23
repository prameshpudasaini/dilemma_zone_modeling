import os
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\dilemma_Wejo")

# MaxView files with signal info
input_path = "ignore/MaxView/raw_data"
input_files = os.listdir(input_path)
output_path = "ignore/MaxView/html_plots"

for file in input_files:
    print(f"Processing file: {file}")
    
    # read signal data
    df = pd.read_csv(os.path.join(input_path, file), sep = '\t')
    
    # rename columns
    df.rename(columns = {'TimeStamp': 'localtime', 'EventId': 'Signal', 'Parameter': 'Approach'}, inplace = True)
    
    # convert timestamp to datetime
    df.localtime = pd.to_datetime(df.localtime)
    
    # map event-signal values
    event = {1: 'G', 8: 'Y', 10: 'R'}
    df.Signal = df.Signal.map(event)
    
    # map parameter-approach values
    parameter = {2: 'EB', 4: 'SB', 6: 'WB', 8: 'NB'}
    df.Approach = df.Approach.map(parameter)
    
    # compute duration of each signal
    df.sort_values(by = ['Approach', 'localtime'], inplace = True)
    df['end_time'] = df.groupby('Approach')['localtime'].shift(-1)
    df['duration'] = round((df.end_time - df.localtime).dt.total_seconds(), 1)
    
    # check signal times
    gdf = df.copy()[df.Signal == 'G']
    ydf = df.copy()[df.Signal == 'Y']
    rdf = df.copy()[df.Signal == 'R']
    
    # get list of unique days for which signal data is available
    days = list(df.localtime.dt.day.unique())
    
    for day in days:
        print("Day: ", day)
        
        # filter signal data for given day
        ddf = df.copy()[df.localtime.dt.day == day]
        
        # plot signal duration info for each approach
        fig = px.timeline(
            ddf,
            x_start = 'localtime',
            x_end = 'end_time',
            y = 'Approach',
            color = 'Signal',
            color_discrete_map = {'G': 'green', 'Y': 'orange', 'R': 'red'},
            category_orders = {'Approach': ['EB', 'WB', 'NB', 'SB']}
        )
        
        # save plot as html file
        output_file = file[:-4] + '_' + str(day).zfill(2) + '.html'
        fig.write_html(os.path.join(output_path, output_file))