import os
import pandas as pd

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\dilemma_zone_modeling")

# MaxView files with signal info
input_file = "ignore/MaxView/MaxView_Aug_Sep_Oct_216_217_517_618.txt"
output_path = "ignore/MaxView/html_plots"

# read file, rename columns and update localtime
df = pd.read_csv(input_file, sep = '\t')
df.rename(columns = {'TimeStamp': 'localtime', 'DeviceId': 'node', 'EventId': 'Signal', 'Parameter': 'Approach'}, inplace = True)
df.localtime = pd.to_datetime(df.localtime)

# map event-signal values
event = {1: 'G', 8: 'Y', 10: 'R'}
df.Signal = df.Signal.map(event)

# map parameter-approach values
parameter = {2: 'EB', 4: 'SB', 6: 'WB', 8: 'NB'}
df.Approach = df.Approach.map(parameter)

for node in list(df.node.unique()):
    ndf = df.copy()[df.node == node]
    
    for month in list(ndf.localtime.dt.month.unique()):
        mdf = ndf.copy()[ndf.localtime.dt.month == month]
        
        # compute duration of each signal
        mdf.sort_values(by = ['Approach', 'localtime'], inplace = True)
        mdf['end_time'] = mdf.groupby('Approach')['localtime'].shift(-1)
        
        for day in list(mdf.localtime.dt.day.unique()):
            ddf = mdf.copy()[mdf.localtime.dt.day == day]
            
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
            output_file = str(node) + '_' + str(month).zfill(2) + '_' + str(day).zfill(2) + '.html'
            fig.write_html(os.path.join(output_path, output_file))