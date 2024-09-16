import os
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\dilemma_zone_modeling")

df = pd.read_csv("ignore/processed_trips_old.txt", sep = '\t')
df.localtime = pd.to_datetime(df.localtime)
df = df[df.localtime.dt.day == 1]

fts_trips = list(df[df.Group == 'FTS'].TripID.unique())
ylr_trips = list(df[df.Group == 'YLR'].TripID.unique())
rlr_trips = list(df[df.Group == 'RLR'].TripID.unique())

tdf = df.copy()[df.TripID == int(fts_trips[22])]
tdf = df.copy()[df.TripID == int(ylr_trips[0])]
tdf = df.copy()[df.TripID == int(rlr_trips[0])]

px.scatter(
    tdf,
    x = 'Xi',
    y = 'localtime',
    color = 'Signal',
    color_discrete_map = {'G': 'green', 'Y': 'orange', 'R': 'red'}
)

fts_trip = 1361230
ylr_trip = 171
rlr_trip = 118

signal_colors = {'G': 'green', 'Y': 'orange', 'R': 'red'}
df['signal_color'] = df.Signal.map(signal_colors)

fdf = df.copy()[df.TripID == fts_trip]
ydf = df.copy()[df.TripID == ylr_trip]
rdf = df.copy()[df.TripID == rlr_trip]

m = 15

# create subplots of Xi vs speed for FTS and YLR
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12, 6))

ax1.scatter(fdf.Xi, fdf.localtime, c = fdf.signal_color, s = m)
ax1.set_title('(a) First to stop (FTS)', fontsize = 13, fontweight = 'bold')
ax1.set_xlabel('Distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
ax1.set_ylabel('Timestamp', fontsize = 12, fontweight = 'bold')
ax1.tick_params(axis = 'both', which = 'major', labelsize = 12)

ax2.scatter(ydf.Xi, ydf.localtime, c = ydf.signal_color, s = m)
ax2.set_title('(b) Yellow light runner (YLR)', fontsize = 13, fontweight = 'bold')
ax2.set_xlabel('Distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
ax2.set_ylabel('Timestamp', fontsize = 12, fontweight = 'bold')
ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)

ax3.scatter(rdf.Xi, rdf.localtime, c = rdf.signal_color, s = m)
ax3.set_title('(c) Red light runner (RLR)', fontsize = 13, fontweight = 'bold')
ax3.set_xlabel('Distance from stop line (ft)', fontsize = 12, fontweight = 'bold')
ax3.set_ylabel('Timestamp', fontsize = 12, fontweight = 'bold')
ax3.tick_params(axis = 'both', which = 'major', labelsize = 12)

plt.tight_layout(pad = 1)
plt.savefig('output/sample_trajectories_FTS_YLR_RLR.png', dpi = 1200)
plt.show()