import os
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\dilemma_Wejo")

# load DZ analysis dataset
ddf = pd.read_csv("ignore/Wejo/trips_stop_go/data_Xs_Xc.txt", sep = '\t')

# compute zone vehicle's position is in
ddf.loc[(((ddf.Xi <= ddf.Xc) & (ddf.Xc <= ddf.Xs)) | ((ddf.Xi <= ddf.Xs) & (ddf.Xs <= ddf.Xc))), 'zone'] = 'Should go'
ddf.loc[(((ddf.Xi >= ddf.Xc) & (ddf.Xc >= ddf.Xs)) | ((ddf.Xi >= ddf.Xs) & (ddf.Xs >= ddf.Xc))), 'zone'] = 'Should stop'
ddf.loc[((ddf.Xc < ddf.Xi) & (ddf.Xi < ddf.Xs)), 'zone'] = 'Dilemma'
ddf.loc[((ddf.Xs < ddf.Xi) & (ddf.Xi < ddf.Xc)), 'zone'] = 'Option'

print(f"{ddf.zone.value_counts()}")
print(f"{ddf.groupby('zone')['Decision'].value_counts()}")
print(f"{ddf.groupby('zone')['Group'].value_counts()}")
print(f"{ddf.groupby(['Node', 'zone'])['Decision'].value_counts()}")

# =============================================================================
# comparison with FHWA method
# =============================================================================

def modelFHWA(v0, node):
    # specify v85 (speed limit) and yellow interval based on node
    if node == 540:
        v85, y = 30, 3.5
    elif node in [217, 444, 446]:
        v85, y = 35, 4
    elif node in [586, 618]:
        v85, y = 40, 4

    v0, v85 = v0*5280/3600, v85*5280/3600 # mph to ft/s

    prt = 1
    dec = 10
    acc = (16 - 0.213 * v0)
    
    Xs = v0*prt + ((v0**2) / (2*dec))
    Xc = v0*y + 0.5*(acc)*((y - prt)**2)
    
    return {'Xs': Xs, 'Xc': Xc}

ddf['Xs_FHWA'] = ddf.apply(lambda x: modelFHWA(x.speed, x.Node)['Xs'], axis = 1)
ddf['Xc_FHWA'] = ddf.apply(lambda x: modelFHWA(x.speed, x.Node)['Xc'], axis = 1)

print(f"Correlation Xs and Xs_FHWA: {ddf.Xs.corr(ddf.Xs_FHWA)}")
print(f"Correlation Xc and Xc_FHWA: {ddf.Xc.corr(ddf.Xc_FHWA)}")

# =============================================================================
# comparison with Li's method
# =============================================================================

def modelLi2013(v0, node):
    # specify v85 (speed limit) and yellow interval based on node
    if node == 540:
        v85, y = 30, 3.5
    elif node in [217, 444, 446]:
        v85, y = 35, 4
    elif node in [586, 618]:
        v85, y = 40, 4

    v0, v85 = v0*5280/3600, v85*5280/3600 # mph to ft/s

    prt = 0.274 + 30.392/v0
    dec = np.exp(3.572 - 25.013/v0) - 17.855 + 480.558/v85
    acc = -23.513 + 658.948/v0 + 0.223*v85
    
    Xs = v0*prt + ((v0**2) / (2*dec))
    Xc = v0*y + 0.5*(acc)*((y - prt)**2)
    
    return {'Xs': Xs, 'Xc': Xc}

ddf['Xs_Li'] = ddf.apply(lambda x: modelLi2013(x.speed, x.Node)['Xs'], axis = 1)
ddf['Xc_Li'] = ddf.apply(lambda x: modelLi2013(x.speed, x.Node)['Xc'], axis = 1)

print(f"Correlation Xs and Xs_Li: {ddf.Xs.corr(ddf.Xs_Li)}")
print(f"Correlation Xc and Xc_Li: {ddf.Xc.corr(ddf.Xc_Li)}")

# plot of Xs vs Xs_Li
px.scatter(
    ddf,
    x = 'Xs',
    y = 'Xs_Li',
    facet_col = 'Node'
).update_traces(marker = dict(size = 7, symbol = 'circle')).show()

# plot of Xc vs Xc_Li
px.scatter(
    ddf,
    x = 'Xc',
    y = 'Xc_Li',
    facet_col = 'Node'
).update_traces(marker = dict(size = 7, symbol = 'circle')).show()
