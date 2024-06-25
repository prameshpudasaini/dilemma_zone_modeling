library(data.table)
library(lubridate)
library(leaflet)

file_path <- "ignore/Wejo/raw_data_compiled/217_EB.txt"
df <- fread(file_path, sep = '\t')

# filter for trip ID
tdf <- copy(df)[TripID == 75667, ]

leaflet(tdf) |> 
    addTiles() |> 
    addCircleMarkers(
        lng = ~Longitude, 
        lat = ~Latitude,
        radius = 5,
        label = ~LocationID
    )
