library(data.table)
library(lubridate)
library(leaflet)

visualize_trip <- function(node, dirc, month, day, trip_id){
    file_path <- paste0("ignore/Wejo/processed_trips2/", node, "_", dirc, ".txt")
    df <- fread(file_path, sep = '\t')
    
    # filter for trip ID, month, day
    tdf <- copy(df)[TripID == trip_id, ]
    tdf[, Month := month(localtime)]
    tdf[, Day := day(localtime)]
    tdf <- tdf[Month == month & Day == day, ]
    
    leaflet(tdf) |> 
        addTiles() |> 
        addCircleMarkers(
            lng = ~Longitude, 
            lat = ~Latitude,
            radius = 5,
            label = ~Xi
        )
}

tdf <- fread("ignore/Wejo/processed_trips1/217_EB.txt", sep = '\t')
tdf = tdf[TripID == 2 & ID == 42, ]

leaflet(tdf) |> 
    addTiles() |> 
    addCircleMarkers(
        lng = ~Longitude, 
        lat = ~Latitude,
        radius = 1
    )

visualize_trip(217, 'EB', 8, 1, 2)
