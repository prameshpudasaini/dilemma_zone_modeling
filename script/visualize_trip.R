library(data.table)
library(lubridate)
library(leaflet)

visualize_trip <- function(node, dirc, month, day, trip_id){
    file_path <- paste0("ignore/Wejo/processed_data/", node, "_", dirc, ".txt")
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

visualize_trip(216, 'EB', 8, 1, 258)
