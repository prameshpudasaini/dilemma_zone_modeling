library(data.table)
library(lubridate)
library(leaflet)

# file paths
file_wejo_traj <- "script/test_Speedway_Campbell/data/Wejo/Speedway_Campbell_08.txt"
file_trip_times <- "script/test_Speedway_Campbell/data/Wejo/Speedway_Campbell_08_trip_times.txt"

# read data
wdf <- fread(file_wejo_traj, sep = '\t')
tdf <- fread(file_trip_times, sep = '\t')

# add day column to wejo data
wdf[, Day := mday(localtime)]

plotTrajectory <- function(day, trip_id) {
    df <- copy(wdf)[Day == day & TripID == trip_id]
    
    fig <- leaflet(df) |> 
        addTiles() |> 
        addCircleMarkers(
            lng = ~Longitude, 
            lat = ~Latitude,
            radius = 5,
            label = ~LocationID
        )
    
    return(fig)
}

plotTrajectory(4, 129781)
plotTrajectory(5, 129781)
plotTrajectory(10, 40641)
plotTrajectory(1, 127286)

plotTrajectory(14, 26501)
plotTrajectory(21, 162968)
plotTrajectory(6, 24941)
plotTrajectory(23, 114253)
