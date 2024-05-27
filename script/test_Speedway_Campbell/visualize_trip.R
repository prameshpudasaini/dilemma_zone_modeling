library(data.table)
library(leaflet)

# function to round direction to nearest 0, 90, 180, 270, 360
round_direction <- function(direction) {
    target_values = c(0, 90, 180, 270, 360)
    
    # find the closest target value for each direction
    sapply(direction, function(x) {
        target_values[which.min(abs(target_values - x))]
    })
}

# read data
DT <- fread("script/test_Speedway_Campbell/data/output_dt=2021-08-17.csv")

# count number of unique trips
length(unique(DT$TripID))

# summarize direction of trips
summary(DT$direction)

# compute average and standard deviation of each trip's direction
DT[, dirc_avg := round(mean(direction), 0), by = .(TripID)]
DT[, dirc_sd := round(sd(direction), 2), by = .(TripID)]

# summarize average and standard deviation of each trip's direction
summary(DT$dirc_avg)
summary(DT$dirc_sd)

# visualize average and standard deviation of each trip's direction
plot(DT$dirc_avg, DT$dirc_sd)

# filter trips with standard deviation of trip direction < 2
DT <- DT[dirc_sd <= 2, ]
plot(DT$dirc_avg, DT$dirc_sd)

# count number of unique trips after filtering
length(unique(DT$TripID))

# round average direction and update direction
DT[, direction := round_direction(dirc_avg)]

# visualize trip
df <- copy(DT)[TripID == 219]

leaflet(df) |> 
    addTiles() |> 
    addCircleMarkers(
        lng = ~Longitude, 
        lat = ~Latitude,
        radius = 5,
        label = ~LocationID
    )
