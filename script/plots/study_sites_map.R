library(data.table)
library(leaflet)

df <- fread("ignore/node_geometry_v2.csv")

df[, Longitude := as.numeric(Longitude)]
df[, Latitude := as.numeric(Latitude)]
df[, Speed_limit := as.factor(as.character(Speed_limit))]

df <- df[, .(Name, Longitude, Latitude, Speed_limit)]
df <- unique(df)

x <- 45 # icon width & height
speed_limit_icons <- iconList(
    '30' = makeIcon(iconUrl = "https://maps.google.com/mapfiles/ms/icons/red-dot.png", iconWidth = x, iconHeight = x),
    '35' = makeIcon(iconUrl = "https://maps.google.com/mapfiles/ms/icons/green-dot.png", iconWidth = x, iconHeight = x),
    '40' = makeIcon(iconUrl = "https://maps.google.com/mapfiles/ms/icons/blue-dot.png", iconWidth = x, iconHeight = x)
)

leaflet(df) %>%
    addTiles() %>%
    addMarkers(
        lng = ~Longitude, lat = ~Latitude,
        icon = ~speed_limit_icons[as.character(Speed_limit)]
    ) |> 
    addLegend(
        position = "bottomleft",
        title = "Posted speed limit",
        values = df$Speed_limit,
        labels = c('30', '35', '40'),
        colors = c("red", "green", "blue"),
        opacity = 1
    )
