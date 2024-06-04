library(data.table)
library(readr)
library(RODBC)

setwd("C:/Users/pramesh/MaxView")

# get SQL connection parameters
source("keys.R")

# load query file
query <- read_file("query_signal_data_by_table.sql")

# get query as data table
DT <- as.data.table(sqlQuery(getSQLConnection('STL4'), query))
options(digits.secs = 3)

# remove device id column
DT$DeviceId <- NULL

# convert timestamp column to character to preserve time zone
DT[, TimeStamp := as.character(TimeStamp)]

# save file
fwrite(DT, "MaxView_08_217.txt", sep = '\t')
