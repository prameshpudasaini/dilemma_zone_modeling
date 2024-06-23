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

# convert timestamp column to character to preserve time zone
DT[, TimeStamp := as.character(TimeStamp)]

# save file
# fwrite(DT, "MaxView_08_586.txt", sep = '\t') # Broadway & Kolb (40, 40)
# fwrite(DT, "MaxView_08_618.txt", sep = '\t') # 22nd & Kolb (40, 40)
# fwrite(DT, "MaxView_08_444.txt", sep = '\t') # Prince & 1st (35, 35)
# fwrite(DT, "MaxView_08_446.txt", sep = '\t') # Prince & Campbell (35, 35)
# fwrite(DT, "MaxView_08_217.txt", sep = '\t') # Speedway & Campbell (35, 35)
# fwrite(DT, "MaxView_08_540.txt", sep = '\t') # 6th & Euclid (30, 30)
