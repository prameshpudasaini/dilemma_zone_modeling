library(data.table)
library(ggplot2)

# load count dataset for node, zone, group
cdf <- fread("ignore/Wejo/trips_stop_go/count_node_zone_group.txt", sep = '\t')
cdf[, percent := percent*100]

node_levels = c('6th & Euclid', 'Prince & 1st', 'Prince & Campbell', 'Speedway & Campbell', 'Broadway & Kolb', '22nd & Kolb')
cdf[Node == 540, Site := node_levels[1]]
cdf[Node == 444, Site := node_levels[2]]
cdf[Node == 446, Site := node_levels[3]]
cdf[Node == 217, Site := node_levels[4]]
cdf[Node == 586, Site := node_levels[5]]
cdf[Node == 618, Site := node_levels[6]]
cdf[, Site := factor(Site, levels = node_levels)]

zone_levels <- c('Should stop', 'Should go', 'Option')
group_levels = c('FTS', 'YLR', 'RLR')

cdf[, zone := factor(zone, levels = zone_levels)]
cdf[, Group := factor(Group, levels = group_levels)]

plot <- ggplot(cdf, aes(zone, percent, fill = Group)) + 
    geom_bar(stat = 'identity') +
    facet_wrap(~Site, ncol = 6) + 
    scale_fill_manual(values = c('grey30', 'orange', 'red'), labels = group_levels) + 
    xlab("Type I decision zone") +
    ylab("Percentage of vehicles") +
    theme_minimal() +
    theme(axis.text.x = element_text(size = 12, angle = 45, hjust = 1),
          axis.text.y = element_text(size = 12),
          axis.title = element_text(size = 14, face = 'bold'),
          legend.title = element_blank(),
          legend.text = element_text(size = 12),
          legend.position = 'top',
          legend.box = 'vertical',
          legend.spacing = unit(0, 'cm'),
          legend.box.margin = margin(0, 0, 0, 0, 'cm'),
          panel.border = element_rect(color = 'black', fill = NA),
          strip.text = element_text(size = 12, face = 'bold'),
          plot.background = element_rect(fill = 'white', color = 'NA'))
plot

# ggsave("output/percentage_vehicles_node_zone.png",
#        plot = plot,
#        units = "cm",
#        width = 29.7,
#        height = 21/2,
#        dpi = 600)

# # load dilemma zone dataset
# df <- fread("ignore/Wejo/trips_stop_go/data_DZ_analysis.txt", sep = '\t')

# df[Decision == 1, Dec := 'Actual decision taken: stop']
# df[Decision == 0, Dec := 'Actual decision taken: go']
# 
# df[Group == 'RLR', is_RLR := TRUE]
# df[Group != 'RLR', is_RLR := FALSE]
# 
# zone_levels <- c('Should stop', 'Should go', 'Dilemma', 'Option')
# df[, zone := factor(zone, levels = zone_levels)]
# 
# col_stop <- 'black'
# col_go <- 'forestgreen'
# col_dilemma <- 'blue'
# col_option <- 'orange'
# 
# ggplot(df) + 
#     geom_point(aes(Xi, speed, color = zone, shape = is_RLR)) + 
#     facet_wrap(~Dec) + 
#     scale_color_manual(values = c(col_stop, col_go, col_dilemma, col_option),
#                        labels = c('Should-stop', 'Should-go', 'Dilemma', 'Option')) +
#     xlab("Yellow onset distance from stop line (ft)") +
#     ylab("Yellow onset speed (mph)") +
#     theme_minimal() + 
#     theme(axis.text = element_text(size = 14),
#           axis.title = element_text(size = 16, face = 'bold'),
#           legend.title = element_text(size = 16, face = 'bold'),
#           legend.text = element_text(size = 14),
#           legend.position = 'top',
#           legend.box = 'vertical',
#           legend.spacing = unit(0, 'cm'),
#           legend.box.margin = margin(0, 0, 0, 0, 'cm'),
#           panel.border = element_rect(color = 'black', fill = NA),
#           strip.text = element_text(size = 16, face = 'bold'),
#           plot.background = element_rect(fill = 'white', color = 'NA'))
