library(data.table)
library(ggplot2)

# load results of accuracy evaluation
df <- fread("ignore/DZ_estimation_accuracy.txt", sep = '\t')

method_levels = c('Type I DZ based on proposed method', 
                  'Type I DZ based on ITE parameters', 
                  'Type II DZ based on travel time to stop', 
                  'Type II DZ based on stopping probability')
group_levels = c('Start of dilemma zone (Xc)', 
                 'End of dilemma zone (Xs)')

df[, Method := factor(Method, levels = method_levels)]
df[, Group := factor(Group, levels = group_levels)]

plot <- ggplot(df, aes(Site, RMSE, fill = Method)) + 
    geom_bar(stat = 'identity', position = 'dodge') +
    facet_wrap(~Group, ncol = 1, scale = 'free') + 
    scale_fill_manual(values = c('blue', 'red', 'forestgreen', 'darkmagenta'), labels = method_levels) + 
    scale_y_continuous(breaks = seq(0, 150, 25)) + 
    xlab("Study site") +
    ylab("Root mean squared error (ft)") +
    theme_minimal() +
    theme(axis.text.x = element_text(size = 12, face = 'bold'),
          axis.text.y = element_text(size = 12),
          axis.title = element_text(size = 14, face = 'bold'),
          legend.title = element_blank(),
          legend.text = element_text(size = 12),
          legend.position = 'top',
          legend.box = 'vertical',
          legend.spacing = unit(0, 'cm'),
          legend.box.margin = margin(0, 0, 0, 0, 'cm'),
          panel.grid.major.x = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_rect(color = 'black', fill = NA),
          strip.text = element_text(size = 12, face = 'bold'),
          plot.background = element_rect(fill = 'white', color = 'NA')) + 
    guides(fill = guide_legend(ncol = 2))
plot

ggsave("output/DZ_estimation_accuracy.png",
       plot = plot,
       units = "cm",
       width = 30,
       height = 20,
       dpi = 1200)
