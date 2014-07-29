# Heatmaps are a great way to display large amounts of information 
# A great place to start is this tutorial by Nathan Yau from Flowing Data  
# http://flowingdata.com/2010/01/21/how-to-make-a-heatmap-a-quick-and-easy-solution/

# Imports
library(plyr)
library(ggplot2)

# Load the data from github
data <- read.csv('https://github.com/mjbommar/scotus-predict/blob/master/model_output/20140515200540/justice_outcome_data.csv?raw=true')

# Generate the justice-year average accuracies and use post-53 rows
year_data <- ddply(data, c("justice", "year"),
                   function(X) mean(X$correct=="True"))
year_data <- year_data[year_data$year>=1953, ]

# Cleanup the justice names for display
year_data$justice_name <- revalue(as.character(year_data$justice),
    c('78'='HLBlack', '79'='SFReed','80'='FFrankfurter',
    '81'='WODouglas','82'='FMurphy','84'='RHJackson',
    '85'='WBRutledge','86'='HHBurton','87'='FMVinson',
    '88'='TCClark','89'='SMinton','90'='EWarren',
    '91'='JHarlan2','92'='WJBrennan','93'='CEWhittaker',
    '94'='PStewart','95'='BRWhite','96'='AJGoldberg',
    '97'='AFortas','98'='TMarshall','99'='WEBurger',
    '100'='HABlackmun','101'='LFPowell','102'='WHRehnquist',
    '103'='JPStevens','104'='SDOConnor','105'='AScalia',
    '106'='AMKennedy','107'='DHSouter','108'='CThomas',
    '109'='RBGinsburg','110'='SGBreyer','111'='JGRoberts',
    '112'='SAAlito','113'='SSotomayor','114'='EKagan'))

#Can we change the order?  Yes but we need to deal with the variable type first
year_data$justice_name <- as.character(year_data$justice_name)
year_data$justice_name <- factor(year_data$justice_name, levels=unique(year_data$justice_name))

# Let's generate a full heatmap
scotusheatmap_final <- ggplot(year_data, aes(year, justice_name)) +
    geom_tile(aes(fill = year_data$V1), colour="grey") +
    scale_fill_continuous(low="#fee08b", high="#1a9850",na.value="#333333") + 
    scale_x_continuous(name="Year", breaks=c(1953,1963,1973,1983, 1993, 2003, 2013)) +
    scale_y_discrete(name="Justice") +
    theme(legend.title=element_blank()) +
    theme(panel.background=element_blank())

# Review the final product 
scotusheatmap_final

# Save to file
ggsave(filename = "scotus_heatmap_final.pdf", width=12, height=8)
