library(dplyr)
library(tidyverse)
library(foreign)
library(data.table)
library(reshape2)

#source("hope_features.R")
source("hope_data_cleaning.R")

# save_path = "~/Dropbox (Personal)/COVID_clinical/covid19_hope/"
save_path = "~/Dropbox (MIT)/COVID_risk/covid19_hope/"
#In case you would like to regenerate the data uncomment this code
#df<-create_data(save_path) 

df = read.csv(paste(save_path,"hope_data_clean.csv",sep = ""), header=TRUE)
data <- filter_columns (df)  
data <-  clean_columns(data)
str(data)

#Filtering outliers
filter_lb=0.01
filter_ub=0.99
filter_output = filter_outliers(data, filter_lb, filter_ub)
  
fl_data = filter_output[[1]]
outliers_table = filter_output[[2]]
str(fl_data)

#Count the extent of missing data
na_counts = t(fl_data %>%
  select(everything()) %>% 
  summarise_all(funs(100*sum(is.na(.))/nrow(fl_data)))) 

na_counts

data %>%
  ggplot(aes(x = ARTERIALBLOODGAS02SATURATION)) +
  xlim(c(0,100)) +
  geom_histogram()

data %>%
  ggplot(aes(x = ARTERIALBLOODGASPH, group = COUNTRY)) +
  geom_histogram()
