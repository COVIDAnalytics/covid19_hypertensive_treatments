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

df = read.csv(paste(save_path,"hope_data_clean.csv",sep = ""), header=TRUE, stringsAsFactors = FALSE)
data <- filter_columns(df)  
data <-  clean_columns(data)
str(data)

data %>% group_by(ANTICOAGULANTS, ANTIVIRAL, CLOROQUINE, REGIMEN) %>% summarize(n())

#Filtering outliers
filter_lb=0.01
filter_ub=0.99
filter_output = filter_outliers(data, filter_lb, filter_ub)
  
fl_data = filter_output[[1]]
outliers_table = filter_output[[2]]
str(fl_data)

# Remove missing data
missing_threshold = 0.4
filtered_missing = filter_missing(fl_data, missing_threshold)
fl_data = filtered_missing[[1]]
na_counts = filtered_missing[[2]]

SELECTED_TREATMENTS <- c('CLOROQUINE',
                         'ANTIVIRAL',
                         'ANTICOAGULANTS')

cols_nonX = c(SELECTED_TREATMENTS, 'REGIMEN', 'DEATH', 'COMORB_DEATH')
col_order = c(setdiff(names(fl_data), cols_nonX), cols_nonX)
fl_data <- fl_data[col_order]

write.csv(fl_data, paste(save_path,"hope_data_clean_filtered.csv",sep = ""),
          row.names = FALSE)

# Missing Data Imputation

treatments = c('CLOROQUINE','ANTIVIRAL','ANTICOAGULANTS','REGIMEN')
outcomes = c('DEATH','COMORB_DEATH')

group1 = c("SPAIN")
group2 = c("ECUADOR","Germany","Italy")
reps = 1
maxiterations = 10

derivation_cohort = imputation(fl_data, reps, maxiterations, group1, treatments, outcomes)
validation_cohort = imputation(fl_data, reps, maxiterations, group2, treatments, outcomes)
  
data_imputed = rbind(derivation_cohort, validation_cohort) 
write.csv(data_imputed, paste(save_path,"hope_data_clean_imputed.csv",sep = ""),
          row.names = FALSE)

fl_data %>%
  ggplot(aes(x = ONSET_DATE_DIFF, fill = COUNTRY, color = COUNTRY)) +
  # xlim(c(0,10)) +
  geom_histogram()

df %>% filter(grepl("remdesivir", tolower(df$OTHER_RELEVANT_COVID19_DRUGS))) 
