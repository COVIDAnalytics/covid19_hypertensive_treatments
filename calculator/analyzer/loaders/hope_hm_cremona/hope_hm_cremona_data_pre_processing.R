library(dplyr)
library(tidyverse)
library(foreign)
library(data.table)
library(reshape2)
library(stringr)

#source("hope_features.R")
# source("hope_data_cleaning.R")
source("hope_hm_cremona_data_cleaning.R")

# save_path = "~/Dropbox (Personal)/COVID_clinical/covid19_hope/"
# save_path = "~/Dropbox (MIT)/COVID_risk/covid19_hope/"
save_path = "~/Dropbox (MIT)/covid19_personal/merging_data/covid19_hope_hm_cremona/"
#In case you would like to regenerate the data uncomment this code
#df<-create_data(save_path) 

# Read in Hope data, filter columns, and clean
df_hope = read.csv(paste(save_path,"hope_data_clean.csv",sep = ""), header=TRUE, stringsAsFactors = FALSE)
data_hope <- filter_columns(df_hope)  
data_hope <- clean_columns(data_hope)
str(data_hope)

# Read in HM Foundation data (should be cleaned and formatted) and filter columns
df_hm = read.csv(paste(save_path,"hm_treatments.csv",sep = ""), header=TRUE, stringsAsFactors = FALSE)
data_hm <- filter_columns_nonhope(df_hm)
# Convert admission date to date format
data_hm <- data_hm %>% mutate(DT_HOSPITAL_ADMISSION = as.Date(DT_HOSPITAL_ADMISSION))
str(data_hm)

# Read in Cremona data (should be cleaned and formatted) and filter columns
df_cremona = read.csv(paste(save_path,"cremona_treatments.csv",sep = ""), header=TRUE, stringsAsFactors = FALSE)
data_cremona <- filter_columns_nonhope(df_cremona)  
# Convert admission date to date format
data_cremona <- data_cremona %>% mutate(DT_HOSPITAL_ADMISSION = as.Date(DT_HOSPITAL_ADMISSION))
str(data_cremona)

# Check stats on each data source to ensure uniformity (e.g., units) before merging
summary(data_hope)
summary(data_hm)
summary(data_cremona)

# Merge separate data sources
data_hope["SOURCE"] = 'Hope'
data_hm["SOURCE"] = 'HM'
data_cremona["SOURCE"] = 'Cremona'
data_merged = rbind(data_hope,data_hm,data_cremona)

data_merged["COUNTRY"] = str_to_title(data_merged$COUNTRY)
data_merged <- data_merged %>% mutate(SOURCE_COUNTRY = paste0(SOURCE, "-", COUNTRY))

# Create summary stats of treatments overall and grouped by dataset
data_merged %>% group_by(ANTICOAGULANTS, ANTIVIRAL, CLOROQUINE, REGIMEN) %>% summarize(n())

data_merged %>% group_by(SOURCE, ANTICOAGULANTS, ANTIVIRAL, CLOROQUINE, REGIMEN) %>% summarize(n())

# Filter outliers
filter_lb=0.01
filter_ub=0.99
filter_output = filter_outliers(data_merged, filter_lb, filter_ub)
  
fl_data = filter_output[[1]]
outliers_table = filter_output[[2]]
str(fl_data)

# Look at column missingness for hope, hm, and cremona separately
missing_threshold = 0.4

fl_data_hm <- fl_data %>% filter(SOURCE == 'HM')
filtered_missing_hm = filter_missing(fl_data_hm, missing_threshold)
fl_data_hm = filtered_missing_hm[[1]]
na_counts_hm = filtered_missing_hm[[2]]

fl_data_cremona <- fl_data %>% filter(SOURCE == 'Cremona')
filtered_missing_cremona = filter_missing(fl_data_cremona, missing_threshold)
fl_data_cremona = filtered_missing_cremona[[1]]
na_counts_cremona = filtered_missing_cremona[[2]]

fl_data_hope <- fl_data %>% filter(SOURCE == 'Hope')
filtered_missing_hope = filter_missing(fl_data_hope, missing_threshold)
fl_data_hope = filtered_missing_hope[[1]]
na_counts_hope = filtered_missing_hope[[2]]

# Remove missing data on merged dataset
missing_threshold = 0.4
filtered_missing = filter_missing(fl_data, missing_threshold)
fl_data = filtered_missing[[1]]
na_counts = filtered_missing[[2]]

SELECTED_TREATMENTS <- c('CLOROQUINE',
                         'ANTIVIRAL',
                         'ANTICOAGULANTS')

INDICATORS <- c('SOURCE',
                'SOURCE_COUNTRY')

cols_nonX = c(SELECTED_TREATMENTS, 'REGIMEN', 'DEATH', 'COMORB_DEATH', INDICATORS)
col_order = c(setdiff(names(fl_data), cols_nonX), cols_nonX)
fl_data <- fl_data[col_order]

write.csv(fl_data, paste(save_path,"hope_hm_cremona_data_clean_filtered.csv",sep = ""),
          row.names = FALSE)

# Perform missing data imputation
treatments = c('CLOROQUINE','ANTIVIRAL','ANTICOAGULANTS','REGIMEN')
outcomes = c('DEATH','COMORB_DEATH')
indicator_cols = c('SOURCE','SOURCE_COUNTRY')

# Derivation group
group1 = c("Hope-Spain","HM-Spain","Cremona-Italy")
# Validation group
group2 = c("Hope-Ecuador","Hope-Germany","Hope-Italy")
reps = 1
maxiterations = 10

derivation_cohort = imputation(fl_data, reps, maxiterations, group1, treatments, outcomes, indicator_cols)
validation_cohort = imputation(fl_data, reps, maxiterations, group2, treatments, outcomes, indicator_cols)
  
data_imputed = rbind(derivation_cohort, validation_cohort) 
write.csv(data_imputed, paste(save_path,"hope_hm_cremona_data_clean_imputed.csv",sep = ""),
          row.names = FALSE)

# fl_data %>%
#   ggplot(aes(x = ONSET_DATE_DIFF, fill = COUNTRY, color = COUNTRY)) +
#   # xlim(c(0,10)) +
#   geom_histogram()
# 
# df %>% filter(grepl("remdesivir", tolower(df$OTHER_RELEVANT_COVID19_DRUGS))) 
