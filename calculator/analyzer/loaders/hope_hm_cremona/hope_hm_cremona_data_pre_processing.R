library(dplyr)
library(tidyverse)
library(foreign)
library(data.table)
library(reshape2)
library(stringr)

#source("hope_features.R")
# source("hope_data_cleaning.R")
source("hope_hm_cremona_data_cleaning.R")

save_path = "~/Dropbox (Personal)/COVID_clinical/covid19_hope/"
alt_save_path = "~/Dropbox (Personal)/COVID_clinical/covid19_treatments_data/"
# save_path = "~/Dropbox (MIT)/COVID_risk/covid19_hope/"
#save_path = "~/Dropbox (MIT)/covid19_personal/merging_data/covid19_hope_hm_cremona/"
#alt_save_path = "~/Dropbox (MIT)/covid19_personal/merging_data/covid19_hope_hm_cremona/"
#In case you would like to regenerate the data uncomment this code
#df<-create_data(save_path) 

# Read in Hope data, filter columns, and clean
df_hope = read.csv(paste(save_path,"hope_data_clean.csv",sep = ""), header=TRUE, stringsAsFactors = FALSE)
data_hope = filter_columns(df_hope)  
data_hope = clean_columns(data_hope)
data_hope = data_hope %>% 
  rename(TOCILIZUMAB_START_DATE = DT_USE_TOCILIZUMAB2,
        CORTICOSTEROIDS_START_DATE = DT_USE_CORTICOIDS2,
        ANTIVIRAL_START_DATE = DT_USE_ANTIVIRAL_DRUGS2,
        CLOROQUINE_START_DATE = DT_USE_CLOROQUINE2)
str(data_hope)

# Read in HM Foundation data (should be cleaned and formatted) and filter columns
df_hm = read.csv(paste(alt_save_path,"hm_treatments_2020-08-02.csv",sep = ""), header=TRUE, stringsAsFactors = FALSE)
# Add extra dates for admission
hm_dates = read.csv(paste(alt_save_path,"hm_meds_with_start_dates.csv",sep = ""), header=TRUE, stringsAsFactors = FALSE)
df_hm = merge(df_hm, hm_dates[,c('PATIENT.ID','CORTICOSTEROIDS_START_DATE','ANTIVIRAL_START_DATE','CLOROQUINE_START_DATE','TOCILIZUMAB_START_DATE')], by = 'PATIENT.ID', all.x = TRUE)
#Clean the columns
data_hm <- filter_columns_nonhope(df_hm)

# Convert admission date to date format
data_hm <- data_hm %>% mutate(DT_HOSPITAL_ADMISSION = as.Date(DT_HOSPITAL_ADMISSION),
                              CORTICOSTEROIDS_START_DATE = as.Date(CORTICOSTEROIDS_START_DATE, format = "%Y-%m-%d"),
                              ANTIVIRAL_START_DATE = as.Date(ANTIVIRAL_START_DATE, format = "%Y-%m-%d"),
                              CLOROQUINE_START_DATE = as.Date(CLOROQUINE_START_DATE, format = "%Y-%m-%d"),
                              TOCILIZUMAB_START_DATE = as.Date(TOCILIZUMAB_START_DATE, format = "%Y-%m-%d"))
str(data_hm)

# Read in Cremona data (should be cleaned and formatted) and filter columns
df_cremona = read.csv(paste(alt_save_path,"cremona_treatments.csv",sep = ""), header=TRUE, stringsAsFactors = FALSE)
# Add extra dates for admission
cremona_dates = read.csv(paste(alt_save_path,"cremona_meds_with_start_dates.csv",sep = ""), header=TRUE, stringsAsFactors = FALSE)
df_cremona = cbind(df_cremona, cremona_dates[,c('CORTICOSTEROIDS_START_DATE','ANTIVIRAL_START_DATE','CLOROQUINE_START_DATE','TOCILIZUMAB_START_DATE')])
data_cremona <- filter_columns_nonhope(df_cremona)  
# Convert admission date to date format
data_cremona <- data_cremona %>% mutate(DT_HOSPITAL_ADMISSION = as.Date(DT_HOSPITAL_ADMISSION),
                                        CORTICOSTEROIDS_START_DATE = as.Date(CORTICOSTEROIDS_START_DATE, format = "%Y-%m-%d"),
                                        ANTIVIRAL_START_DATE = as.Date(ANTIVIRAL_START_DATE, format = "%Y-%m-%d"),
                                        CLOROQUINE_START_DATE = as.Date(CLOROQUINE_START_DATE, format = "%Y-%m-%d"),
                                        TOCILIZUMAB_START_DATE = as.Date(TOCILIZUMAB_START_DATE, format = "%Y-%m-%d"))
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
str(data_merged)

# Create summary stats of treatments overall and grouped by dataset
data_merged %>% group_by(ANTICOAGULANTS, ANTIVIRAL, CLOROQUINE, REGIMEN) %>% dplyr::summarize(n())
data_merged %>% group_by(SOURCE, ANTICOAGULANTS, ANTIVIRAL, CLOROQUINE, REGIMEN) %>% dplyr::summarize(n())

# Look at drug use with dates
data_merged = data_merged%>%
                  mutate(date_diff_corticosteroids = as.numeric(difftime(CORTICOSTEROIDS_START_DATE ,DT_HOSPITAL_ADMISSION , units = c("days"))))

# For all the patients for which CORTICOSTEROIDS is equal to 1 we will leave it as it is.
# For those patients for which CORTICOSTEROIDS=0 but there is a date and it is prior to the admission, we will leave it as it is.
# For those patients for which CORTICOSTEROIDS=0 but there is a date and it is after the admission, we will change it to 1.

data_merged$CORTICOSTEROIDS[which(data_merged$CORTICOSTEROIDS==0 & data_merged$date_diff_corticosteroids>=0  & !is.na(data_merged$date_diff_corticosteroids))]<-1

table(data_merged$CORTICOSTEROIDS)
summary(data_merged$date_diff_corticosteroids)
nrow(data_merged)-sum(is.na(data_merged$date_diff_corticosteroids))
table(data_merged$CORTICOSTEROIDS)

data_merged%>%
  filter(!is.na(CORTICOSTEROIDS),COUNTRY!="China", COUNTRY!="Cuba")%>%
  group_by(COUNTRY,SOURCE,CORTICOSTEROIDS)%>%
  summarise(count=n(),
            num_nas=sum(is.na(date_diff_corticosteroids)), 
            date_known = sum(!is.na(date_diff_corticosteroids)), 
            perc_with_date = 100*date_known/count)

#Remove the date difference column
data_merged<-data_merged%>%dplyr::select(-date_diff_corticosteroids)

# Filter outliers
filter_lb=0.01
filter_ub=0.99
filter_output = filter_outliers(data_merged, filter_lb, filter_ub)

fl_data = filter_output[[1]]
outliers_table = filter_output[[2]]
str(fl_data)

# Look at column missingness for hope, hm, and cremona separately
missing_threshold = 0.4
DATES = c('CORTICOSTEROIDS_START_DATE','CLOROQUINE_START_DATE','ANTIVIRAL_START_DATE','TOCILIZUMAB_START_DATE')

fl_data_hm <- fl_data %>% filter(SOURCE == 'HM')%>%select(-DATES)
filtered_missing_hm = filter_missing(fl_data_hm, missing_threshold)
fl_data_hm = filtered_missing_hm[[1]]
na_counts_hm = filtered_missing_hm[[2]]
fl_data_hm[,DATES] <- fl_data %>% filter(SOURCE == 'HM')%>%select(DATES)

fl_data_cremona <- fl_data %>% filter(SOURCE == 'Cremona')%>%select(-DATES)
filtered_missing_cremona = filter_missing(fl_data_cremona, missing_threshold)
fl_data_cremona = filtered_missing_cremona[[1]]
na_counts_cremona = filtered_missing_cremona[[2]]
fl_data_cremona[,DATES] <- fl_data %>% filter(SOURCE == 'Cremona')%>%select(DATES)

fl_data_hope <- fl_data %>% filter(SOURCE == 'Hope')%>%select(-DATES)
filtered_missing_hope = filter_missing(fl_data_hope, missing_threshold)
fl_data_hope = filtered_missing_hope[[1]]
na_counts_hope = filtered_missing_hope[[2]]
fl_data_hope[,DATES] <- fl_data %>% filter(SOURCE == 'Hope')%>%select(DATES)

# Remove missing data on merged dataset
missing_threshold = 0.4
fl_dates = fl_data%>%select(DATES)
filtered_missing = filter_missing(fl_data%>%select(-DATES), missing_threshold)
fl_data = filtered_missing[[1]]
na_counts = filtered_missing[[2]]
fl_data[,DATES] <- fl_dates

SELECTED_TREATMENTS <- c('CLOROQUINE',
                         'ANTIVIRAL',
                         'ANTICOAGULANTS',
                         'ACEI_ARBS',
                         'CORTICOSTEROIDS',
                         'INTERFERONOR')

INDICATORS <- c('SOURCE',
                'SOURCE_COUNTRY')

cols_nonX = c(SELECTED_TREATMENTS, 'REGIMEN', 'DEATH', 'COMORB_DEATH', INDICATORS, DATES)
col_order = c(setdiff(names(fl_data), cols_nonX), cols_nonX)
fl_data <- fl_data[col_order]
str(fl_data)

write.csv(fl_data, paste(alt_save_path,"hope_hm_cremona_data_clean_filtered_addl_outcomes.csv",sep = ""),
          row.names = FALSE)

# Perform missing data imputation
treatments = c('CLOROQUINE','ANTIVIRAL','ANTICOAGULANTS','ACEI_ARBS',
               'CORTICOSTEROIDS',
               'INTERFERONOR','REGIMEN')
outcomes = c('DEATH','COMORB_DEATH')
indicator_cols = c('SOURCE','SOURCE_COUNTRY')
dates = c('CORTICOSTEROIDS_START_DATE','CLOROQUINE_START_DATE','ANTIVIRAL_START_DATE','TOCILIZUMAB_START_DATE')


# Derivation group
group1 = c("Hope-Spain","HM-Spain")
# Validation group
group2 = c("Hope-Ecuador","Hope-Germany","Hope-Italy","Cremona-Italy")
reps = 1
maxiterations = 10

derivation_cohort = imputation(fl_data, reps, maxiterations, group1, treatments, outcomes, indicator_cols,dates)
validation_cohort = imputation(fl_data, reps, maxiterations, group2, treatments, outcomes, indicator_cols,dates)

data_imputed = rbind(derivation_cohort, validation_cohort) 
write.csv(data_imputed, paste(alt_save_path,"hope_hm_cremona_data_clean_imputed_addl_outcomes.csv",sep = ""),
          row.names = FALSE)

# fl_data %>%
#   ggplot(aes(x = ONSET_DATE_DIFF, fill = COUNTRY, color = COUNTRY)) +
#   # xlim(c(0,10)) +
#   geom_histogram()
# 
# df %>% filter(grepl("remdesivir", tolower(df$OTHER_RELEVANT_COVID19_DRUGS))) 
