library(dplyr)
library(tidyverse)
library(foreign)
library(data.table)
library(reshape2)

# Helper functions --------------------------------------------------------
source('descriptive_functions.R')

# Generate results --------------------------------------------------------
save_path = "~/Dropbox (MIT)/COVID_risk/covid19_hope/"
fl_data <- read.csv(paste0(save_path,"hope_data_clean_filtered.csv"), stringsAsFactors = FALSE)

summ_list <- descriptive_table(fl_data,  short_version = TRUE)
summ_numeric <- summ_list[[1]]
summ_categorical <- summ_list[[2]]

results_bycountry <- lapply(unique(fl_data$COUNTRY), function(c){
  print(c)
  data_sub = fl_data %>% filter(COUNTRY == c)
  t = descriptive_table(data_sub, short_version = TRUE)[[1]]
  colnames(t) <- c('Feature', paste0("Summary_",c), paste0("Missing_",c))
  return(t)}
  ) %>% 
  reduce(left_join, by = "Feature")

results_bycountry_cat <- lapply(unique(fl_data$COUNTRY), function(c){
  print(c)
  data_sub = fl_data %>% filter(COUNTRY == c)
  t = descriptive_table(data_sub, short_version = TRUE)[[2]]
  colnames(t) <- c('Feature', 'Value', paste0("Freq_",c))
  return(t)}
) %>% 
  reduce(full_join, by = c("Feature","Value")) %>%
  filter(!Feature %in% c('COUNTRY','HOSPITAL','DT_HOSPITAL_ADMISSION')) %>%
  arrange(Feature, Value)

       
write.csv(results_bycountry, paste0(save_path, "description_bycountry.csv"), row.names = FALSE)
write.csv(results_bycountry_cat, paste0(save_path, "description_bycountry_categoric.csv"), row.names = FALSE)


# Pairwise significance ---------------------------------------------------
data_derivation <- fl_data %>% filter(COUNTRY %in% c('SPAIN'))
data_validation <- fl_data %>% filter(COUNTRY %in% c('Germany','Italy','Cuba','ECUADOR'))

t_deriv = descriptive_table(data_derivation, short_version = TRUE)[[1]] %>%
  `colnames<-`(c('Feature', paste0("Summary_Derivation"), paste0("Missing_Derivation")))
t_val = descriptive_table(data_validation, short_version = TRUE)[[1]] %>%
  `colnames<-`(c('Feature', paste0("Summary_Validation"), paste0("Missing_Validation")))

ttest <- do.call(rbind, lapply(names(data_derivation), function(col){
  if (is.numeric(data_derivation[[col]])){
    r = t.test(data_derivation[[col]], data_validation[[col]], var.equal = FALSE, na.action = "na.omit")
    return(c(col, r$p.value))
  }})) %>%
  as.data.frame() %>%
  `colnames<-`(c('Feature','P-Value'))

pop_compare <- list(t_deriv, t_val, ttest) %>%
  reduce(full_join, by = 'Feature')

write.csv(pop_compare, paste0(save_path, "description_bysplit.csv"), row.names = FALSE)

# Pairwise significance ---------------------------------------------------
data_a <- data %>% filter(DEATH == 1)
label_a <- 'NonSurvivor'
data_b <- data %>% filter(DEATH == 0)
label_b <- 'Survivor'
file_name <- 'description_bysurvival.csv'

write.csv(pop_compare, paste0(save_path, file_name), row.names = FALSE)
