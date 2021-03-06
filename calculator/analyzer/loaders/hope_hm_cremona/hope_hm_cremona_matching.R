library(dplyr)
library(tidyverse)
library(foreign)
library(data.table)
library(reshape2)
library(caret)
library(designmatch)
library(dplyr)
library(purrr)
library(cobalt)
library(gurobi)
library(Hmisc)
# install.packages('/Library/gurobi902/mac64/R/gurobi_9.0-2_R_3.6.1.tgz', repos=NULL)
library(slam)
library(tableone)
library(caret) # used to randomly split training/test sets

# source("hope_data_cleaning.R")
source("hope_hm_cremona_data_cleaning.R")
source("matching_functions.R")
source("descriptive_functions.R")

# Set the path
# save_path = "~/Dropbox (MIT)/covid19_personal/merging_data/covid19_hope_hm_cremona/"
# save_path = "~/Dropbox (MIT)/COVID_risk/covid19_hope/"
save_path = "~/Dropbox (Personal)/COVID_clinical/covid19_treatments_data/"

write_path = "~/Dropbox (Personal)/COVID_clinical/covid19_treatments_data/matched_single_treatments_hope_bwh/"
 
# Read in the data
data = read.csv(paste(save_path, "hope_hm_cremona_data_clean_imputed_addl_outcomes.csv",sep=""), header = TRUE)

# Look at descriptive stats on the different sources of data
merge_mult <- function(x, y){
  df <- merge(x, y, by = "Feature", all.x = TRUE, all.y = TRUE)
  return(df)
}


# Treatment for which we will split on
create_treatment_specific_dataset <- function(z){

outcomes = c('DEATH','COMORB_DEATH',"HF" ,"ARF","SEPSIS","EMBOLIC","OUTCOME_VENT")
#treatment_options = c('CORTICOSTEROIDS','CLOROQUINE', 'ANTIVIRAL', 'ANTICOAGULANTS',
#                      'INTERFERONOR', 'TOCILIZUMAB', 'ANTIBIOTICS', 'ACEI_ARBS')
treatment_options = c('CORTICOSTEROIDS','CLOROQUINE', 'ANTIVIRAL', 'ANTICOAGULANTS',
                      'INTERFERONOR', 'ACEI_ARBS')

dates = c('CORTICOSTEROIDS_START_DATE','CLOROQUINE_START_DATE','ANTIVIRAL_START_DATE','TOCILIZUMAB_START_DATE')

treatments = treatment_options[!treatment_options %in% z]

# data_hope <- data %>% filter(SOURCE == 'Hope')
# stats_data_hope <- descriptive_table(data_hope, short_version = TRUE)[[1]] %>% dplyr::select(-Missing) %>% rename(Hope_Summary = Summary)
# data_hm <- data %>% filter(SOURCE == 'HM')
# stats_data_hm <- descriptive_table(data_hm, short_version = TRUE)[[1]] %>% dplyr::select(-Missing) %>% rename(HM_Summary = Summary)
# data_cremona <- data %>% filter(SOURCE == 'Cremona')
# stats_data_cremona <- descriptive_table(data_cremona, short_version = TRUE)[[1]] %>% dplyr::select(-Missing) %>% rename(Cremona_Summary = Summary)
# stats_compare <- Reduce(merge_mult, list(stats_data_hope, stats_data_hm, stats_data_cremona))
# 
# stats_data_hope_text <- descriptive_table(data_hope, short_version = TRUE)[[2]] %>%
#   dplyr::filter(Feature == 'GENDER'|Feature == 'RACE'|Feature == 'CORTICOSTEROIDS') %>% dplyr::select(-Feature) %>% rename(Hope_Freq = Freq, Feature = Value)
# stats_data_hm_text <- descriptive_table(data_hm, short_version = TRUE)[[2]] %>% 
#   dplyr::filter(Feature == 'GENDER'|Feature == 'RACE'|Feature == 'CORTICOSTEROIDS') %>% dplyr::select(-Feature) %>% rename(HM_Freq = Freq, Feature = Value)
# stats_data_cremona_text <- descriptive_table(data_cremona, short_version = TRUE)[[2]] %>% 
#   dplyr::filter(Feature == 'GENDER'|Feature == 'RACE'|Feature == 'CORTICOSTEROIDS') %>% dplyr::select(-Feature) %>% rename(Cremona_Freq = Freq, Feature = Value)
# stats_compare_text <- Reduce(merge_mult, list(stats_data_hope_text, stats_data_hm_text, stats_data_cremona_text))

# Source and countries to include
# Derivation group
# groups = c("Hope-Spain","HM-Spain")
groups = c("Hope-Spain")
# groups = c("SPAIN")

#For treatments for which the response is NA, we will consider them as non-prescribed
data[,treatments][is.na(data[,treatments])] <- 0

# Filter the appropriate dataframe
df_full = data %>% filter(SOURCE_COUNTRY %in% groups)%>%dplyr::select(-REGIMEN)
df_other = data %>% filter(!(SOURCE_COUNTRY %in% groups))%>%dplyr::select(-REGIMEN)

# Keep the treatment as an independent vector
regimens_col = df_full %>%dplyr::select(z) 
regimens_col = replace(regimens_col, regimens_col==0, paste("NO_",z,sep=""))
regimens_col =replace(regimens_col, regimens_col==1, z)

# Select columns based on which the matching will take place
features_matching = c("AGE",
                      "GENDER",
                      "SAT02_BELOW92",
                      # "FAST_BREATHING",
                      "CREATININE",
                      # "ONSET_DATE_DIFF",
                      "LEUCOCYTES",
                      "HEMOGLOBIN",
                      "LYMPHOCYTES",
                      "PLATELETS",
                      "MAXTEMPERATURE_ADMISSION",
                      "OBESITY",
                      "HYPERTENSION",
                      "RENALINSUF",
                      "DISLIPIDEMIA",
                      "ANYLUNGDISEASE",
                      "ANYHEARTDISEASE",
                      "ANYCEREBROVASCULARDISEASE",
                      "PCR_B",
                      "DDDIMER_B",
                      "LDL_B",
                      "TRANSAMINASES_B",
                      "IN_DVITAMINSUPLEMENT",
                      "BLOOD_PRESSURE_ABNORMAL_B",
                      treatments
                      )

df <- df_full[,features_matching]

# One hot encode the dataset used for matching
dmy_out = dummyVars(" ~ .", data = df, drop2nd = TRUE, fullRank=T)
one_hot = predict(dmy_out, newdata = df)

# Split based on the treatments column
out <- split(data.frame(one_hot), f = regimens_col)

# We will pick as reference the treatment option that has the highest sample size
min_treat = nrow(df_full)
idx = 0
for (i in 1:length(out)){
  print(paste("Treatment option ", names(out)[i], " has ", nrow(out[[i]]), " observations.", sep = ""))
  if(nrow(out[[i]])<min_treat){
    min_treat = nrow(out[[i]])
    idx = i
    }
}

# Base on that statement we will pick as treatment of reference:
base_treatment = idx
t = 1:2
to_match_treatments = t[-base_treatment]
n_base = nrow(out[[base_treatment]])

## Global variables for matching
# Solver options
t_max = 600
solver_option = "gurobi"
approximate = 0

matched_data = list()
referenced_data = list()
matched_object_list = list()


for (t in to_match_treatments){
  # cycle through the different treatments which are each considered the "reference".
  # base_treatment is always the treatment (that we are matching to)
  match_output = matching_process(out, t, base_treatment, t_max, solver_option, approximate)
  matched_object_list[[t]] = match_output
  matched_data[[t]] = match_output$matched_data
  referenced_data[[t]] =  match_output$reference_data
}

# control = "treatment", so these are the first indices in the matrix 
common_control = 1:n_base 
for (i in to_match_treatments) {
  print(paste("Treatment option: ", names(out)[i], " (",i,")", sep = ""))
  print(paste("The matched dataframe has now ", nrow(matched_data[[i]]), " from ", nrow(out[[i]]), " observations."), sep = "")
  print(paste("The base dataframe has now ", nrow(referenced_data[[i]]), " from ", nrow(out[[base_treatment]]) , " observations"), sep = "")
  print("")
  control_inds = matched_object_list[[i]]$t_id
  common_control = intersect(common_control, control_inds)
  # common_control = union(common_control, control_inds)
}

length(common_control)

# Evaluate a single treatment ---------------------------------------------
vline = 0.15

# Select a treatment option to investigate
to_treat=to_match_treatments
t_inds = which(matched_object_list[[to_treat]]$t_ind == 1)

# The loveplot plots the absolute  differences in means 
loveplot_common(names(out)[to_treat], # 
                matched_object_list[[to_treat]]$mdt0, # matrix
                t_inds, # treatment indicators (original)
                matched_object_list[[to_treat]]$t_id, #(treatment indicators - matched)
                matched_object_list[[to_treat]]$c_id, #(control indicators - matched)
                common_control, # common base treatment indices (common)
                v_line=0.15) 

x = compare_features(df_full, base_treatment, to_treat, regimens_col, common_control = common_control)
ttest_original = x[['original']] 
ttest_filtered = x[['filtered']] 
ttest_compare = x[['compare']] 
ttest_compare %>% arrange(P_Filtered) %>% head(10)

# Save all selected data -------------------------------------------------
# Initialize with base treatment

matched_data = df_full %>% filter(get(z)==1) %>% slice(common_control)

# Add in all other treatments 
for (t in to_match_treatments){
  c_id = matched_object_list[[t]]$c_id - n_base #adjust down
  c_data = df_full %>% filter(get(z)==0) %>% slice(c_id)
  matched_data = rbind(matched_data, c_data)
}

# Look at summary stats stratified by treatment before and after matching ---------------------------------------------
covariateNames <- c(
  #Treatments
  z,
  # Demographic
  "AGE", "GENDER", "RACE",
  # Pre-Admission Comorbidities
  "DIABETES", "HYPERTENSION", "DISLIPIDEMIA", "OBESITY", "RENALINSUF", "ANYLUNGDISEASE", "AF", 
  "VIH", "ANYHEARTDISEASE", "ANYCEREBROVASCULARDISEASE", "CONECTIVEDISEASE", "LIVER_DISEASE", "CANCER",
  # Pre-Admission Medications
  "IN_PREVIOUSASPIRIN", "IN_OTHERANTIPLATELET", "IN_ORALANTICOAGL", "IN_BETABLOCKERS", "IN_BETAGONISTINHALED", 
  "IN_GLUCORTICOIDSINHALED", "IN_DVITAMINSUPLEMENT", "IN_BENZODIACEPINES", "IN_ANTIDEPRESSANT",
  # Vitals/labs at Admission
  "MAXTEMPERATURE_ADMISSION", "SAT02_BELOW92", "DDDIMER_B", "PCR_B", "TRANSAMINASES_B", "LDL_B", "BLOOD_PRESSURE_ABNORMAL_B", "CREATININE",
  "SODIUM", "LEUCOCYTES", "LYMPHOCYTES", "HEMOGLOBIN", "PLATELETS")

factorVars <- c(
"GENDER", "RACE", "DIABETES", "HYPERTENSION", "DISLIPIDEMIA", "OBESITY", "RENALINSUF", "ANYLUNGDISEASE", "AF", 
"VIH", "ANYHEARTDISEASE", "ANYCEREBROVASCULARDISEASE", "CONECTIVEDISEASE", "LIVER_DISEASE", "CANCER",
"IN_PREVIOUSASPIRIN", "IN_OTHERANTIPLATELET", "IN_ORALANTICOAGL", "IN_BETABLOCKERS", "IN_BETAGONISTINHALED", 
"IN_GLUCORTICOIDSINHALED", "IN_DVITAMINSUPLEMENT", "IN_BENZODIACEPINES", "IN_ANTIDEPRESSANT",
"SAT02_BELOW92", "DDDIMER_B", "PCR_B", "TRANSAMINASES_B", "LDL_B", "BLOOD_PRESSURE_ABNORMAL_B")

df_pre_treat_full <- df_full[covariateNames]
df_pre_treat_matched <- matched_data[covariateNames]

tableOne_beforematching <- CreateTableOne(vars = covariateNames, strata = z, 
                                data = df_pre_treat_full, factorVars = factorVars)
tableOne_aftermatching <- CreateTableOne(vars = covariateNames, strata =z, 
                                          data = df_pre_treat_matched, factorVars = factorVars)

print(tableOne_beforematching, smd = TRUE, quote = TRUE, noSpaces = TRUE)
print(tableOne_aftermatching, smd = TRUE, quote = TRUE, noSpaces = TRUE)
# Look at summary stats stratified by treatment before and after matching ---------------------------------------------

# Add in other countries (which will be test set)
# matched_data = rbind(matched_data, df_other)

## Remove irrelevant columns and regimen components
## keep source_country for train/test split
final <- matched_data %>% 
  dplyr::select(-c('DT_HOSPITAL_ADMISSION','HOSPITAL','SOURCE',dates))

## check sizes
table(final[,z], final$SOURCE_COUNTRY == "Hope-Spain"| final$SOURCE_COUNTRY == "HM-Spain" | final$SOURCE_COUNTRY == "Cremona-Italy")

final= final%>% mutate(REGIMEN = if_else(get(z)==1,z,paste("NO_",z,sep="")))%>%dplyr::select(-z)

write.csv(final, paste0(write_path, z,"_hope_matched.csv"), row.names = FALSE)
write.csv(matched_data[,c('DT_HOSPITAL_ADMISSION','HOSPITAL','SOURCE',dates)], paste0(write_path, z,"_hope_matched_dates.csv"), row.names = FALSE)

## Split into training, testing, and validation sets
# Unmatched
split_unmatched <- df_full %>% mutate(REGIMEN = if_else(get(z)==1,z,paste("NO_",z,sep="")))%>%dplyr::select(-z)%>%group_split(REGIMEN)

to_split = c(1,2)
unmatched_train = data.frame(matrix(ncol=0,nrow=0))
unmatched_test = data.frame(matrix(ncol=0,nrow=0))
for (t in to_split){
  set.seed(144)
  split <- createDataPartition(split_unmatched[[t]]$DEATH, p = 0.85, list = FALSE)
  unmatched.train <- split_unmatched[[t]][split,]
  unmatched.test <- split_unmatched[[t]][-split,]
  unmatched_train <- rbind(unmatched_train, unmatched.train)
  unmatched_test <- rbind(unmatched_test, unmatched.test)
}

write.csv(unmatched_train, paste0(write_path, z, "_hope_unmatched_all_treatments_train.csv"), row.names = FALSE)
write.csv(unmatched_test, paste0(write_path, z, "_hope_unmatched_all_treatments_test.csv"), row.names = FALSE)

# Matched
split_matched <- matched_data %>% mutate(REGIMEN = if_else(get(z)==1,z,paste("NO_",z,sep="")))%>%dplyr::select(-z)%>%group_split(REGIMEN)

to_split = c(1,2)
matched_train = data.frame(matrix(ncol=0,nrow=0))
matched_test = data.frame(matrix(ncol=0,nrow=0))
for (t in to_split){
  set.seed(144)
  split <- createDataPartition(split_matched[[t]]$DEATH, p = 0.85, list = FALSE)
  matched.train <- split_matched[[t]][split,]
  matched.test <- split_matched[[t]][-split,]
  matched_train <- rbind(matched_train, matched.train)
  matched_test <- rbind(matched_test, matched.test)
}

write.csv(matched_train, paste0(write_path, z, "_hope_matched_all_treatments_train.csv"), row.names = FALSE)
write.csv(matched_test, paste0(write_path, z, "_hope_matched_all_treatments_test.csv"), row.names = FALSE)

# Validation
df_other2 = df_other%>%filter(!HOSPITAL%in% c("Cremona","HM Foundation"))%>% mutate(REGIMEN = if_else(get(z)==1,z,paste("NO_",z,sep="")))%>%dplyr::select(-z)

write.csv(df_other2, paste0(write_path, z, "_hope_all_treatments_validation.csv"), row.names = FALSE)

write.csv(df_other2%>%filter(COUNTRY=='Italy', HOSPITAL!='Cremona'), paste0(write_path, z, "_hope_all_treatments_validation_hope_italy.csv"), row.names = FALSE)

write.csv(df_other%>%filter(HOSPITAL!='Cremona'), paste0(write_path, z, "_hope_all_treatments_validation_hope.csv"), row.names = FALSE)
 return(z)
}


treatment_options = c('CORTICOSTEROIDS','CLOROQUINE', 'ANTIVIRAL', 'ANTICOAGULANTS',
                      'INTERFERONOR', 'TOCILIZUMAB', 'ANTIBIOTICS', 'ACEI_ARBS')

z = "ACEI_ARBS"
create_treatment_specific_dataset(z)
  
