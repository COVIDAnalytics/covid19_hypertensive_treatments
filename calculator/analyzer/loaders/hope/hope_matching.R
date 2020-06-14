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

source("hope_data_cleaning.R")
source("matching_functions.R")

#Set the path
save_path = "~/Dropbox (MIT)/COVID_risk/covid19_hope/"

#Read in the data
data = read.csv(paste(save_path, "hope_data_clean_imputed.csv",sep=""), header = TRUE)

#Countries to include
groups = c("SPAIN")
treatments = c('CLOROQUINE','ANTIVIRAL','ANTICOAGULANTS','REGIMEN')
outcomes = c('DEATH','COMORB_DEATH')
#Filter the appropriate dataframe
df_full = data %>%filter(COUNTRY %in% groups) %>% dplyr::select(-outcomes, DT_HOSPITAL_ADMISSION)
#Keep the treatment as an independent vector
regimens_col = df_full%>%dplyr::select(REGIMEN)

#Select columns based on which the matching will take place
features_matching = c("AGE",
                      "GENDER",
                      "SAT02_BELOW92",
                      "FAST_BREATHING",
                      "CREATININE",
                      "ONSET_DATE_DIFF",
                      "LEUCOCYTES",
                      "HEMOGLOBIN",
                      "LYMPHOCYTES",
                      "PLATELETS",
                      "MAXTEMPERATURE_ADMISSION")

df<-df_full[,features_matching]

#One hot encode the dataset
dmy_out = dummyVars(" ~ .", data = df, drop2nd = TRUE, fullRank=T)
one_hot = predict(dmy_out, newdata = df)

# Split based on the treatments column
out <- split(data.frame(one_hot), f = regimens_col)

#We will pick as reference the treatment option that has the highest sample size
for (i in 1:length(out)){
  print(paste("Treatment option ", names(out)[i], " has ", nrow(out[[i]]), " observations.", sep = ""))
}

#Base on that statement we will pick as treatment of reference:
# Chloroquine and Antivirals- 3
ref_treatment = 3
t = 1:5
to_match_treatments = t[-ref_treatment]
n_control = nrow(out[[ref_treatment]])

## Global variables for matching
# Solver options
t_max = 60
solver_option = "gurobi"
approximate = 0

matched_data =list()
referenced_data =list()
matched_object_list = list()


for (to_treat in to_match_treatments){
  match_output = matching_process(out, ref_treatment, to_treat, t_max, solver_option, approximate)
  matched_object_list[[to_treat]] = match_output
  matched_data[[to_treat]] = match_output$matched_data
  referenced_data[[to_treat]] =  match_output$reference_data
}

common_control = 1:n_control
for (i in to_match_treatments) {
  print(paste("Treatment option :", names(out)[i], sep = ""))
  print(paste("The original dataset has ", nrow(out[[i]]), " observations.", sep = ""))
  print(paste("The matched dataframe has now ", nrow(matched_data[[i]]), " observations"), sep = "")
  print(paste("The referenced dataframe has now ", nrow(referenced_data[[i]]), " from ", nrow(out[[ref_treatment]]) , " observations"), sep = "")
  print("")
  common_control = intersect(common_control, matched_object_list[[i]]$c_id)
}

common_control


# Evaluate a single treatment ---------------------------------------------

#Select a treatment option to investigate
to_treat=4

t_ind = matched_object_list[[to_treat]]$t_ind
t_inds = which(t_ind == 1)

#I have an issue with where the legend appears
# Box is before matching and * after matching

# The loveplot plots the absolute  differences in means 
# We can change the function to reflect the absolute standardized differences in means
# Vertical line for satisfactory balance
vline = 0.15

plt = loveplot_common(names(out)[to_treat], # 
                matched_object_list[[to_treat]]$mdt0, # matrix
                t_inds, # treatment indicators (original)
                matched_object_list[[to_treat]]$t_id, #(treatment indicators - matched)
                matched_object_list[[to_treat]]$c_id, #(control indicators - matched)
                common_control, # control_indicators (common)
                vline) 


compare_features <- function(ref_treat, to_treat, common_control){
  label_a = names(out)[ref_treatment]
  label_b = names(out)[to_treat]
  data_a = df_full %>%filter(REGIMEN == label_a)
  data_b = df_full %>%filter(REGIMEN == label_b)
  data_stack = rbind(data_a, data_b)
  nrow(data_stack) == nrow(matched_object_list[[to_treat]]$mdt0)
  # data_a_filtered = data_stack[matched_object_list[[to_treat]]$c_id,]
  data_a_filtered = data_stack[common_control,]
  data_b_filtered = data_stack[matched_object_list[[to_treat]]$t_id,]
  
  ttest_original = run_ttest(data_a, data_b, label_a, label_b, cols_exclude = treatments)
  ttest_filtered = run_ttest(data_a_filtered, data_b_filtered, label_a, label_b, cols_exclude = treatments)
  
  violate_original = ttest_original %>% filter(`P-Value` < 0.01) %>% pull(Feature) %>% sort
  violate_filtered = ttest_filtered %>% filter(`P-Value` < 0.01)  %>% pull(Feature) %>% sort 
  
  ttest_compare = ttest_original %>% dplyr::select(Feature, P_0 = `P-Value`) %>%
    left_join(ttest_filtered %>% dplyr::select(Feature, P_Filtered = `P-Value`), on = 'Feature')
  
  new_violations = ttest_compare %>%
    filter(P_0 > P_Filtered) %>%
    filter(P_Filtered < 0.01) %>%
    arrange(P_Filtered)
  
  print(paste0("Original Significant Differences (Count = ", 
               length(violate_original),"): ",
               paste0(violate_original, collapse = ", ")))
  
  print(paste0("Filtered Significant Differences (Count = ", 
               length(violate_filtered),"): ",
               paste0(violate_filtered, collapse = ", ")))
  
  print("New Violations: ")
  print(new_violations)
  
  return(list(ttest_original,ttest_filtered,ttest_compare))
}

x = compare_features(3,1,common_control)

