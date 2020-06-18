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
source("descriptive_functions.R")

#Set the path
save_path = "~/Dropbox (MIT)/COVID_risk/covid19_hope/"
# save_path = "~/Dropbox (Personal)/COVID_clinical/covid19_hope/"

#Read in the data
data = read.csv(paste(save_path, "hope_data_clean_imputed.csv",sep=""), header = TRUE)

#Countries to include
groups = c("SPAIN")
treatments = c('CLOROQUINE','ANTIVIRAL','ANTICOAGULANTS','REGIMEN')
outcomes = c('DEATH','COMORB_DEATH')
#Filter the appropriate dataframe
df_full = data %>%filter(COUNTRY %in% groups) 
df_other = data %>%filter(!(COUNTRY %in% groups))
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
# Chloroquine Only with 725 observations
base_treatment = 4
t = 1:5
to_match_treatments = t[-base_treatment]
n_base = nrow(out[[base_treatment]])

## Global variables for matching
# Solver options
t_max = 600
solver_option = "gurobi"
approximate = 0

matched_data =list()
referenced_data =list()
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
  # common_control = intersect(common_control, control_inds)
  # common_control = union(common_control, control_inds)
}

length(common_control)

# Evaluate a single treatment ---------------------------------------------
vline = 0.15

#Select a treatment option to investigate
to_treat=5
t_inds = which(matched_object_list[[to_treat]]$t_ind == 1)

# The loveplot plots the absolute  differences in means 
loveplot_common(names(out)[to_treat], # 
                matched_object_list[[to_treat]]$mdt0, # matrix
                t_inds, # treatment indicators (original)
                matched_object_list[[to_treat]]$t_id, #(treatment indicators - matched)
                matched_object_list[[to_treat]]$c_id, #(control indicators - matched)
                common_control, # common base treatment indices (common)
                v_line=0.15) 

x = compare_features(df_full, base_treatment, to_treat, common_control = common_control)
ttest_original = x[['original']] 
ttest_filtered = x[['filtered']] 
ttest_compare = x[['compare']] 
ttest_compare %>% arrange(P_Filtered) %>% head(10)


# Save all selected data -------------------------------------------------
# Initialize with base treatment

matched_data = df_full %>% filter(REGIMEN ==names(out)[[base_treatment]]) %>% slice(common_control)

# Add in all other treatments 
for (t in to_match_treatments){
  c_id = matched_object_list[[t]]$c_id - n_base #adjust down
  c_data = df_full %>% filter(REGIMEN == names(out)[[t]]) %>% slice(c_id)
  matched_data = rbind(matched_data, c_data)
}

# Add in other countries (which will be test set)
matched_data = rbind(matched_data, df_other)

## Remove irrelevant columns and regimen components
## keep country for train/test split
final <- matched_data %>% 
  dplyr::select(-c('CLOROQUINE','ANTIVIRAL','ANTICOAGULANTS','DT_HOSPITAL_ADMISSION','HOSPITAL'))

## check sizes
table(final$REGIMEN, final$COUNTRY == "SPAIN")

write.csv(final, paste0(save_path, "hope_matched.csv"), row.names = FALSE)
