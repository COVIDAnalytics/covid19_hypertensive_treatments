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
#install.packages('/Library/gurobi811/mac64/R/gurobi_8.1-1_R_3.5.0.tgz', repos=NULL)

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
df = data %>%filter(COUNTRY %in% groups) %>% dplyr::select(-outcomes, DT_HOSPITAL_ADMISSION)
#Keep the treatment as an independent vector
regimens_col = df%>%select(REGIMEN)

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

df<-df[,features_matching]

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

## Global variables for matching
# Solver options
t_max = 60
solver_option = "gurobi"
approximate = 0

matched_data =list()
referenced_data =list()
matched_object_list = list()

for (to_treat in to_match_treatments){
  matched_object_list[[to_treat]] = matching_process(out, ref_treatment, to_treat, t_max, solver_option, approximate)
  matched_data[[to_treat]] = matched$matched_data
  referenced_data[[to_treat]] =  matched$reference_data
}

for (i in to_match_treatments) {
  print(paste("Treatment option :", names(out)[i], sep = ""))
  print(paste("The original dataset has ", nrow(out[[i]]), " observations.", sep = ""))
  print(paste("The matched dataframe has now ", nrow(matched_data[[i]]), " observations"), sep = "")
  print(paste("The referenced dataframe has now ", nrow(referenced_data[[i]]), " from ", nrow(out[[ref_treatment]]) , " observations"), sep = "")
  print("")
}


# The loveplot plots the absolute standardized differences in means 
# Vertical line for satisfactory balance
vline = 0.15

#Select a treatment option to investigate
to_treat=4
loveplot_custom(names(out)[to_treat],mdt, matched_object_list[[to_treat]]$t_id, matched_object_list[[to_treat]]$c_id, vline) 

