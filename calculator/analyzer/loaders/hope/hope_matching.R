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
#install.packages('/Library/gurobi811/mac64/R/gurobi_8.1-1_R_3.5.0.tgz', repos=NULL)

source("hope_data_cleaning.R")

# Discretize covariates
quantiles = function(covar, n_q) {
  p_q = seq(0, 1, 1/n_q)
  val_q = quantile(covar, probs = p_q, na.rm = TRUE)
  covar_out = rep(NA, length(covar))
  for (i in 1:n_q) {
    if (i==1) {covar_out[covar<val_q[i+1]] = i}
    if (i>1 & i<n_q) {covar_out[covar>=val_q[i] & covar<val_q[i+1]] = i}
    if (i==n_q) {covar_out[covar>=val_q[i] & covar<=val_q[i+1]] = i}}
  return(covar_out)
}

matching_process<-function(data, reference_df, matched_df, t_max, solver, approximate){
  
  #Create a treatment column
  data[[reference_df]]$treatment = 0
  data[[matched_df]]$treatment = 1
  
  t_ind = c(data[[reference_df]]$treatment, data[[matched_df]]$treatment)
  mdt = rbind(data[[reference_df]], data[[matched_df]])
  
  #Find all binary columns
  bin_cols = apply(mdt,2,function(x) { all(na.omit(x) %in% 0:1) })
  #Discretize continuous columns
  mdt[,!bin_cols] = apply(mdt[,!bin_cols],2,quantiles,n_q=5)
  
  #Remove the treatment column
  mdt$treatment = NULL
  
  #Set the solver options
  solver = list(name = solver_option, t_max = t_max, approximate = approximate,
                round_cplex = 0, trace = 0)
  
  # Fine balance
  fine = list(covs = mdt)
  # Match
  matched1 = cardmatch(t_ind, fine = fine, solver = solver)
  
  # Indices of the treated units and matched controls
  t_id_1 = matched1$t_id
  c_id_1 = matched1$c_id
  
  for (i in 1:ncol(mdt)) {
    print(names(mdt)[i])
    print(finetab(mdt[, i], t_id_1, c_id_1))
  }
  
  reference_data = mdt[c_id_1,]
  matched_data = mdt[t_id_1,]
  
  summary_means = meantab(mdt, t_ind, t_id_1, c_id_1)
  
  match_result = list(matched = matched1, reference_data = reference_data, 
                      matched_data = matched_data, 
                      summary_means= summary_means, mdt = mdt, 
                      t_ind = t_ind, t_id= t_id_1, c_id = c_id_1)
  return(match_result)
}

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


# Plots the absolute standardized differences in means 
# Vertical line for satisfactory balance
vline = 0.15

#Select a treatment option to investigate
to_treat=2
loveplot(mdt, matched_object_list[[to_treat]]$t_id, matched_object_list[[to_treat]]$c_id, vline) 




