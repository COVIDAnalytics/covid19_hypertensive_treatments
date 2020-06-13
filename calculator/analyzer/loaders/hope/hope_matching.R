library(dplyr)
library(tidyverse)
library(foreign)
library(data.table)
library(reshape2)
library(caret)
library(designmatch)
library(dplyr)
library(purrr)
library(gurobi)
#install.packages('/Library/gurobi811/mac64/R/gurobi_8.1-1_R_3.5.0.tgz', repos=NULL)

source("hope_data_cleaning.R")

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

# Split based on the treatments column
out <- split( df , f = df$REGIMEN )

# Select columns based on which the matching will take place
# We will start by the following but we can adjust. 

cols_include = c(DEMOGRAPHICS, COMORBIDITIES, 
                 DRUGS_ADMISSIONS, VITALS,
                 BINARY_LABS_VITALS_ADMISSION, CONTINUE_LABS_ADMISSION,
                 XRAY_RESULTS)

#Select only columns based on which we will match
out = lapply(out, function(x) subset(x, select = intersect(cols_include, colnames(x))))

#We will need to one hot encode every subdataframe
dmy_out = lapply(out, function(x) dummyVars(" ~ .", data = x))
one_hot_out = mapply(function(x, y) data.frame(predict(x, newdata = y)), dmy_out, out)

one_hot_out = list()                                            
for (i in 1:length(out)) {
  one_hot_out[[i]] = data.frame(predict(dmy_out[[i]], newdata = out[[i]]))
}

#Question: Should we exclude one of the options as it is determined by the rest of the columns?
one_hot_out[[1]]$treatment = 0 
one_hot_out[[2]]$treatment = 1 

treatment = c(one_hot_out[[1]]$treatment, one_hot_out[[2]]$treatment)
t_ind = treatment
t_ind
mdt = rbind(one_hot_out[[1]], one_hot_out[[2]])

#################################
# Step 1: use cardinality matching to find the largest sample of matched pairs for which
# all the covariates are finely balanced.
#################################

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

quantiles(mdt$AGE, 5)
bin_cols = apply(mdt,2,function(x) { all(na.omit(x) %in% 0:1) })

#Discretize continuous columns
mdt[,!bin_cols] = apply(mdt[,!bin_cols],2,quantiles,n_q=5)

#Remove the treatment column
mdt$treatment = NULL

## Global variables for matching
# Solver options
t_max = 5
solver = "gurobi"
approximate = 0
solver = list(name = solver, t_max = t_max, approximate = approximate,
              round_cplex = 0, trace = 0)

# Fine balance
fine = list(covs = mdt)

# Match
matched1 = cardmatch(t_ind, fine = fine, solver = solver)

#Get the indices of the treated units and the matched controls
t_id_1 = matched1$t_id
c_id_1 = matched1$c_id

# Mean balance
covs = cbind(mdt$GENDER.FEMALE, mdt$AGE)
meantab(covs, t_ind, t_id_1, c_id_1)

# Fine balance (note here we are getting an approximate solution)
for (i in 1:ncol(fine_covs)) {
  print(finetab(fine_covs[, i], t_id_1, c_id_1))
}
