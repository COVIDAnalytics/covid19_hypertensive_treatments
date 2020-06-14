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

loveplot_custom <- function (treatment_name, X_mat, t_id, t_id_new, c_id, v_line, legend_position = "topright") 
{
  X_mat_t_before = X_mat[t_id, ]
  X_mat_t_before_mean = apply(X_mat_t_before, 2, mean)
  X_mat_t_before_var = apply(X_mat_t_before, 2, var)
  X_mat_c_before = X_mat[-t_id, ]
  X_mat_c_before_mean = apply(X_mat_c_before, 2, mean)
  X_mat_c_before_var = apply(X_mat_c_before, 2, var)
  
  std_dif_before = (X_mat_t_before_mean - X_mat_c_before_mean)#/sqrt((X_mat_t_before_var +  X_mat_c_before_var)/2)
  
  X_mat_t = X_mat[t_id_new, ]
  X_mat_t_mean = apply(X_mat_t, 2, mean)
  X_mat_t_var = apply(X_mat_t, 2, var)
  
  X_mat_c_after = X_mat[c_id, ]
  X_mat_c_after_mean = apply(X_mat_c_after, 2, mean)
  std_dif_after = (X_mat_t_mean - X_mat_c_after_mean)#/sqrt((X_mat_t_var +  X_mat_c_before_var)/2)
  
  abs_std_dif_before = abs(std_dif_before)
  n_aux = length(abs_std_dif_before)
  abs_std_dif_after = abs(std_dif_after)
  par(mar = c(3, 3, 3, 3),xpd=FALSE)
  dotchart(abs_std_dif_before[n_aux:1], labels = colnames(X_mat)[n_aux:1], 
           cex = 0.7, pch = " ", color = , main = paste("Matching for treatment ", 
                                                        treatment_name,sep=""), xlim = c(0,0.4), 
           xlab = "Absolute standardized differences in means", bg = par("bg"))
  points(abs_std_dif_before[n_aux:1], y = 1:ncol(X_mat), cex = 0.9, 
         pch = 0)
  points(abs_std_dif_after[n_aux:1], y = 1:ncol(X_mat), cex = 0.8, 
         pch = 8, col = "blue")
  legend(legend_position, c("Before matching", "After matching"), 
         cex = 0.5, bty = "n", pch = c(0, 8), col = c("black", 
                                                      "blue"), 
         xpd=TRUE, y.intersp=0.1)
  abline(v = v_line, lty = 2)
}
