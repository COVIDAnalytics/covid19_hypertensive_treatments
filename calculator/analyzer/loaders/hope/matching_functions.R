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

matching_process<-function(data, reference_df, matched_df, t_max, solver, approximate, verbose = FALSE){
  
  #Create a treatment column
  data[[matched_df]]$treatment = 1
  data[[reference_df]]$treatment = 0
  
  t_ind = c(data[[matched_df]]$treatment, data[[reference_df]]$treatment)
  mdt = rbind(data[[matched_df]], data[[reference_df]])
  
  # t_ind[1:1000]=2; t_ind[1001:2000]=1; t_ind[2000:2436]=0
  
  #Remove the treatment column
  mdt$treatment = NULL
  
  # Save the non-discretized data
  mdt0 = mdt
  
  #Find all binary columns
  bin_cols = apply(mdt,2,function(x) { all(na.omit(x) %in% 0:1) })
  #Discretize continuous columns
  mdt[,!bin_cols] = apply(mdt[,!bin_cols],2,quantiles,n_q=5)
  
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
  
  if (verbose) {
    for (i in 1:ncol(mdt)) {
      print(names(mdt)[i])
      print(finetab(mdt[, i], t_id_1, c_id_1))
    }
  }
  
  reference_data = mdt[c_id_1,]
  matched_data = mdt[t_id_1,]
  
  summary_means = meantab(mdt, t_ind, t_id_1, c_id_1)
  
  match_result = list(matched = matched1, reference_data = reference_data, 
                      matched_data = matched_data, 
                      summary_means= summary_means, mdt = mdt, mdt0 = mdt0,
                      t_ind = t_ind, t_id= t_id_1, c_id = c_id_1)
  return(match_result)
}

loveplot_common<- function (treatment_name, X_mat, t_id, t_id_new, c_id, t_id_common, v_line = .1, legend_position = "topright") 
{
  ## Original Treated Group (base treatment)
  X_mat_t_before = X_mat[t_id, ]
  X_mat_t_before_mean = apply(X_mat_t_before, 2, mean)
  X_mat_t_before_var = apply(X_mat_t_before, 2, var)
  
  ## Matched Treated Group
  X_mat_t = X_mat[t_id_new, ]
  X_mat_t_mean = apply(X_mat_t, 2, mean)
  X_mat_t_var = apply(X_mat_t, 2, var)
  
  ## Common Treated Group 
  X_mat_t_common = X_mat[t_id_common, ]
  X_mat_t_common_mean = apply(X_mat_t_common, 2, mean)
  X_mat_t_common_var = apply(X_mat_t_common, 2, var)
  
  ## Original Control Group (dependent on treatment_name)
  X_mat_c_before = X_mat[-t_id, ]
  X_mat_c_before_mean = apply(X_mat_c_before, 2, mean)
  X_mat_c_before_var = apply(X_mat_c_before, 2, var)
  
  ## Matched Control Group
  X_mat_c_after = X_mat[c_id, ]
  X_mat_c_after_mean = apply(X_mat_c_after, 2, mean)
  X_mat_c_after_var = apply(X_mat_c_after, 2, var)
  
  std_dif_before = (X_mat_t_before_mean - X_mat_c_before_mean)/sqrt((X_mat_t_before_var +  X_mat_c_before_var)/2)
  abs_std_dif_before = abs(std_dif_before)
  std_dif_after = (X_mat_t_mean - X_mat_c_after_mean)/sqrt((X_mat_t_var +  X_mat_c_after_var)/2)
  abs_std_dif_after = abs(std_dif_after)
  std_dif_after_common = (X_mat_t_common_mean - X_mat_c_after_mean)/sqrt((X_mat_t_common_var +  X_mat_c_after_var)/2)
  abs_std_dif_after_common = abs(std_dif_after_common)
  
  n_aux = length(abs_std_dif_before)
  
  par(mar = c(3, 3, 3, 3),xpd=FALSE)
  dotchart(abs_std_dif_before[n_aux:1], labels = colnames(X_mat)[n_aux:1], 
           cex = 0.7, pch = " ", color = , main = paste("Matching for treatment ", 
                                                        treatment_name,sep=""), xlim = c(0,0.4), 
           xlab = "Absolute standardized differences in means", bg = par("bg"))
  points(abs_std_dif_before[n_aux:1], y = 1:ncol(X_mat), cex = 0.9, 
         pch = 0)
  points(abs_std_dif_after[n_aux:1], y = 1:ncol(X_mat), cex = 0.8, 
         pch = 8, col = "blue")
  points(abs_std_dif_after_common[n_aux:1], y = 1:ncol(X_mat), cex = 0.8, 
         pch = 8, col = "red")
  legend(legend_position, c("Before matching", "After matching", "After matching - common control"), 
         cex = 0.5, bty = "n", pch = c(0, 8, 8), col = c("black", 
                                                      "blue", "red"), 
         xpd=TRUE, 
         y.intersp=0.1)
  abline(v = v_line, lty = 2)
}


compare_features <- function (df_full, base_treat, var_treat, common_control=c()){
  # data A = base treatment
  label_a = names(out)[base_treat]
  
  # data B = changing treatment ("control")
  label_b = names(out)[var_treat]
  
  data_a = df_full %>%filter(REGIMEN == label_a)
  data_b = df_full %>%filter(REGIMEN == label_b)
  
  nrow(data_a) + nrow(data_b) == nrow(matched_object_list[[to_treat]]$mdt0)
  
  # Filter to t_ids. If common control, adjust the common controls to start at the correct index
  data_a_filtered = data_a[matched_object_list[[to_treat]]$t_id,]
  if (length(common_control)  > 0) {
    data_a_filtered = data_a[common_control,]
  }
  c_id = matched_object_list[[to_treat]]$c_id - nrow(data_a) #adjust down
  data_b_filtered = data_b[c_id,]
  
  ttest_original = run_ttest(data_a, data_b, label_a, label_b, cols_exclude = treatments) %>%
    mutate_if(is.numeric, funs(round(., digits = 3)))
  ttest_filtered = run_ttest(data_a_filtered, data_b_filtered, label_a, label_b, cols_exclude = treatments) %>%
    mutate_if(is.numeric, funs(round(., digits = 3)))
  
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
  
  return(list(original = ttest_original,filtered = ttest_filtered,compare = ttest_compare))
}
