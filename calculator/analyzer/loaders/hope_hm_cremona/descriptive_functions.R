is.binary <- function(col){
  # print(col)
  t = sort(unique(col))
  if (is.numeric(t)){
    if (length(t) <= 2 & (min(t) %in% c(0,1) & max(t) %in% c(0,1))) {
      return(TRUE)
    }
  }
  return(FALSE)
}

descriptive_table <- function(data, short_version = FALSE){
  numeric_binary <- data %>% select_if(is.binary)
  numeric_cont <- data %>% select_if(negate(is.binary))  %>% select_if(is.numeric)
  categorical <- data %>% select_if(negate(is.numeric))
  
  cont_summary <- numeric_cont %>% 
    summarize_all(funs(quantile(., c(0,.25,.5,.75,1.0), na.rm = TRUE))) %>%
    rbind(numeric_cont %>% 
            summarize_all(funs(mean(., na.rm = TRUE)))) %>%
    rbind(numeric_cont %>% 
            summarize_all(funs(sum(is.na(.))/nrow(data)))) %>%
    `row.names<-`(c('min', 'q1', 'med', 'q3', 'max','mean','Missing')) %>% 
    t() %>%
    as.data.frame() %>%
    mutate(Feature = rownames(.)) %>%
    mutate_if(is.numeric, funs(round(., digits = 1))) %>%
    mutate(Summary = paste0(med, " (",  q1, "-", q3, ")"))
  
  bin_summary <- numeric_binary %>% 
    summarize_all(funs(quantile(., c(0,.25,.5,.75,1.0), na.rm = TRUE))) %>%
    rbind(numeric_binary %>% 
            summarize_all(funs(mean(., na.rm = TRUE)))) %>%
    rbind(numeric_binary %>% 
            summarize_all(funs(sum(is.na(.))/nrow(data)))) %>%
    `row.names<-`(c('min', 'q1', 'med', 'q3', 'max','mean','Missing')) %>% 
    t() %>%
    as.data.frame() %>%
    mutate(Feature = rownames(.)) %>%
    mutate(Summary = paste0(round(mean*nrow(numeric_binary)), " (", round(mean*100,1), "%)")) 
  
  num_summary <- rbind(cont_summary, bin_summary) 
  if (short_version){
    num_summary <- num_summary %>% 
      mutate(Missing = paste0(round(Missing*100,1), "%")) %>%
      dplyr::select('Feature','Summary','Missing')
  }
  
  if (length(categorical) > 0) {
    cat_summary <- do.call(rbind, lapply(names(categorical), function(col){
      t <- (table(categorical[col], useNA = "ifany")/nrow(data)) %>% as.data.frame() %>% 
        mutate(Feature = col) %>% 
        dplyr::select(Feature, Value = Var1, Freq)
      return(t)
    }))
    return(list(num_summary, cat_summary))
  } else {
    return(list(num_summary))
  }
}

run_ttest <- function(data_a, data_b, label_a, label_b, cols_exclude=c()){
  t_a = descriptive_table(data_a, short_version = TRUE)[[1]] %>%
    `colnames<-`(c('Feature', paste0("Summary_",label_a), paste0("Missing_",label_a)))
  t_b = descriptive_table(data_b, short_version = TRUE)[[1]] %>%
    `colnames<-`(c('Feature', paste0("Summary_",label_b), paste0("Missing_",label_b)))
  
  ttest <- do.call(rbind, lapply(setdiff(names(data_a),cols_exclude), function(col){
    # print(col)
    if (is.numeric(data_a[[col]])){
      r = t.test(data_a[[col]], data_b[[col]], var.equal = FALSE, na.action = "na.omit")
      return(c(col, r$p.value))
    }})) %>%
    as.data.frame() %>%
    `colnames<-`(c('Feature','P-Value'))
  
  pop_compare <- list(t_a, t_b, ttest) %>%
    reduce(full_join, by = 'Feature') %>%
    mutate_if(is.factor, ~ as.numeric(as.character(.x)))
  
  return(pop_compare)
}
