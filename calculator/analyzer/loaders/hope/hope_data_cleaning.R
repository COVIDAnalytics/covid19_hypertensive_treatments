library(dplyr)
library(tidyverse)
library(foreign)
library(data.table)
library(reshape2)
library(scales)

filter_columns<-function(df){
#Categories of features

LOCATION = c('HOSPITAL','COUNTRY')
  

DEMOGRAPHICS = c('GENDER','RACE','PREGNANT','WEIGHT','HEIGHT', 'AGE')

COMORBIDITIES = c('DIABETES', 'HYPERTENSION', 'DISLIPIDEMIA', 'OBESITY',
                      'SMOKING','RENALINSUF','ANYLUNGDISEASE', 'AF', 'VIH', 
                      'TBPASSED', 'ANYHEARTDISEASE','MAINHEARTDISEASE',
                      'ANYCEREBROVASCULARDISEASE', 'CONECTIVEDISEASE','LUNGDISEASE',
                      'LIVER_DISEASE', 'CANCER')

DRUGS_ADMISSIONS = c('HOME_OXIGEN_THERAPY', 'IN_PREVIOUSASPIRIN', 'IN_OTHERANTIPLATELET',
                         'IN_ORALANTICOAGL',  'IN_ACEI_ARB', 'IN_BETABLOCKERS',
                         'IN_BETAGONISTINHALED', 'IN_GLUCORTICOIDSINHALED',
                         'IN_DVITAMINSUPLEMENT','IN_BENZODIACEPINES', 'IN_ANTIDEPRESSANT')

COVID19_TREATMENTS = c('CORTICOSTEROIDS',
                    'CLOROQUINE',
                    'ANTIVIRAL',
                    'INTERFERONOR',
                    'TOCILIZUMAB', 
                    'ANTIBIOTICS',
                    'ACEI_ARBS', 
                    'ANTICOAGULANTS',
                    'ANTICOAGULANTS_TYPE',
                    'OTHER_RELEVANT_COVID19_DRUGS',
                    'CLOROQUINE_DATE', 
                    'ANTIVIRAL_DATE',
                    'TOCILIZUMAB_DATE')

ADD_COVID19_TREATMENTS = c('CORTICOSTEROIDS',
                       'INTERFERONOR',
                       'TOCILIZUMAB', 
                       'ANTIBIOTICS',
                       'ACEI_ARBS')

#Column: 'OTHER_RELEVANT_COVID19_DRUGS' might give us some additional details

VITALS = c('FAST_BREATHING','MAXTEMPERATURE_ADMISSION','SAT02_BELOW92')

BINARY_LABS_VITALS_ADMISSION = c('DDDIMER_B',
                                     'PROCALCITONIN_B', 'PCR_B', 'TN_B',
                                     'TRANSAMINASES_B', 'FERRITINE_B',
                                     'TRIGLYCERIDES_B', 'LDL_B',
                                     'BLOOD_PRESSURE_ABNORMAL_B')

CONTINUE_LABS_ADMISSION = c('CREATININE','ARTERIALBLOODGASPH',
                                'ARTERIALBLOODGASPA02', 'ARTERIALBLOODGASPAC02',
                                'ARTERIALBLOODGAS02SATURATION','SODIUM',
                                'LEUCOCYTES', 'LYMPHOCYTES',
                                'HEMOGLOBIN','PLATELETS',
                                'GLASGOW_COMA_SCORE')

XRAY_RESULTS = c('CHESTXRAY_BNORMALITY')

O2_PROCEDURES = c('X02DURINGADMISSION', 'HIGHFLOWNASALCANNULA',
                   'NONINVASIVEMECHANICALVENTILATION',
                   'INVASIVEMECHANICALVENTILATION', 'DAYS_ON_MECHANICALVENTILATION',
                   'PRONEDURINGADMISSION', 'CIRCULATORYORECMOSUPPORT',
                   'ECMO_SIMILAR_SUPPORT')

OUTCOMES = c('ICUADMISSION','RESPIRATORY_INSUFFICIENCY',
                 'HEARTFAILURE', 'RENALFAILURE',
                 'UPPER_RESPIRATORY_TRACT_INFECTION', 'PNEUMONIA',
                 'SEPSIS', 'SYSTEMIC_INFLAMATORY_REPONSE_SYNDROME',
                 'ANYRELEVANT_BLEEDING', 'HEMOPTYSIS', 'EMBOLIC_EVENT',
                 'CO_DEATHCAUSE', 'DEATH','DISCHARGE_TO','FOLLOWUPDAYS')

DATES = c('DT_ONSETSYMPTOMS','DT_TEST_COVID',
              'DT_HOSPITAL_ADMISSION')

#df <- read.csv("~/Dropbox (Personal)/COVID_clinical/covid19_hope/hope_data_clean.csv", header=TRUE)

SELECTED_TREATMENTS <- c('CLOROQUINE',
                         'ANTIVIRAL',
                         'ANTICOAGULANTS')
  
SELECTED_OUTCOMES <- c('HEARTFAILURE', 'RENALFAILURE','SEPSIS', 'EMBOLIC_EVENT','DEATH')

cols_include = c(LOCATION, DATES, DEMOGRAPHICS, COMORBIDITIES, 
                 DRUGS_ADMISSIONS, VITALS,
                 BINARY_LABS_VITALS_ADMISSION, CONTINUE_LABS_ADMISSION,
                 XRAY_RESULTS, ADD_COVID19_TREATMENTS, SELECTED_TREATMENTS, SELECTED_OUTCOMES)
data<-df[,cols_include]

return(data)
}

#First we will edit the dates to include two variables that indicate the difference between the date of admission and the date of symptom onset and pcr test

clean_columns<-function(data){
data <- data %>%
          mutate(ONSET_DATE_DIFF = as.numeric(as.Date(DT_HOSPITAL_ADMISSION) -as.Date(DT_ONSETSYMPTOMS)))%>%
          mutate(ONSET_DATE_DIFF = replace(ONSET_DATE_DIFF, ONSET_DATE_DIFF<0, NA))%>%
          mutate(TEST_DATE_DIFF = as.numeric(as.Date(DT_HOSPITAL_ADMISSION) -as.Date(DT_TEST_COVID)))%>%
          mutate(TEST_DATE_DIFF = replace(TEST_DATE_DIFF, TEST_DATE_DIFF<0, NA))%>%
          select(-DT_ONSETSYMPTOMS, -DT_TEST_COVID)%>%
          mutate(DT_HOSPITAL_ADMISSION = as.Date(DT_HOSPITAL_ADMISSION))

#Clean MAINHEARTDISEASE
data <- data %>%
  mutate(MAINHEARTDISEASE = as.character(MAINHEARTDISEASE),
         MAINHEARTDISEASE = replace(MAINHEARTDISEASE, MAINHEARTDISEASE=='HEARTFAILURE/MYOPATHY', 'HEARTFAILURE-MYOPATHY'),
         MAINHEARTDISEASE = replace(MAINHEARTDISEASE, MAINHEARTDISEASE=='NODISCLOSED', 'OTHER'))

#Clean ARTERIALBLOODGASPH
data <- data %>%
       mutate(ARTERIALBLOODGASPH = as.numeric(as.character(ARTERIALBLOODGASPH)),
              ARTERIALBLOODGASPH = if_else(ARTERIALBLOODGASPH > 1000, ARTERIALBLOODGASPH/1000, ARTERIALBLOODGASPH))

#Clean CHEST X RAY AND CREATININE
data <- data %>%
  mutate(CHESTXRAY_BNORMALITY = replace(CHESTXRAY_BNORMALITY, CHESTXRAY_BNORMALITY=="", NA),
         CREATININE = replace(CREATININE, CREATININE=="0m7", NA),
         CREATININE = replace(CREATININE, CREATININE=="1.-4", "1.4"),
         CREATININE = replace(CREATININE, CREATININE=="22.03.2020", NA),
         CREATININE = replace(CREATININE, CREATININE=="2..6", "2.06"),
         CREATININE = replace(CREATININE, CREATININE=="1..02", "1.02"),
         CREATININE = replace(CREATININE, CREATININE=="0..74", "0.74"),
         CREATININE = as.numeric(as.character(CREATININE)))

#CLEAN MAXTEMPERATURE_ADMISSION

data <- data %>%
  mutate(MAXTEMPERATURE_ADMISSION = replace(MAXTEMPERATURE_ADMISSION, MAXTEMPERATURE_ADMISSION=="", NA),
         MAXTEMPERATURE_ADMISSION = replace(MAXTEMPERATURE_ADMISSION, MAXTEMPERATURE_ADMISSION=="NORMAL", "36.5"),
         MAXTEMPERATURE_ADMISSION = replace(MAXTEMPERATURE_ADMISSION, MAXTEMPERATURE_ADMISSION=="NO TERMOME", NA),
         MAXTEMPERATURE_ADMISSION = str_replace(MAXTEMPERATURE_ADMISSION,"Âº",""),
         MAXTEMPERATURE_ADMISSION = str_replace(MAXTEMPERATURE_ADMISSION,"Âª",""),
         MAXTEMPERATURE_ADMISSION = str_replace(MAXTEMPERATURE_ADMISSION,"Â´",""),
         MAXTEMPERATURE_ADMISSION = str_replace(MAXTEMPERATURE_ADMISSION,"Â°C",""),
         MAXTEMPERATURE_ADMISSION = str_replace(MAXTEMPERATURE_ADMISSION,"C",""),
         MAXTEMPERATURE_ADMISSION = str_replace(MAXTEMPERATURE_ADMISSION,",","."),
         MAXTEMPERATURE_ADMISSION = str_replace(MAXTEMPERATURE_ADMISSION,"-","."),
         MAXTEMPERATURE_ADMISSION = str_replace(MAXTEMPERATURE_ADMISSION,"36Â´8","36.8"),
         MAXTEMPERATURE_ADMISSION = str_replace(MAXTEMPERATURE_ADMISSION,"3709","37.9"),
         MAXTEMPERATURE_ADMISSION = replace(MAXTEMPERATURE_ADMISSION, MAXTEMPERATURE_ADMISSION=="357", "35.7"),
         MAXTEMPERATURE_ADMISSION = replace(MAXTEMPERATURE_ADMISSION, MAXTEMPERATURE_ADMISSION=="n", NA),
         MAXTEMPERATURE_ADMISSION = replace(MAXTEMPERATURE_ADMISSION, MAXTEMPERATURE_ADMISSION=="N", NA),
         MAXTEMPERATURE_ADMISSION = replace(MAXTEMPERATURE_ADMISSION, MAXTEMPERATURE_ADMISSION=="384", "38.4"),
         MAXTEMPERATURE_ADMISSION = replace(MAXTEMPERATURE_ADMISSION, MAXTEMPERATURE_ADMISSION==">38", NA),
         MAXTEMPERATURE_ADMISSION = as.numeric(as.character(MAXTEMPERATURE_ADMISSION)))

data <- data %>%
  mutate(CLOROQUINE = coalesce(CLOROQUINE, 0),
         ANTIVIRAL = coalesce(ANTIVIRAL, 0),
         ANTICOAGULANTS = if_else(coalesce(ANTICOAGULANTS, 0) > 0, 1, 0)) %>%
  mutate(REGIMEN = if_else(CLOROQUINE == 0, "Non-Chloroquine", ## No Chloroquine
                           if_else(ANTIVIRAL == 0,
                                   if_else(ANTICOAGULANTS == 0, "Chloroquine Only",
                                           "Chloroquine and Anticoagulants"),
                                   if_else(ANTICOAGULANTS == 0, "Chloroquine and Antivirals",
                                           "All"))))
                                           

data <- data %>%
  mutate(COMORB_DEATH = pmax(HEARTFAILURE, RENALFAILURE,SEPSIS, EMBOLIC_EVENT,DEATH, na.rm = TRUE)) %>%
  select(-c('HEARTFAILURE', 'RENALFAILURE','SEPSIS', 'EMBOLIC_EVENT'))

return(data)
}

remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, c(filter_lb, filter_ub), na.rm = TRUE)
  y <- x
  y[x < (qnt[1])] <- NA
  y[x > (qnt[2])] <- NA
  y
} 



filter_outliers<-function(data, filter_lb, filter_ub){
  #Create a dataframe with the bounds and the median
  bounds_df = data.frame(feature=character(),
                   median_val=numeric(), 
                   lb=numeric(),
                   up=numeric())
  
  #Select the appropriate columns
  nums <- unlist(lapply(data, is.numeric))  
  vals = apply(data,2,function(x) { all(length(unique(x)) >7) })
  cols_filter = (nums & vals)
  #Create a matrix with the applied filters
  bounds_df = apply(data[,cols_filter],2,function(x) { quantile(x, c(filter_lb, .5, filter_ub), na.rm = TRUE)  })
  data[,cols_filter] = apply(data[,cols_filter],2,remove_outliers)
  
  d = list(data, bounds_df)
  
  return(d)
}

filter_outliers<-function(data, filter_lb, filter_ub){
  #Create a dataframe with the bounds and the median
  bounds_df = data.frame(feature=character(),
                         median_val=numeric(), 
                         lb=numeric(),
                         up=numeric())
  
  #Select the appropriate columns
  nums <- unlist(lapply(data, is.numeric))  
  vals = apply(data,2,function(x) { all(length(unique(x)) >7) })
  cols_filter = (nums & vals)
  #Create a matrix with the applied filters
  bounds_df = apply(data[,cols_filter],2,function(x) { quantile(x, c(filter_lb, .5, filter_ub), na.rm = TRUE)  })
  data[,cols_filter] = apply(data[,cols_filter],2,remove_outliers)
  
  d = list(data, bounds_df)
  
  return(d)
}

filter_missing<-function(data, threshold){
  na_counts = t(data %>%
                  select(everything()) %>% 
                  summarise_all(funs(sum(is.na(.))/nrow(data)))) %>% 
    as.data.frame() %>%
    mutate(Feature = row.names(.)) %>%
    select(Feature, Missing_Proportion = V1)
  cols_filled = na_counts %>% filter(Missing_Proportion <= threshold) %>% pull(Feature)
  print("Excluded Columns: ")
  print(paste(setdiff(names(data), cols_filled), collapse = ", "))
  d = list(data[cols_filled], na_counts)
  return(d)
}


