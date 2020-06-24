library(dplyr)
library(tidyverse)
library(foreign)
library(data.table)
library(reshape2)
library(scales)
library(mice)
library(caret)
library(imputeMissings)
library(stringr)

#Define variable categories

LOCATION = c('HOSPITAL','COUNTRY')

DEMOGRAPHICS = c('GENDER','RACE','PREGNANT','WEIGHT','HEIGHT', 'AGE')

COMORBIDITIES = c('DIABETES', 'HYPERTENSION', 'DISLIPIDEMIA', 'OBESITY',
                  'SMOKING','RENALINSUF','ANYLUNGDISEASE', 'AF', 'VIH', 
                  'TBPASSED', 'ANYHEARTDISEASE','MAINHEARTDISEASE',
                  'ANYCEREBROVASCULARDISEASE', 'CONECTIVEDISEASE',
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



filter_columns<-function(df){
#Categories of features

LOCATION = c('HOSPITAL', 'COUNTRY')

DEMOGRAPHICS = c('GENDER', 'RACE', 'PREGNANT', 'WEIGHT', 'HEIGHT', 'AGE')
SELECTED_DEMOGRAPHICS = c('GENDER', 'RACE', 'AGE')

COMORBIDITIES = c('DIABETES', 'HYPERTENSION', 'DISLIPIDEMIA', 'OBESITY',
                  'SMOKING','RENALINSUF','ANYLUNGDISEASE', 'AF', 'VIH', 
                  'TBPASSED', 'ANYHEARTDISEASE','MAINHEARTDISEASE',
                  'ANYCEREBROVASCULARDISEASE', 'CONECTIVEDISEASE',
                  'LIVER_DISEASE', 'CANCER')
SELECTED_COMORBIDITIES = c('DIABETES', 'HYPERTENSION', 'DISLIPIDEMIA', 'OBESITY',
                          'RENALINSUF','ANYLUNGDISEASE', 'AF', 'VIH', 
                          'ANYHEARTDISEASE', 'ANYCEREBROVASCULARDISEASE', 
                          'CONECTIVEDISEASE', 'LIVER_DISEASE', 'CANCER')

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

SELECTED_TREATMENTS <- c('CLOROQUINE',
                         'ANTIVIRAL',
                         'ANTICOAGULANTS')
SELECTED_TREATMENTS_NONHOPE <- c('CLOROQUINE',
                         'ANTIVIRAL',
                         'ANTICOAGULANTS',
                         'REGIMEN')

VITALS = c('FAST_BREATHING','MAXTEMPERATURE_ADMISSION','SAT02_BELOW92')
SELECTED_VITALS = c('MAXTEMPERATURE_ADMISSION','SAT02_BELOW92')

BINARY_LABS_VITALS_ADMISSION = c('DDDIMER_B',
                                 'PROCALCITONIN_B', 'PCR_B', 'TN_B',
                                 'TRANSAMINASES_B', 'FERRITINE_B',
                                 'TRIGLYCERIDES_B', 'LDL_B',
                                 'BLOOD_PRESSURE_ABNORMAL_B')
SELECTED_BINARY_LABS_VITALS_ADMISSION = c('DDDIMER_B',
                                          'PROCALCITONIN_B', 'PCR_B',
                                          'TRANSAMINASES_B', 'LDL_B',
                                          'BLOOD_PRESSURE_ABNORMAL_B')

CONTINUE_LABS_ADMISSION = c('CREATININE','ARTERIALBLOODGASPH',
                            'ARTERIALBLOODGASPA02', 'ARTERIALBLOODGASPAC02',
                            'ARTERIALBLOODGAS02SATURATION','SODIUM',
                            'LEUCOCYTES', 'LYMPHOCYTES',
                            'HEMOGLOBIN','PLATELETS',
                            'GLASGOW_COMA_SCORE')
SELECTED_CONTINUE_LABS_ADMISSION = c('CREATININE', 'SODIUM',
                            'LEUCOCYTES', 'LYMPHOCYTES',
                            'HEMOGLOBIN','PLATELETS')

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
SELECTED_OUTCOMES <- c('HEARTFAILURE', 'RENALFAILURE','SEPSIS', 'EMBOLIC_EVENT','DEATH')

SELECTED_OUTCOMES_NONHOPE <- c('COMORBID','DEATH')

DATES = c('DT_ONSETSYMPTOMS','DT_TEST_COVID',
              'DT_HOSPITAL_ADMISSION')
SELECTED_DATES = c('DT_HOSPITAL_ADMISSION')

cols_include = c(LOCATION, SELECTED_DATES, SELECTED_DEMOGRAPHICS, 
                 SELECTED_COMORBIDITIES, DRUGS_ADMISSIONS, SELECTED_VITALS,
                 SELECTED_BINARY_LABS_VITALS_ADMISSION, SELECTED_CONTINUE_LABS_ADMISSION,
                 ADD_COVID19_TREATMENTS, SELECTED_TREATMENTS, SELECTED_OUTCOMES)

data <- df[,cols_include]

df[,COMORBIDITIES]

return(data)
}

filter_columns_nonhope<-function(df){
  #Categories of features
  
  LOCATION = c('HOSPITAL', 'COUNTRY')
  
  DEMOGRAPHICS = c('GENDER', 'RACE', 'PREGNANT', 'WEIGHT', 'HEIGHT', 'AGE')
  SELECTED_DEMOGRAPHICS = c('GENDER', 'RACE', 'AGE')
  
  COMORBIDITIES = c('DIABETES', 'HYPERTENSION', 'DISLIPIDEMIA', 'OBESITY',
                    'SMOKING','RENALINSUF','ANYLUNGDISEASE', 'AF', 'VIH', 
                    'TBPASSED', 'ANYHEARTDISEASE','MAINHEARTDISEASE',
                    'ANYCEREBROVASCULARDISEASE', 'CONECTIVEDISEASE',
                    'LIVER_DISEASE', 'CANCER')
  SELECTED_COMORBIDITIES = c('DIABETES', 'HYPERTENSION', 'DISLIPIDEMIA', 'OBESITY',
                             'RENALINSUF','ANYLUNGDISEASE', 'AF', 'VIH', 
                             'ANYHEARTDISEASE', 'ANYCEREBROVASCULARDISEASE', 
                             'CONECTIVEDISEASE', 'LIVER_DISEASE', 'CANCER')
  
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
  
  SELECTED_TREATMENTS <- c('CLOROQUINE',
                           'ANTIVIRAL',
                           'ANTICOAGULANTS')
  SELECTED_TREATMENTS_NONHOPE <- c('CLOROQUINE',
                                   'ANTIVIRAL',
                                   'ANTICOAGULANTS',
                                   'REGIMEN')
  
  VITALS = c('FAST_BREATHING','MAXTEMPERATURE_ADMISSION','SAT02_BELOW92')
  SELECTED_VITALS = c('MAXTEMPERATURE_ADMISSION','SAT02_BELOW92')
  
  BINARY_LABS_VITALS_ADMISSION = c('DDDIMER_B',
                                   'PROCALCITONIN_B', 'PCR_B', 'TN_B',
                                   'TRANSAMINASES_B', 'FERRITINE_B',
                                   'TRIGLYCERIDES_B', 'LDL_B',
                                   'BLOOD_PRESSURE_ABNORMAL_B')
  SELECTED_BINARY_LABS_VITALS_ADMISSION = c('DDDIMER_B',
                                   'PROCALCITONIN_B', 'PCR_B',
                                   'TRANSAMINASES_B', 'LDL_B',
                                   'BLOOD_PRESSURE_ABNORMAL_B')
  
  CONTINUE_LABS_ADMISSION = c('CREATININE','ARTERIALBLOODGASPH',
                              'ARTERIALBLOODGASPA02', 'ARTERIALBLOODGASPAC02',
                              'ARTERIALBLOODGAS02SATURATION','SODIUM',
                              'LEUCOCYTES', 'LYMPHOCYTES',
                              'HEMOGLOBIN','PLATELETS',
                              'GLASGOW_COMA_SCORE')
  SELECTED_CONTINUE_LABS_ADMISSION = c('CREATININE', 'SODIUM',
                                       'LEUCOCYTES', 'LYMPHOCYTES',
                                       'HEMOGLOBIN','PLATELETS')
  
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
  SELECTED_OUTCOMES <- c('HEARTFAILURE', 'RENALFAILURE','SEPSIS', 'EMBOLIC_EVENT','DEATH')
  
  SELECTED_OUTCOMES_NONHOPE <- c('COMORB_DEATH','DEATH')
  
  DATES = c('DT_ONSETSYMPTOMS','DT_TEST_COVID',
            'DT_HOSPITAL_ADMISSION')
  SELECTED_DATES = c('DT_HOSPITAL_ADMISSION')
  
  cols_include = c(LOCATION, SELECTED_DATES, SELECTED_DEMOGRAPHICS,
                   SELECTED_COMORBIDITIES, DRUGS_ADMISSIONS, SELECTED_VITALS,
                   SELECTED_BINARY_LABS_VITALS_ADMISSION, SELECTED_CONTINUE_LABS_ADMISSION,
                   ADD_COVID19_TREATMENTS, SELECTED_TREATMENTS_NONHOPE, SELECTED_OUTCOMES_NONHOPE)
  
  data <- df[,cols_include]
  
  df[,SELECTED_COMORBIDITIES]
  
  return(data)
}

#First we will edit the dates to include two variables that indicate the difference between the date of admission and the date of symptom onset and pcr test

clean_columns<-function(data){
data <- data %>%
          # mutate(ONSET_DATE_DIFF = as.numeric(as.Date(DT_HOSPITAL_ADMISSION) -as.Date(DT_ONSETSYMPTOMS)))%>%
          # mutate(ONSET_DATE_DIFF = replace(ONSET_DATE_DIFF, ONSET_DATE_DIFF<0, NA))%>%
          # mutate(TEST_DATE_DIFF = as.numeric(as.Date(DT_HOSPITAL_ADMISSION) -as.Date(DT_TEST_COVID)))%>%
          # mutate(TEST_DATE_DIFF = replace(TEST_DATE_DIFF, TEST_DATE_DIFF<0, NA))%>%
          # dplyr::select(-DT_ONSETSYMPTOMS, -DT_TEST_COVID)%>%
          mutate(DT_HOSPITAL_ADMISSION = as.Date(DT_HOSPITAL_ADMISSION))

#Clean MAINHEARTDISEASE
# data <- data %>%
#   mutate(MAINHEARTDISEASE = as.character(MAINHEARTDISEASE),
#          MAINHEARTDISEASE = replace(MAINHEARTDISEASE, MAINHEARTDISEASE=='HEARTFAILURE/MYOPATHY', 'HEARTFAILURE-MYOPATHY'),
#          MAINHEARTDISEASE = replace(MAINHEARTDISEASE, MAINHEARTDISEASE=='NODISCLOSED', 'OTHER'))

#Clean ARTERIALBLOODGASPH
# data <- data %>%
#        mutate(ARTERIALBLOODGASPH = as.numeric(as.character(ARTERIALBLOODGASPH)),
#               ARTERIALBLOODGASPH = if_else(ARTERIALBLOODGASPH > 1000, ARTERIALBLOODGASPH/1000, ARTERIALBLOODGASPH))

#Clean CHEST X RAY AND CREATININE
# data <- data %>%
#   mutate(CHESTXRAY_BNORMALITY = replace(CHESTXRAY_BNORMALITY, CHESTXRAY_BNORMALITY=="", NA),
#          CREATININE = replace(CREATININE, CREATININE=="0m7", NA),
#          CREATININE = replace(CREATININE, CREATININE=="1.-4", "1.4"),
#          CREATININE = replace(CREATININE, CREATININE=="22.03.2020", NA),
#          CREATININE = replace(CREATININE, CREATININE=="2..6", "2.06"),
#          CREATININE = replace(CREATININE, CREATININE=="1..02", "1.02"),
#          CREATININE = replace(CREATININE, CREATININE=="0..74", "0.74"),
#          CREATININE = as.numeric(as.character(CREATININE)))

#Clean CREATININE
data <- data %>%
  mutate(CREATININE = replace(CREATININE, CREATININE=="0m7", NA),
         CREATININE = replace(CREATININE, CREATININE=="1.-4", "1.4"),
         CREATININE = replace(CREATININE, CREATININE=="22.03.2020", NA),
         CREATININE = replace(CREATININE, CREATININE=="2..6", "2.06"),
         CREATININE = replace(CREATININE, CREATININE=="1..02", "1.02"),
         CREATININE = replace(CREATININE, CREATININE=="0..74", "0.74"),
         CREATININE = as.numeric(as.character(CREATININE)))

#Clean MAXTEMPERATURE_ADMISSION
data = data_hope
data <- data %>%
  mutate(MAXTEMPERATURE_ADMISSION = replace(MAXTEMPERATURE_ADMISSION, MAXTEMPERATURE_ADMISSION=="", NA),
         MAXTEMPERATURE_ADMISSION = replace(MAXTEMPERATURE_ADMISSION, MAXTEMPERATURE_ADMISSION=="NORMAL", "36.5"),
         MAXTEMPERATURE_ADMISSION = replace(MAXTEMPERATURE_ADMISSION, MAXTEMPERATURE_ADMISSION=="NO TERMOME", NA),
         MAXTEMPERATURE_ADMISSION = gsub("????","",MAXTEMPERATURE_ADMISSION),
         MAXTEMPERATURE_ADMISSION = gsub("????C","",MAXTEMPERATURE_ADMISSION),
         MAXTEMPERATURE_ADMISSION = gsub("C","",MAXTEMPERATURE_ADMISSION),
         MAXTEMPERATURE_ADMISSION = gsub(",",".",MAXTEMPERATURE_ADMISSION),
         MAXTEMPERATURE_ADMISSION = gsub(">36.8","38",MAXTEMPERATURE_ADMISSION),
         MAXTEMPERATURE_ADMISSION = gsub("-",".",MAXTEMPERATURE_ADMISSION),
         MAXTEMPERATURE_ADMISSION = gsub("36????8","36.8",MAXTEMPERATURE_ADMISSION),
         MAXTEMPERATURE_ADMISSION = gsub("3709","37.9",MAXTEMPERATURE_ADMISSION),
         MAXTEMPERATURE_ADMISSION = gsub("????","",MAXTEMPERATURE_ADMISSION),
         MAXTEMPERATURE_ADMISSION = gsub("Âº","",MAXTEMPERATURE_ADMISSION),
         MAXTEMPERATURE_ADMISSION = gsub("Â°","",MAXTEMPERATURE_ADMISSION),
         MAXTEMPERATURE_ADMISSION = gsub("Âª","",MAXTEMPERATURE_ADMISSION),
         MAXTEMPERATURE_ADMISSION = gsub("Â´",".",MAXTEMPERATURE_ADMISSION),
         MAXTEMPERATURE_ADMISSION = replace(MAXTEMPERATURE_ADMISSION, MAXTEMPERATURE_ADMISSION=="357", "35.7"),
         MAXTEMPERATURE_ADMISSION = replace(MAXTEMPERATURE_ADMISSION, MAXTEMPERATURE_ADMISSION=="n", NA),
         MAXTEMPERATURE_ADMISSION = replace(MAXTEMPERATURE_ADMISSION, MAXTEMPERATURE_ADMISSION=="N", NA),
         MAXTEMPERATURE_ADMISSION = replace(MAXTEMPERATURE_ADMISSION, MAXTEMPERATURE_ADMISSION=="384", "38.4"),
         MAXTEMPERATURE_ADMISSION = replace(MAXTEMPERATURE_ADMISSION, MAXTEMPERATURE_ADMISSION=="362", "36.2"),
         MAXTEMPERATURE_ADMISSION = replace(MAXTEMPERATURE_ADMISSION, MAXTEMPERATURE_ADMISSION=="369", "36.9"),
         MAXTEMPERATURE_ADMISSION = replace(MAXTEMPERATURE_ADMISSION, MAXTEMPERATURE_ADMISSION==">38", NA),
         MAXTEMPERATURE_ADMISSION = str_replace(MAXTEMPERATURE_ADMISSION, "36.8.*", "36.8"),
         MAXTEMPERATURE_ADMISSION = str_replace(MAXTEMPERATURE_ADMISSION, ">", ""),
         MAXTEMPERATURE_ADMISSION = as.numeric(as.character(MAXTEMPERATURE_ADMISSION)))

#Determine regimen
data <- data %>%
  mutate(CLOROQUINE = coalesce(CLOROQUINE, 0L),
         ANTIVIRAL = coalesce(ANTIVIRAL, 0L),
         ANTICOAGULANTS = if_else(coalesce(ANTICOAGULANTS, 0L) > 0, 1, 0)) %>%
  mutate(REGIMEN = if_else(CLOROQUINE == 0, "Non-Chloroquine", ## No Chloroquine
                           if_else(ANTIVIRAL == 0,
                                   if_else(ANTICOAGULANTS == 0, "Chloroquine Only",
                                           "Chloroquine and Anticoagulants"),
                                   if_else(ANTICOAGULANTS == 0, "Chloroquine and Antivirals",
                                           "All"))))
                                           
data <- data %>%
  mutate(COMORB_DEATH = pmax(HEARTFAILURE, RENALFAILURE,SEPSIS, EMBOLIC_EVENT, DEATH, na.rm = TRUE)) %>%
  dplyr::select(-c('HEARTFAILURE', 'RENALFAILURE','SEPSIS', 'EMBOLIC_EVENT'))

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

imputation <- function(dat, reps, maxiterations, group, treatments, outcomes, indicator_cols){
  
  #Select the appropriate columns to impute
  DF = dat %>% 
          filter(SOURCE_COUNTRY %in% group)%>%
          select(-treatments,-outcomes,-indicator_cols)
  
  #Convert characters to factors
  DF[sapply(DF, is.character)] <- lapply(DF[sapply(DF, is.character)], as.factor)
  
  #Impute the data first with CART
  tempData <- mice(DF,m=reps,maxit=maxiterations,meth='cart',seed=500)
  completedData <- complete(tempData,1)
  
  #Impute categorical values with the mode in case they were left NA
  completedData = impute(completedData,method = "median/mode", flag=FALSE)
  
  #Combine back the data into the aggregate columns
  rel_dat = dat%>% filter(SOURCE_COUNTRY %in% group)
  
  completedData[,treatments] = rel_dat[,treatments]
  completedData[,outcomes] = rel_dat[,outcomes]

  return(completedData)
}



