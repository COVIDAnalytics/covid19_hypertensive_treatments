library(dplyr)
library(tidyverse)
library(foreign)
library(data.table)
library(reshape2)

convert_zero_one<-function(dat, columns){
  x = dat[,c(columns,'UUID')]
  x = reshape2::dcast(
    dplyr::mutate(
      reshape2::melt(x,id.var="UUID"),
      value=plyr::mapvalues(
        value, c("YES","NO"),c(1,0))
    ),UUID~variable)
 return(x[,columns]) 
}

create_data<-function(save_path){
  df <- read.csv(paste(save_path,"hope_data.csv", sep=""), header=TRUE)
  
  i <- sapply(df, is.factor)
  df[i] <- lapply(df[i], as.character)
  
  #Split columns in categories
  LOCATION = c('INV_HOSPITAL1','INV_COUNTRY1')
  LOCATION_new = c('HOSPITAL','COUNTRY')
  setnames(df, old = LOCATION, new = LOCATION_new)
  
  DEMOGRAPHICS = c('CO_GENDER','CO_RACE','PREGNANT','VL_WEIGHT','VL_HEIGHT', 'EDAD')
  DEMOGRAPHICS_new = c('GENDER','RACE','PREGNANT','WEIGHT','HEIGHT', 'AGE')
  setnames(df, old = DEMOGRAPHICS, new = DEMOGRAPHICS_new)
  
  
  #Excluded: 'CO_DM', 'CO_CURRENTSMOKER','CO_TYPEOFCANCER', 'IN_TABACOSIONO',IN_ANYIMMUNOSUPRESSIONCONDITION, DS_OTHERRELEVANTANTECEDENT
  COMORBIDITIES = c('INDM', 'IN_HYPERTENSION', 'IN_DISPLEMIA', 'IN_OBESITY',
                    'IN_TABACONUMERICO',
                    'IN_RENALINSUF','IN_ANYLUNGDISEASE', 'AF', 'VIH', 
                    'TBPASSED', 'IN_ANYHEARTDISEASE',
                    'CO_MAINHEARTDISEASE',
                    'IN_ANYCEREBROVASCULARDISEASE', 'IN_CONECTIVEDISEASE',
                    'IN_LIVER_DISEASE', 'IN_ANYCANCER')
  COMORBIDITIES_new = c('DIABETES', 'HYPERTENSION', 'DISLIPIDEMIA', 'OBESITY',
                        'SMOKING','RENALINSUF','ANYLUNGDISEASE', 'AF', 'VIH', 
                        'TBPASSED', 'ANYHEARTDISEASE','MAINHEARTDISEASE',
                        'ANYCEREBROVASCULARDISEASE', 'CONECTIVEDISEASE',
                        'LIVER_DISEASE', 'CANCER')
  setnames(df, old = COMORBIDITIES, new = COMORBIDITIES_new)
  df[,COMORBIDITIES_new] = convert_zero_one(df, COMORBIDITIES_new)
  
  #We will exclude allergies for now.
  OTHER = c('ANYDRUGALLERGY','DS_KNOWNALLERGIES')
  
  #Exclude the type and details of anticoagulation: 'IN_ORALANTICOAGL_TYPE', 'CO_ORALANTICOAGL_TYPE_OAC', 'DS_OTHERPREVIOUSTREATMENT'
  DRUGS_ADMISSIONS = c('IN_HOME_OXIGEN_THERAPY', 'IN_PREVIOUSASPIRIN', 'IN_OTHERANTIPLATELET',
                       'IN_ORALANTICOAGL',  'IN_ACEI_ARB', 'IN_BETABLOCKERS',
                       'IN_BETAGONISTINHALED', 'IN_GLUCORTICOIDSINHALED',
                       'IN_DVITAMINSUPLEMENT','IN_BENZODIACEPINES', 'IN_ANTIDEPRESSANT')
  DRUGS_ADMISSIONS_new = c('HOME_OXIGEN_THERAPY', 'IN_PREVIOUSASPIRIN', 'IN_OTHERANTIPLATELET',
                           'IN_ORALANTICOAGL',  'IN_ACEI_ARB', 'IN_BETABLOCKERS',
                           'IN_BETAGONISTINHALED', 'IN_GLUCORTICOIDSINHALED',
                           'IN_DVITAMINSUPLEMENT','IN_BENZODIACEPINES', 'IN_ANTIDEPRESSANT')
  setnames(df, old = DRUGS_ADMISSIONS, new = DRUGS_ADMISSIONS_new)
  df[,DRUGS_ADMISSIONS_new] = convert_zero_one(df, DRUGS_ADMISSIONS_new)
  
  COVID_DRUGS = c('IN_USEOFCORTICOIDSDURINGADMISSION',
                  'IN_USEOFCLOROQUINEORSIMILARDURINGADMISSION',
                  'IN_USEOFANTIVIRALDRUGSDURINGADMISSION',
                  'IN_USEOFINTERFERONORSIMILARDURINGADMISSION',
                  'IN_USEOFTOCILIZUMABORSIMILARDURINGADMISSION', 'IN_USEOFANTIBIOTICS',
                  'IN_ACEI_ARBS_DURING_HOSPITAL_STAY', 'IN_ANTICOAGLDURINGADMISSION',
                  'CO_MAIN_ANTICOAGULATION_INHOSPITAL_STAY','DS_RELEVANTDRUGSDURINGADMISSION',
                  'DT_USE_CLOROQUINE2', 'DT_USE_ANTIVIRAL_DRUGS2',
                  'DT_USE_TOCILIZUMAB2')
  COVID_DRUGS_NEW = c('CORTICOSTEROIDS',
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
  setnames(df, old = COVID_DRUGS, new = COVID_DRUGS_NEW)
  df[,COVID_DRUGS_NEW] = convert_zero_one(df, COVID_DRUGS_NEW)
  
  #We will exclude the discharge drugs
  DISCHARGE_DRUGS = c('IN_DISCHARGEANTIPLATELET','IN_DISCHARGEACEI', 'IN_DISCHARGEANTICOAGU',
                      'DS_DISCHARGEMEDS')
  
  ALL_SYMPTOMS = c('IN_ASYMPTOMATIC', 'CO_DISPNEA', 'IN_TAQUIPNEA', 'IN_FATIGUE',
                   'IN_HIPO_ANOSMIA', 'IN_DISGEUSIA', 'IN_SORETHROAT','VL_MAXTEMPDURINGADMISSION', 
                   'IN_COUGH', 'IN_VOMITING', 'IN_DIARRHEA',
                   'IN_MYALGIAORARTHALGIA')
  
  VITALS = c('IN_TAQUIPNEA','VL_MAXTEMPDURINGADMISSION','IN_02SAT_92')
  VITALS_NEW = c('FAST_BREATHING','MAXTEMPERATURE_ADMISSION','SAT02_BELOW92')
  setnames(df, old = VITALS, new = VITALS_NEW)
  
  #Exclude because they exist in continuous format: 'IN_CRELEVADA1.5OMAS',
  #'IN_LEUCOS4000OMENOS','IN_LINFOS1500OMENOS','IN_HB12OMENOS',
  # 'IN_PLAQTS150000OMENOS'
  BINARY_LABS_VITALS_ADMISSION = c('IN_ELEVATEDDDIMER',
                                   'IN_ELEVATEDPROCALCITONIN', 'IN_ELEVATEDPCR', 'IN_ELEVATEDTN',
                                   'IN_ELEVATEDTRANSAMINASES', 'IN_ELEVATED_FERRITINE',
                                   'IN_ELEVATED_TRIGLYCERIDES', 'IN_ELEVATED_LDH',
                                   'IN_BLOOD_PRESSURE_ABNORMAL')
  BINARY_LABS_VITALS_ADMISSION_NEW = c('DDDIMER_B',
                                       'PROCALCITONIN_B', 'PCR_B', 'TN_B',
                                       'TRANSAMINASES_B', 'FERRITINE_B',
                                       'TRIGLYCERIDES_B', 'LDL_B',
                                       'BLOOD_PRESSURE_ABNORMAL_B')
  setnames(df, old = BINARY_LABS_VITALS_ADMISSION, new = BINARY_LABS_VITALS_ADMISSION_NEW)
  df[,BINARY_LABS_VITALS_ADMISSION_NEW] = convert_zero_one(df, BINARY_LABS_VITALS_ADMISSION_NEW)
  
  CONTINUE_LABS_ADMISSION = c('VL_ONSETCREATININELEVELS','VL_ONSETARTERIALBLOODGASPH',
                              'VL_ONSETARTERIALBLOODGASPA02', 'VL_ONSETARTERIALBLOODGASPAC02',
                              'LV_ONSETARTERIALBLOODGAS02SATURATION','VL_ONSET_NA_LEVELS',
                              'VL_TOTAL_ONSET_LEUCOCYTES_COUNT', 'VL_TOTAL_ONSET_LINPHOCYTES_COUNT',
                              'VL_ONSET_HEMOGLOBIN','VL_TOTAL_ONSET_PLATELET_COUNT',
                              'VL_GLASGOW_COMA_SCORE')
  CONTINUE_LABS_ADMISSION_new = c('CREATININE','ARTERIALBLOODGASPH',
                                  'ARTERIALBLOODGASPA02', 'ARTERIALBLOODGASPAC02',
                                  'ARTERIALBLOODGAS02SATURATION','SODIUM',
                                  'LEUCOCYTES', 'LYMPHOCYTES',
                                  'HEMOGLOBIN','PLATELETS',
                                  'GLASGOW_COMA_SCORE')
  setnames(df, old = CONTINUE_LABS_ADMISSION, new = CONTINUE_LABS_ADMISSION_new)
  #Replace 0 with NA
  df[,c('ARTERIALBLOODGASPA02','ARTERIALBLOODGASPAC02','ARTERIALBLOODGAS02SATURATION')][df[,c('ARTERIALBLOODGASPA02','ARTERIALBLOODGASPAC02','ARTERIALBLOODGAS02SATURATION')]==0]=NA
  
  #Exclude other relevant findings because of unstructured nature: 'DS_OTHERRELEVANTFINDINGS'
  XRAY_RESULTS=c('CO_ANYCHESTRXABNORMALITY')
  XRAY_RESULTS_new = c('CHESTXRAY_BNORMALITY')
  setnames(df, old = XRAY_RESULTS, new = XRAY_RESULTS_new)
  
  PROCEDURES = c('IN_02DURINGADMISSION', 'IN_HIGHFLOWNASALCANNULA',
                 'IN_NOINVASIVEMECHANICALVENTILATION',
                 'IN_INVASIVEMECHANICALVENTILATION', 'NU_DAYSONMECHANICALVENTILATION',
                 'IN_PRONEDURINGADMISSION', 'IN_CIRCULATORYORECMOSUPPORT',
                 'IN_ECMO_SIMILAR_SUPPORT')
  PROCEDURES_new = c('X02DURINGADMISSION', 'HIGHFLOWNASALCANNULA',
                     'NONINVASIVEMECHANICALVENTILATION',
                     'INVASIVEMECHANICALVENTILATION', 'DAYS_ON_MECHANICALVENTILATION',
                     'PRONEDURINGADMISSION', 'CIRCULATORYORECMOSUPPORT',
                     'ECMO_SIMILAR_SUPPORT')
  setnames(df, old = PROCEDURES, new = PROCEDURES_new)
  
  #Exclude 'IN_PNEUMONIA', 'DS_COMPLICATIONSDESCRIPTION', 'IN_RAS_CUTANEOUS_INVOLVEMENT','DIASAMUERTEOUCI', # 'IN_MUERTEOUCI'
  OUTCOMES = c('IN_ICUADMISSION','IN_RESPIRATORYINSUFFICIENCYADMISSION',
               'IN_HEARTFAILUREADMISSION', 'IN_RENALFAILURE',
               'IN_UPPERRESPIRAROTYTRACTINFECTIONDATA', 'CO_PNEUMONIA',
               'IN_SEPSIS', 'IN_SYSTEMICINFLAMATORYREPONSESYNDROME',
               'IN_ANYRELEVANTBLEEDING', 'IN_HEMOPTYSIS', 'IN_EMBOLICEVENT',
               'CO_DEATHCAUSE', 'DEATH','CO_DISCHARGETO','FOLLOWUPDAYSKM')
  OUTCOMES_new = c('ICUADMISSION','RESPIRATORY_INSUFFICIENCY',
                   'HEARTFAILURE', 'RENALFAILURE',
                   'UPPER_RESPIRATORY_TRACT_INFECTION', 'PNEUMONIA',
                   'SEPSIS', 'SYSTEMIC_INFLAMATORY_REPONSE_SYNDROME',
                   'ANYRELEVANT_BLEEDING', 'HEMOPTYSIS', 'EMBOLIC_EVENT',
                   'CO_DEATHCAUSE', 'DEATH','DISCHARGE_TO','FOLLOWUPDAYS')
  setnames(df, old = OUTCOMES, new = OUTCOMES_new)
  
  DATES = c('DT_BORN2', 'DT_ONSETSYMPTOMS2',
            'DT_HOSPITAL_ADMISSION2', 'DT_ADMISSION_ICU2',
            'DT_ICUDISCHARGE2', 'DT_DISCHARGE2',
            'DT_LASTFOLLOWUPDATE2','DT_DEATHORICU', 'DT_DEATH2',
            'DT_TEST_COVID2')
  DATES_new = c('DT_BIRTH', 'DT_ONSETSYMPTOMS',
                'DT_HOSPITAL_ADMISSION', 'DT_ADMISSION_ICU',
                'DT_ICUDISCHARGE', 'DT_DISCHARGE',
                'DT_LASTFOLLOWUPDATE','DT_DEATHORICU', 'DT_DEATH',
                'DT_TEST_COVID')
  setnames(df, old = DATES, new = DATES_new)
  cols_include = c(LOCATION_new, DATES_new, DEMOGRAPHICS_new, COMORBIDITIES_new, 
                   DRUGS_ADMISSIONS_new, COVID_DRUGS_NEW, VITALS_NEW,
                   BINARY_LABS_VITALS_ADMISSION_NEW, CONTINUE_LABS_ADMISSION_new,
                   XRAY_RESULTS_new, PROCEDURES_new, OUTCOMES_new)
  data<-df[,cols_include]
  write.csv(data, paste(save_path,"hope_data_clean.csv", sep=""))
  return(data)
}

