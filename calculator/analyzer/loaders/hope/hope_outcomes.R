library(tidyverse)
library(foreign)

# df <- read.csv("~/Dropbox (MIT)/COVID_risk/covid19_hope/hope_data_clean.csv", 
#                stringsAsFactors = FALSE, encoding="UTF-8")

df <- read.spss("~/Dropbox (MIT)/COVID_risk/covid19_hope/HOPEDATABASE 5.5.20V3.9AGNI.sav", to.data.frame=TRUE)
n = dim(df)[1]

df_clean <- df %>% mutate(COUNTRY = trimws(INV_COUNTRY1), 
              HOSPITAL = trimws(INV_HOSPITAL1),
              CHLOROQINE = coalesce(IN_USEOFCLOROQUINEORSIMILARDURINGADMISSION,"NO"), 
              ANTIVIRAL = coalesce(IN_USEOFANTIVIRALDRUGSDURINGADMISSION,"NO"), 
              CORTICOID = coalesce(IN_USEOFCORTICOIDSDURINGADMISSION,"NO"), 
              ANTICOAG = if_else(coalesce(IN_ANTICOAGLDURINGADMISSION, "NO") == "NO", "NO", "YES")) %>%
  mutate_at(vars(c("CHLOROQINE", "ANTIVIRAL", "ANTICOAG","CORTICOID", "DEATH", 
                   "IN_SEPSIS","IN_RENALFAILURE","IN_HEARTFAILUREADMISSION")), 
            function(x){if_else(x=="NO",0,1)}) %>%
  mutate(COMPLICATION = if_else(pmax(IN_SEPSIS,IN_RENALFAILURE,IN_HEARTFAILUREADMISSION,DEATH, na.rm = TRUE) > 0, 1, 0)) %>%
  select(COUNTRY, HOSPITAL, CHLOROQINE, ANTIVIRAL, ANTICOAG, CORTICOID, DEATH, COMPLICATION)

df_clean %>% group_by(COUNTRY) %>%
  summarize(Hospital_Count = n_distinct(HOSPITAL),
            Patient_Count = n(),
            # CHLOROQINE_PCT = mean(CHLOROQINE),
            ANTIVIRAL_PCT = mean(ANTIVIRAL),
            ANTICOAG_PCT = mean(ANTICOAG),
            CORTICOID_PCT = mean(CORTICOID),
            DEATH_PCT = mean(DEATH),
            COMPLICATION_PCT = mean(COMPLICATION))%>%
  arrange(desc(Patient_Count))

df_clean %>% 
  # filter(COUNTRY %in% c("Italy", "ECUADOR", "Germany")) %>%
  group_by(CHLOROQINE, ANTIVIRAL, ANTICOAG) %>%
  summarize(Patient_Count = n(),
            DEATH_PCT = mean(DEATH),
            COMPLICATION_PCT = mean(COMPLICATION))
  

# Explore anticoagulants --------------------------------------------------

table(df$CO_MAIN_ANTICOAGULATION_INHOSPITAL_STAY)
table(df$CO_ORALANTICOAGL_TYPE_OAC)
table(df$CO_ORALANTICOAGL_TYPE_OAC)
df %>% select(contains('ANTICOAG')) %>%
  summary()

table(coalesce(df$IN_ANTICOAGLDURINGADMISSION,"NO"))

# Parenteral: e.g. Heparin
# AVK: e.g. Warfarin
# DOAC?
# Prophylactic
               
# Oxygen Notes ------------------------------------------------------------
names(df)
PROCEDURES = c('IN_02DURINGADMISSION', 'IN_HIGHFLOWNASALCANNULA',
               'IN_NOINVASIVEMECHANICALVENTILATION',
               'IN_INVASIVEMECHANICALVENTILATION', 
               # 'NU_DAYSONMECHANICALVENTILATION',
               'IN_PRONEDURINGADMISSION', 'IN_CIRCULATORYORECMOSUPPORT',
               'IN_ECMO_SIMILAR_SUPPORT')

t <- df %>% select(PROCEDURES) %>% summary() %>% as.data.frame() %>%
  separate(Freq, sep = ":", into = c("Value","Frequency")) %>%
  mutate_all(trimws) %>%
  spread(key = "Value", value = "Frequency") %>%
  mutate(Proportion = as.numeric(YES)/n)

sum(df[df$IN_ECMO_SIMILAR_SUPPORT == "YES","DEATH"] == "YES", na.rm = TRUE)
