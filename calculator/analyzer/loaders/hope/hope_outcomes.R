library(tidyverse)
library(foreign)

df <- read.csv("~/Dropbox (MIT)/COVID_risk/covid19_hope/hope_data.csv", 
               stringsAsFactors = FALSE, encoding="UTF-8")

df <- read.spss("~/Dropbox (MIT)/COVID_risk/covid19_hope/HOPEDATABASE 5.5.20V3.9AGNI.sav", to.data.frame=TRUE)

df_clean <- df %>% mutate(COUNTRY = trimws(INV_COUNTRY1), 
              HOSPITAL = trimws(INV_HOSPITAL1),
              CHLOROQINE = coalesce(IN_USEOFCLOROQUINEORSIMILARDURINGADMISSION,"NO"), 
              ANTIVIRAL = coalesce(IN_USEOFANTIVIRALDRUGSDURINGADMISSION,"NO"), 
              ANTICOAG = if_else(coalesce(IN_ANTICOAGLDURINGADMISSION, "NO") == "NO", "NO", "YES")) %>%
  mutate_at(vars(c("CHLOROQINE", "ANTIVIRAL", "ANTICOAG", "DEATH")), function(x){if_else(x=="NO",0,1)}) %>%
  select(COUNTRY, HOSPITAL, CHLOROQINE, ANTIVIRAL, ANTICOAG, DEATH)

df_clean %>% group_by(COUNTRY) %>%
  summarize(Hospital_Count = n_distinct(HOSPITAL),
            Patient_Count = n(),
            CHLOROQINE_PCT = mean(CHLOROQINE),
            ANTIVIRAL_PCT = mean(ANTIVIRAL),
            ANTICOAG_PCT = mean(ANTICOAG),
            DEATH_PCT = mean(DEATH))%>%
  arrange(desc(Patient_Count))

df_clean %>% 
  filter(COUNTRY %in% c("Italy", "ECUADOR", "Germany")) %>%
  group_by(CHLOROQINE, ANTIVIRAL, ANTICOAG) %>%
  summarize(Patient_Count = n(),
            DEATH_PCT = mean(DEATH))
  
               