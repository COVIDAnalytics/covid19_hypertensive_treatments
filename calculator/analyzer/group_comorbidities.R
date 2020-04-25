#SET THE WORKING DIRECTORY TO SOURCE FILE LOCATION (LOOK UP TO THE "SESSION" TAB ABOVE)

library(dplyr)
c = read.csv("nosologico_com.csv")
c = c %>% select('id', 'comorb')
coms = comorbid_ccs(com, icd_name = comorb, preclean = TRUE, return_df = TRUE)
coms = coms %>% select(-"0")
write.csv(coms, "comorbidities.csv")
