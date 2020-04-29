#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 00:18:10 2020

@author: agni
"""

import pandas as pd
#  import datetime
import numpy as np
import pickle

import analyzer.loaders.hmfundacion.utils as u

#path = '../../../Dropbox (Personal)/COVID_clinical/covid19_hmfoundation'

def load_fundacionhm(path, discharge_data = True, comorbidities_data = True, vitals_data = True, lab_tests=True, demographics_data = False, extra_data = False):
        
    # Load admission info
    admission = pd.read_csv('%s/admissions.csv' % path, sep=';' , encoding= 'unicode_escape')
    
    #Create dataset with relevant information and patients for which we know the outcome
    dataset_admissions = u.create_dataset_admissions(admission)
    
     # Filter to relevant patients
    patients = dataset_admissions['PATIENT ID'].unique()
    
    #Demographic information
    dataset_demographics = u.create_dataset_demographics(admission)
    
    #Vital values
    dataset_vitals = u.create_vitals_dataset(admission)
    
    dataset_demographics = dataset_demographics[dataset_demographics['PATIENT ID'].isin(dataset_admissions['PATIENT ID'])]
    dataset_vitals = dataset_vitals[dataset_vitals['PATIENT ID'].isin(dataset_admissions['PATIENT ID'])]

    
    # Lab values
    #Read in the dataset
    labs = pd.read_csv('%s/labs.csv' % path, encoding= 'unicode_escape')
    #Get the values
    dataset_labs = u.create_lab_dataset(labs, dataset_admissions, dataset_vitals)

    if lab_tests:
        dataset_vitals.drop('SaO2', axis=1)  # Remove oxygen saturation if we have lab values (it is there)


    #Comorbidities
    
    # Read all comorbidities from both the emergency and inpatient files
    comorbidities_emerg = pd.read_csv('%s/comorbidities_emerg.csv' % path, sep=';',encoding= 'unicode_escape')
    comorbidities_inpatient = pd.read_csv('%s/comorbidities_inpatient.csv' % path, sep=';', encoding= 'unicode_escape')

    comorb_long = u.prepare_dataset_comorbidities(comorbidities_emerg, comorbidities_inpatient)
    
    #Input category of icd codes
    icd_category = 10
    dataset_comorbidities = u.create_dataset_comorbidities(comorb_long, icd_category, dataset_admissions)
       
    #These are the columns for which we do not have values
    dataset_extra = u.add_extra_features(dataset_admissions)
    
    # Filter patients common to all datasets
    patients = u.filter_patients([dataset_admissions, dataset_demographics,dataset_vitals,
                                  dataset_labs, dataset_comorbidities, dataset_extra])
    
    dataset_admissions.drop(['Date_Admission', 'Date_Emergency'], axis=1, inplace=True)
    
    
    name_datasets = np.asarray(['discharge', 'comorbidities', 'vitals', 'lab', 'demographics', 'extras'])
    dataset_array = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, extra_data])

    # Set index
    dataset_admissions.set_index('PATIENT ID', inplace=True)
    dataset_comorbidities.set_index('PATIENT ID', inplace=True)
    dataset_vitals.set_index('PATIENT ID', inplace=True)
    dataset_labs.set_index('PATIENT ID', inplace=True)
    dataset_demographics.set_index('PATIENT ID', inplace=True)
    dataset_extra.set_index('PATIENT ID', inplace=True)

    datasets = []

    # Create final dataset
    if discharge_data:
        datasets.append(dataset_admissions)
    
    if comorbidities_data:
        datasets.append(dataset_comorbidities)
    
    if vitals_data:
        datasets.append(dataset_vitals)
    
    if lab_tests:
        datasets.append(dataset_labs)

    if demographics_data:
        datasets.append(dataset_demographics)

    if extra_data:
        datasets.append(dataset_extra)

    datasets = np.asarray(datasets)

    data = dict(zip(name_datasets[dataset_array], datasets))

    return data