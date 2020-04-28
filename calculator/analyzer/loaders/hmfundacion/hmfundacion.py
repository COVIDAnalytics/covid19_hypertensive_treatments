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

def load_fundacionhm(path, lab_tests=True):
        
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
    dataset_labs = u.create_lab_dataset(labs, dataset_admissions)
    
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

    data = {'admissions': dataset_admissions,
            'demographics': dataset_demographics,
            'comorbidities': dataset_comorbidities,
            'vitals': dataset_vitals,
            'lab': dataset_labs,
            'extra':dataset_extra}

    return data