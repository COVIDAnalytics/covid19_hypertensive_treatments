#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:12:02 2020

@author: agni
"""
import pyreadstat


pathfull = '../../../Dropbox (Personal)/COVID_clinical/covid19_hope/IVAN J NUNEZ GIL - HOPEDATABASE 5.5.20V3.8AGNI.sav'

path  = '../../../Dropbox (Personal)/COVID_clinical/covid19_hope'

df, meta = pyreadstat.read_sav(pathfull)
df.to_csv(path+'/hope_data.csv')
