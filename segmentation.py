import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn import preprocessing

t_original_data = pd.read_excel("C:/Users/ajayunagar/Desktop/EXL/IDLE/EXL_EQ2016_Insurance_Case_Question_Set.xlsx",sheetname = 'TD1')
t_data = t_original_data.drop(['QUOTE_NUM','ORIGINAL_QUOTE_DATE','IND_BURGLAR_ALARM',
                               'IND_SPRINKLER_SYSTEM','PREMIUM_AMOUNT'],axis = 1)

def convert(data):
    number = preprocessing.LabelEncoder()
    data['IND_DEADBOLT_LOCKS'] = number.fit_transform(data.IND_DEADBOLT_LOCKS)
    data['NUM_FLOORS'] = number.fit_transform(data.NUM_FLOORS)
    data['NUM_APPLIANCES'] = number.fit_transform(data.NUM_APPLIANCES)
    data['INSURANCE_SCORE'] = number.fit_transform(data.INSURANCE_SCORE)
    data['STATE'] = number.fit_transform(data.STATE)
    data['PROPERTY_AGE'] = number.fit_transform(data.STATE)
    data['AGENCY_NAME'] = number.fit_transform(data.AGENCY_NAME)
    data['CONSTRUCTION_TYPE'] = number.fit_transform(data.CONSTRUCTION_TYPE)
    data['COVERAGE_BIN'] = number.fit_transform(data.COVERAGE_BIN)
    data['IND_PRIOR_CLAIM'] = number.fit_transform(data.IND_PRIOR_CLAIM)
    data['IND_SWIMMING_POOL'] = number.fit_transform(data.IND_SWIMMING_POOL)
    data['IND_UNDER_CONSTRUCTION'] = number.fit_transform(data.IND_UNDER_CONSTRUCTION)
    data['IND_FIRE_PROTECT_SYSTEM'] =number.fit_transform(data.IND_FIRE_PROTECT_SYSTEM)
    data['NUM_FAMILY_MEM'] = number.fit_transform(data.NUM_FAMILY_MEM)
    data['HEATING_PIPING_SYS_AGE'] = number.fit_transform(data.HEATING_PIPING_SYS_AGE)
    data['CAT_ZONE'] = number.fit_transform(data.CAT_ZONE)
    data['IND_ORIGINAL_ROOF'] = number.fit_transform(data.IND_ORIGINAL_ROOF)
    data['NUM_SAFETY_DEVICES'] = number.fit_transform(data.NUM_SAFETY_DEVICES)
    data['IND_SMOKE_ALARM'] = number.fit_transform(data.IND_SMOKE_ALARM)
    data['IND_SCHEDULED_PROPERTY'] = number.fit_transform(data.IND_SCHEDULED_PROPERTY)
    data['IND_PETS'] = number.fit_transform(data.IND_PETS)
    data['IND_COASTAL_AREA'] = number.fit_transform(data.IND_COASTAL_AREA)
    return data

t_data = convert(t_data)

t_data_array = np.array(t_data.head(10000))
customer_clust = KMeans(4,n_init = 20,max_iter=500).fit_predict(t_data_array)
s4 = silhouette_score(t_data_array,customer_clust)

