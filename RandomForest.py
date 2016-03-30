import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

t_original_data = pd.read_excel("C:/Users/ajayunagar/Desktop/EXL/IDLE/EXL_EQ2016_Insurance_Case_Question_Set.xlsx",sheetname='TD1')
t_data = t_original_data.drop(['QUOTE_NUM','ORIGINAL_QUOTE_DATE','IND_BURGLAR_ALARM',
                               'IND_SPRINKLER_SYSTEM','CLUSTER'],axis=1)
def convert(data):
    number = preprocessing.LabelEncoder()
    data['IND_DEADBOLT_LOCKS'] = number.fit_transform(data.IND_DEADBOLT_LOCKS)
    data['NUM_FLOORS'] = number.fit_transform(data.NUM_FLOORS)
    data['NUM_APPLIANCES'] = number.fit_transform(data.NUM_APPLIANCES)
    data['INSURANCE_SCORE'] = number.fit_transform(data.INSURANCE_SCORE)
    data['STATE'] = number.fit_transform(data.STATE)
    data['PROPERTY_AGE'] = number.fit_transform(data.PROPERTY_AGE)
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
    data['PREMIUM_AMOUNT'] = number.fit_transform(data.PREMIUM_AMOUNT)
    avg = data['PREMIUM_AMOUNT'].mean()
    std = data['PREMIUM_AMOUNT'].std()
    data['PREMIUM_AMOUNT'] = (data['PREMIUM_AMOUNT']-avg) /std
    return data

t_data = convert(t_data)
t_data['is_t_data'] = np.random.uniform(0,1,len(t_data))<=0.75
train, validate = t_data[t_data['is_t_data']==True], t_data[t_data['is_t_data']==False]
 
y_train = train['IND_QUOTE_CONVERSION']
x_train = train.drop(['IND_QUOTE_CONVERSION'],axis=1)
y_validate = validate['IND_QUOTE_CONVERSION']
x_validate = validate.drop(['IND_QUOTE_CONVERSION'],axis=1)

lg = LogisticRegression()
fit_data = lg.fit(x_train,y_train)
Disbursed_lg = lg.predict_proba(x_validate)
fpr,tpr,_ = roc_curve(y_validate,Disbursed_lg[:,1])
roc_auc = auc(fpr,tpr)
print roc_auc

rf = RandomForestClassifier(n_estimators = 100, n_jobs = 5)
rf.fit(x_train,y_train)
disbursed_rf = rf.predict_proba(x_validate)
fpr, tpr,_ = roc_curve(y_validate,disbursed_rf[:,1])
roc_auc = auc(fpr,tpr)
print roc_auc


