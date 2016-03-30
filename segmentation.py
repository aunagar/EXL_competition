#Importing Libraries
import pandas as pd
import openpyxl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

#getting data
t_original_data = pd.read_excel("C:/Users/ajayunagar/Desktop/EXL_EQ2016_Insurance_Case_Question_Set.xlsx",sheetname = "TD2")

#cleaning data

t_data = t_original_data.drop(['ORIGINAL_QUOTE_DATE','IND_QUOTE_CONVERSION','QUOTE_NUM', 'STATE', 'AGENCY_NAME',
                               'INSURANCE_SCORE', 'CONSTRUCTION_TYPE','IND_BURGLAR_ALARM',
                               'IND_SPRINKLER_SYSTEM', 'PREMIUM_AMOUNT'], axis=1)


#Standardizing data
max_floors = t_data["NUM_FLOORS"].max()
t_data["NUM_FLOORS"] = t_data["NUM_FLOORS"]/max_floors

#2. num of appliances
max_app = t_data["NUM_APPLIANCES"].max()
t_data["NUM_APPLIANCES"] = t_data["NUM_APPLIANCES"]/max_app

#4. Property Age
max_property_age = t_data["PROPERTY_AGE"].max()
t_data["PROPERTY_AGE"] = t_data["PROPERTY_AGE"]/max_property_age

#6. Family members
max_FM = t_data["NUM_FAMILY_MEM"].max()
t_data["NUM_FAMILY_MEM"] = t_data["NUM_FAMILY_MEM"]/max_FM

# Heating Pipe System Age
max_hsys_age = t_data["HEATING_PIPING_SYS_AGE"].max()
t_data["HEATING_PIPING_SYS_AGE"] = t_data["HEATING_PIPING_SYS_AGE"]/max_hsys_age

#safety devices
max_safe_dev = t_data["NUM_SAFETY_DEVICES"].max()
t_data["NUM_SAFETY_DEVICES"] = t_data["NUM_SAFETY_DEVICES"]/max_safe_dev
# 5. Coverage Bin
max_cov_bin = t_data["COVERAGE_BIN"].max()
t_data["COVERAGE_BIN"] = t_data["COVERAGE_BIN"]/max_cov_bin

#cat zone
max_cat_zone = t_data["CAT_ZONE"].max()
t_data["CAT_ZONE"] = t_data["CAT_ZONE"]/max_cat_zone

t_data['is_t_data'] = np.random.uniform(0,1,len(t_data))<=0.35
t_data_again = t_data[t_data['is_t_data']==True]
t_data_array = np.array(t_data_again)
customer_clust = KMeans(7,n_jobs=50,n_init = 20,max_iter = 500).fit_predict(t_data_array)
silh = silhouette_score(t_data_array,customer_clust)




K = range(1,10)
KM = [kmeans(t_data_array,k) for k in K]
centroids = [cent for (cent,var) in KM]
avgWithinSS = [var for (cent,var) in KM]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K,avgWithinSS,'b*-')
plt.grid(True)
plt.xlabel('Number of Clusters')
plt.ylabel('Avg within_clusters sum of squares')
plt.title('Elbow for KMeans clustering')
plt.show()
