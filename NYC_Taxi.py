# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 00:45:30 2015

@author: venki_k07
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Reading data from Excel files as dataframes
trip_data = pd.read_csv("C:/DI_files/trip_data/trip_data_3.csv")
trip_fare = pd.read_csv("C:/DI_files/trip_fare/trip_fare_3.csv")

#reindexing
trip_data.columns = ['medallion', 'hack_license', 'vendor_id', 'rate_code', 'store_and_fwd_flag',\
                     'pickup_datetime', 'dropoff_datetime', 'passenger_count', 'trip_time_in_secs',\
                     'trip_distance', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
                     
trip_fare.columns = ['medallion', 'hack_license', 'vendor_id', 'pickup_datetime', 'payment_type', \
                     'fare_amount', 'surcharge', 'mta_tax', 'tip_amount', 'tolls_amount', 'total_amount']
                     
#retaining only required variables                    
trip_data = trip_data[['medallion', 'hack_license', 'pickup_datetime', 'dropoff_datetime', 'trip_time_in_secs',\
                       'trip_distance', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]
trip_fare = trip_fare[['medallion', 'hack_license', 'pickup_datetime', 'payment_type', 'fare_amount',\
                       'tip_amount', 'total_amount']]
#part 1 
trip_fare_amt_5_CRD = trip_fare.loc[(trip_fare.total_amount <= 5) & (trip_fare.payment_type == 'CRD'), 'total_amount']
trip_fare_amt_5 = trip_fare.loc[(trip_fare.total_amount <= 5), 'total_amount']
print('Average no. of. credit card transaction for < $5: ', len(trip_fare_amt_5_CRD)/len(trip_fare_amt_5))

#part 2
trip_fare_amt_50_CRD = trip_fare.loc[(trip_fare.total_amount > 50) & (trip_fare.payment_type == 'CRD'), 'total_amount']

#plots to identify the outliers
plt.figure()
plt.boxplot(pd.Series(trip_fare_amt_50_CRD))
plt.ylim(0, 200)
plt.show()

plt.figure()
plt.hist(pd.Series(trip_fare_amt_50_CRD), bins = 40)
plt.show()

#As there are extremely high values for the fare, we can consider the values within 5 - 95 percentiles
fare_5 = np.percentile(pd.Series(trip_fare_amt_50_CRD), 5)
fare_95 = np.percentile(pd.Series(trip_fare_amt_50_CRD), 95)

trip_fare_amt_50 = trip_fare.loc[(trip_fare.total_amount > fare_5) & (trip_fare.total_amount < fare_95), 'total_amount']
trip_fare_amt_50_CRD = trip_fare_amt_50_CRD[(trip_fare_amt_50_CRD > fare_5) & (trip_fare_amt_50_CRD < fare_95)]
print('Average no. of. credit card transaction for > $50: ', len(trip_fare_amt_50_CRD)/len(trip_fare_amt_50))

#part 3
trip_fare_amt = trip_fare.fare_amount.loc[trip_data.trip_time_in_secs > 60] 
trip_data_time = trip_data.trip_time_in_secs.loc[trip_data.trip_time_in_secs > 60]
fare_per_min = pd.Series(fare_per_min = pd.Series(trip_fare_amt/(trip_data_time/60)))

plt.figure()
plt.hist(fare_per_min[fare_per_min <4], bins = 40)
plt.ylim(0, 2100000)
plt.show() 

plt.figure()
plt.boxplot(fare_per_min[fare_per_min < 4])
plt.show()

Q1 = np.percentile(fare_per_min, 5)
Q2 = np.percentile(fare_per_min, 95)
fare_per_min = fare_per_min[(fare_per_min > Q1) & (fare_per_min < Q2)]
print('Mean fare per minute driven:', fare_per_min.mean())

#part 4
fare_per_mile = pd.Series(trip_fare.fare_amount/trip_data.trip_distance)
Q1_mile = np.percentile(fare_per_mile, 5)
Q2_mile = np.percentile(fare_per_mile, 95)
fare_per_mile = fare_per_mile[(fare_per_mile > Q1_mile) & (fare_per_mile < Q2_mile)]
print('Median fare per mile driven:', fare_per_mile.median())

#part 5
Q1_dist = np.percentile(trip_data.trip_distance, 5)
Q2_dist = np.percentile(trip_data.trip_distance, 95)
trip_dist_IQR = trip_data.trip_distance[(trip_data.trip_distance > Q1_dist) & (trip_data.trip_distance < Q2_dist)]
trip_time_limit = trip_data.trip_time_in_secs.loc[trip_data.trip_time_in_secs > 60]
miles_hr = pd.Series(trip_dist_IQR /(trip_time_limit/3600))
print('95 percentile of mean miles/hr:', np.percentile(miles_hr.dropna(), 95))

#part 6
trip_data['dlat'] = np.radians(trip_data['pickup_latitude']- trip_data['dropoff_latitude'])
trip_data['dlon'] = np.radians(trip_data['pickup_longitude'] - trip_data['dropoff_longitude'])

trip_data['a'] = np.sin(trip_data['dlat']/2)**2 +\
                 np.cos(np.radians(trip_data['pickup_latitude'])) * np.cos(np.radians(trip_data['dropoff_latitude']))*\
                 np.sin(trip_data['dlon']/2)**2
                 
trip_data['c'] = 2*np.arctan2(np.sqrt(trip_data['a']), np.sqrt(1 - trip_data['a']))
trip_data['dist'] = 3961 * trip_data['c']   #3961 miles is the radius of the Earth

ratio_dist_driven = trip_data['dist']/trip_data['trip_distance'] # always less than 1
ratio_dist_driven = ratio_dist_driven[ratio_dist_driven < 1]
#As there are extremely high values for the distance traveled ratio, we can consider the values within 5 - 95 percentiles
dist_5 = np.percentile(pd.Series(ratio_dist_driven), 5)
dist_95 = np.percentile(pd.Series(ratio_dist_driven), 95)

mean_dist_driven = ratio_dist_driven[(ratio_dist_driven > dist_5) & (ratio_dist_driven < dist_95)].mean()
print('Average ratio of the distance and the distance driven:', mean_dist_driven)

#Part 7 - JFK airport
#The latitude and longitude range near the JFK airport is 40.6448 - 40.65 and -73.7858 and -73.7927

trip_data_JFK = trip_data.index[((trip_data.pickup_latitude > 40.6448) & (trip_data.pickup_latitude < 40.65)) &\
                                ((trip_data.pickup_longitude > -73.7927) & (trip_data.pickup_longitude < -73.7858))] 
tip_JFK = trip_fare.tip_amount[trip_data_JFK]
Q1_tip = np.percentile(tip_JFK, 5)
Q2_tip = np.percentile(tip_JFK, 95)

avg_tip_JFK = tip_JFK[(tip_JFK > Q1_tip) & (tip_JFK < Q2_tip)].mean()

#Part 8
median_rev_driver = trip_fare.groupby('hack_license')['total_amount'].median()
median_rev_driver = median_rev_driver.mean()


"""
plt.figure()
plt.hist(trip_cost_per_min[trip_cost_per_min <4], bins = 40)
plt.ylim(0, 2100000)
plt.show() 

plt.figure()
plt.boxplot(pd.Series(trip_fare_amount_50['total_amount']))
plt.ylim(0, 200)
plt.show()

plt.figure()
plt.plot(np.ravel(trip_fare_amount_5[['total_amount']]))
plt.ylim(0, 5.5)
plt.show()

np.percentile(trip_cost_per_min[trip_cost_per_min <4], 3) #To remove the outliers
# 2 percentile - 0.73
# 5 percentile - 0.79167
# 98 percentile - 2.927
# 95 percentile - 2.25

np.percentile(pd.Series(trip_fare_amount_50['total_amount']), 10) #To remove the outliers
# 1 percentile - 50.33
# 99 percentile - 130.75

plt.figure()
plt.boxplot(trip_cost_per_min[trip_cost_per_min <3])
plt.show()

trip_cost_per_min [(trip_cost_per_min > 0.73) & (trip_cost_per_min < 2.927)].mean() 

#Identifying columns with missing data
missing_columns = []
for type in trip_data.columns.values:
    length_missing_data = len(trip_data.loc[trip_data.loc[:,type].isnull(), type])
    if length_missing_data > 0:
        missing_columns.append(type)
if len(missing_columns)>0:
    print('Columns that needs imputation in the trip data:', missing_columns)
else:
    print('No missing values in the trip data')
#Fare data
missing_columns = []
for type in trip_fare.columns.values:
    length_missing_data = len(trip_fare.loc[trip_fare.loc[:,type].isnull(), type])
    if length_missing_data > 0:
        missing_columns.append(type)
if len(missing_columns)>0:
    print('Columns that needs imputation in the fare data:', missing_columns)
else:
    print('No missing values in the fare data')
"""
#trip_data = trip_data[['passenger_count', 'trip_time_in_secs', 'trip_distance']]