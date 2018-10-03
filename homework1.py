# coding:utf-8
from __future__ import print_function, division

import os.path
import sys
import pandas as pd
import seaborn
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

users_original_df = pd.read_csv('./users.csv')
orders_original_df = pd.read_csv('./orders.csv')

users_original_df["interpolate_age"] = users_original_df["age"].interpolate(method='linear')

orders_count_by_user = orders_original_df.groupby('user_id').size().reset_index(name='counts')
orders_sum_price_by_user = orders_original_df.groupby('user_id')['price'].sum().to_frame().reset_index()
users = pd.merge(orders_count_by_user, orders_sum_price_by_user, how="left", on="user_id")
users.columns = ['id', 'order_count', 'ltv']

users_orders_count_ltv =  pd.merge(users_original_df, users, how='left', on="id")

users_gender_age_df = users_orders_count_ltv.loc[:,[
      'gender',
      'interpolate_age',
      'communication_carrier',
      'marriage',
      'pref',
      'order_count',
      'ltv'
]]
users_gender_age_df['ltv'] = users_gender_age_df['ltv'].fillna(0)
users_gender_age_df['order_count'] = users_gender_age_df['order_count'].fillna(0)

def age_class(age):
  if age < 10:
    return '1'
  elif age < 20:
    return '2'
  elif age < 30:
    return '3'
  elif age < 40:
    return '4'
  elif age < 50:
    return '5'
  elif age < 60:
    return '6'
  elif age < 70:
    return '7'
  elif age < 80:
    return '8'
  elif age < 90:
    return '9'
  elif age < 100:
    return '10'
  else:
    return '11'


users_gender_age_df['byage'] = users_gender_age_df['interpolate_age'].apply(age_class).to_frame()
communication_carrier_df = users_gender_age_df.groupby('communication_carrier').size().reset_index()
communication_carrier_df = communication_carrier_df.assign(cc_id=range(1, len(communication_carrier_df) + 1)).loc[:, ['cc_id', 'communication_carrier']]
users_gender_age_df = pd.merge(users_gender_age_df, communication_carrier_df, how="left", on="communication_carrier")

gender_df = users_gender_age_df.groupby('gender').size().reset_index()
gender_df = gender_df.assign(gender_id=range(1, len(gender_df) + 1)).loc[:, ['gender_id', 'gender']]
users_gender_age_df = pd.merge(users_gender_age_df, gender_df, how="left", on="gender")

pref_df = users_gender_age_df.groupby('pref').size().reset_index()
pref_df = pref_df.assign(pref_id=range(1, len(pref_df) + 1)).loc[:, ['pref_id', 'pref']]
users_gender_age_df = pd.merge(users_gender_age_df, pref_df, how="left", on="pref")

marriage_df = users_gender_age_df.groupby('marriage').size().reset_index()
marriage_df = marriage_df.assign(marriage_id=range(1, len(marriage_df) + 1)).loc[:, ['marriage_id', 'marriage']]
users_gender_age_df = pd.merge(users_gender_age_df, marriage_df, how="left", on="marriage")
users_gender_age_df = users_gender_age_df.fillna(0)

Y  = pd.Series(users_gender_age_df.byage)
X = users_gender_age_df.drop(["gender","interpolate_age","communication_carrier","marriage","pref"], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# pairplot
plt.figure(figsize=(50, 50))
seaborn.pairplot(pd.DataFrame(X_train_scaled).corr())
plt.savefig('pairplot.png')
plt.clf()
