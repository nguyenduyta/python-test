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

orders_original_df = pd.read_csv('orders.csv')
users_original_df = pd.read_csv('users.csv')
users_original_df['deleted_at'] = pd.to_datetime(users_original_df['deleted_at'])
users_original_df['interpolate_age'] = users_original_df['age'].interpolate(method='linear')
orders_groupby_user_df = orders_original_df.groupby('user_id')['id'].count().to_frame().reset_index()
orders_groupby_user_sum_df = orders_original_df.groupby('user_id')['price'].sum().to_frame().reset_index()
orders_groupby_user_df = pd.merge(orders_groupby_user_df, orders_groupby_user_sum_df, how="left", on="user_id")
orders_groupby_user_df.columns = ['id', 'order_count', 'ltv']

def deleted_class(deleted_at):
  if pd.isnull(deleted_at):
    return 0
  else:
    return 1

def gender_class(gender):
    if  gender == '未入力':
        return 0
    elif gender == '男':
        return 1
    elif gender == '女':
        return 2
    elif pd.isnull(gender):
        return 1
    else:
        return 99

def age_class(age):
  if age < 10:
    return 0
  elif age < 20:
    return 10
  elif age < 30:
    return 20
  elif age < 40:
    return 30
  elif age < 50:
    return 40
  elif age < 60:
    return 50
  elif age < 70:
    return 60
  elif age < 80:
    return 70
  elif age < 90:
    return 80
  elif age < 100:
    return 90
  elif pd.isnull(age):
      return np.nan
  else:
    return 100

def pref_class(pref):
  if pref == '北海道':
    return 1
  elif pref == '青森県':
    return 2
  elif pref == '岩手県':
    return 3
  elif pref == '宮城県':
    return 4
  elif pref == '秋田県':
    return 5
  elif pref == '山形県':
    return 6
  elif pref == '福島県':
    return 7
  elif pref == '茨城県':
    return 8
  elif pref == '栃木県':
    return 9
  elif pref == '群馬県':
    return 10
  elif pref == '埼玉県':
    return 11
  elif pref == '千葉県':
    return 12
  elif pref == '東京都':
    return 13
  elif pref == '神奈川県':
    return 14
  elif pref == '新潟県':
    return 15
  elif pref == '富山県':
    return 16
  elif pref == '石川県':
    return 17
  elif pref == '福井県':
    return 18
  elif pref == '山梨県':
    return 19
  elif pref == '長野県':
    return 20
  elif pref == '岐阜県':
    return 21
  elif pref == '静岡県':
    return 22
  elif pref == '愛知県':
    return 23
  elif pref == '三重県':
    return 24
  elif pref == '滋賀県':
    return 25
  elif pref == '京都府':
    return 26
  elif pref == '大阪府':
    return 27
  elif pref == '兵庫県':
    return 28
  elif pref == '奈良県':
    return 29
  elif pref == '和歌山県':
    return 30
  elif pref == '鳥取県':
    return 31
  elif pref == '島根県':
    return 32
  elif pref == '岡山県':
    return 33
  elif pref == '広島県':
    return 34
  elif pref == '山口県':
    return 35
  elif pref == '徳島県':
    return 36
  elif pref == '香川県':
    return 37
  elif pref == '愛媛県':
    return 38
  elif pref == '高知県':
    return 39
  elif pref == '福岡県':
    return 40
  elif pref == '佐賀県':
    return 41
  elif pref == '長崎県':
    return 42
  elif pref == '熊本県':
    return 43
  elif pref == '大分県':
    return 44
  elif pref == '宮崎県':
    return 45
  elif pref == '鹿児島県':
    return 46
  elif pref == '沖縄県':
    return 47
  elif pd.isnull(pref):
      return np.nan
  else:
    return 99

def marriage_class(marriage):
    if  marriage == '未婚':
        return 0
    elif marriage == '既婚':
        return 1
    elif pd.isnull(marriage):
        return np.nan
    else:
        return 99

def communication_carrier_class(communication_carrier):
    if  communication_carrier == 'ドコモ':
        return 1
    elif communication_carrier == 'ソフトバンク':
        return 2
    elif communication_carrier == 'au':
        return 3
    elif communication_carrier == 'ツーカー':
        return 4
    elif pd.isnull(communication_carrier):
        return np.nan
    else:
        return 99

users_original_df['deleted'] = users_original_df['deleted_at'].apply(deleted_class).to_frame()
users_original_df['byage'] = users_original_df['interpolate_age'].apply(age_class).to_frame()
users_original_df['gender_id'] = users_original_df['gender'].apply(gender_class).to_frame()
users_original_df['pref_id'] = users_original_df['pref'].apply(pref_class).to_frame()
users_original_df['marriage_id'] = users_original_df['marriage'].apply(marriage_class).to_frame()
users_original_df['communication_carrier_id'] = users_original_df['communication_carrier'].apply(communication_carrier_class).to_frame()

users_with_ltv_df =  pd.merge(users_original_df, orders_groupby_user_df, how='left', on="id")
users_with_ltv_df[['ltv']] = users_with_ltv_df[['ltv']].fillna(0)
users_with_ltv_df[['order_count']] = users_with_ltv_df[['order_count']].fillna(0)

target_columns = ['byage', 'gender_id', 'pref_id', 'marriage_id', 'communication_carrier_id', 'order_count', 'ltv']
target_label = 'deleted'

y = users_with_ltv_df[target_label]
len(y)

X = users_with_ltv_df[target_columns]
len(X)

W = pd.merge(X, pd.DataFrame(y), right_index=True, left_index=True)
seaborn.pairplot(W, hue = target_label)
plt.savefig('pairplot_noscalled.png')
plt.clf()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Z = pd.merge(pd.DataFrame(X_scaled), pd.DataFrame(y), right_index=True, left_index=True)

target_columns.append(target_label)
Z.columns = target_columns
plt.figure(figsize=(50, 50))
seaborn.pairplot(Z, hue = target_label)
plt.savefig('pairplot_scalled.png')
plt.clf()
