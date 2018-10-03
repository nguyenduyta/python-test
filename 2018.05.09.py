# coding:utf-8
from __future__ import print_function, division

import os.path
import sys
import pandas as pd
import seaborn
import numpy as np
import matplotlib as mpl
mpl.use('agg')

users = pd.read_csv('./users.csv')
orders = pd.read_csv('./orders.csv', index_col='ordered_at', parse_dates=True)
orders_add_weekday_df = orders.set_index([orders.index.weekday, orders.index])

orders_add_weekday_df.index.names = ['weekday', 'date']
users.columns
users.describe(include = 'all')
list(users.columns)
# loc
# iloc
users.describe(include='all')
users_dropna_df = users.dropna(subset=['email'])
users.dropna(axis=0)
df = users.fillna(0)


users['age'].interpolate(method='values')
