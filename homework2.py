import os.path
import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

users_original_df = pd.read_csv('users.csv')
users_original_df['deleted_at'] = pd.to_datetime(users_original_df['deleted_at'])
users_original_df = users_original_df.dropna(subset=["deleted_at"])
users_original_df['deleted_period'] = users_original_df.deleted_at.apply(lambda x: x.strftime('%m'))

cohort_df = users_original_df.set_index('id')
cohort_df['cohort_group'] = cohort_df.groupby(level=0)['deleted_at'].min().apply(lambda x: x.strftime('%m'))
cohort_df = cohort_df.reset_index()
cohort_df = cohort_df.sort_values('id')
grouped = cohort_df.groupby(['cohort_group', 'deleted_period'])

cohorts = grouped.agg({'id': pd.Series.nunique})
cohorts.rename(columns={"id": "total_users"})
def cohort_period(df):
      df['cohort_period'] = np.arange(len(df)) + 1
      return df
