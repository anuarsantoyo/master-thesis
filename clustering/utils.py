import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

behaviour_cols = ['Q1_2_covid_is_threat',
 'Q2_1_easy_to_follow_advice',
 'Q2_2_can_follow_advice_if_wanted',
 'Q2_3_if_follow_advice_safe',
 'Q2_4_if_follow_advice_others_safe',
 'Q2_5_follow_advice_relationships_impared',
 'Q2_6_follow_advice_life_degraded',
 'Q3_1_aware_hand_hygiene',
 'Q3_2_avoid_contact',
 'Q3_3_ensure_frequent_cleaning',
 'Q3_4_avoid_risk_groups',
 'Q3_5_keep_distance',
 'Q3_6_avoid_crowds',
 'Q3_7_minimize_activities_w_contact',
 'Q5_4_yourself_kept_distance',
 'Q5_5_feel_urge_scold',
 'Q6_2_advices_important',
 'Q6_3_others_can_avoid_spreading',
 'Q6_5_ownership_of_advice',
 'Q6_6_clear_information_on_advice_reason',
 'Q6_7_advice_limits_daily_activities',
 'Q6_9_trust_political_strategy',
 'Ny1_nr_times_wearing_masks_last_week']

def get_cluster_input_data(data_path='data/preprocessing/220216_preprocessed_data_missing_data.csv', scaler = MinMaxScaler(), pca_data=False):
  if pca_data == False:
      df = pd.read_csv(data_path)
      cluster_input_cols = behaviour_cols
      cluster_input_raw = df[cluster_input_cols].to_numpy()
    
      scaler.fit(cluster_input_raw)
      cluster_input = scaler.transform(cluster_input_raw)
    
  if pca_data == True:
     data_path = 'data/preprocessing/dim_reduction/220324_pca_data.csv'
     df = pd.read_csv(data_path)
     cluster_input_cols = df.iloc[:,:11].columns.tolist()
     cluster_input = df[cluster_input_cols].to_numpy()

  info_dict = {'data_path': data_path, 'cluster_input_cols': cluster_input_cols, 'scaler_type':  scaler.__str__(), 'pca_data': pca_data}

  return df, cluster_input, info_dict

def get_behaviour_cols():
    return behaviour_cols
