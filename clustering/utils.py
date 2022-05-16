import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

behaviour_cols = ['Q1_1_feel_exposed',
 'Q1_2_covid_is_threat',
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
 'Q3b_1_sneeze_sleeve',
 'Q4_1_nr_contact_nonhouse_family',
 'Q4_2_nr_contact_colleagues',
 'Q4_3_nr_contact_friends',
 'Q4_4_nr_contact_strangers',
 'Q5_1_others_took_distance',
 'Q5_2_others_follow_advice',
 'Q5_3_others_not_care_spreading',
 'Q5_4_yourself_kept_distance',
 'Q5_5_feel_urge_scold',
 'Q6_1_sanctions_are_too_harsh',
 'Q6_2_advices_important',
 'Q6_3_others_can_avoid_spreading',
 'Q6_4_advices_create_fair_burden_dristribution',
 'Q6_5_ownership_of_advice',
 'Q6_6_clear_information_on_advice_reason',
 'Q6_7_advice_limits_daily_activities',
 'Q6_8_advices_enough_for_prevention',
 'Q6_9_trust_political_strategy']
 
behaviour_cols_original = ['Q1_2_covid_is_threat',
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
 'Q3b_1_sneeze_sleeve',
 'Q5_4_yourself_kept_distance',
 'Q5_5_feel_urge_scold',
 'Q6_2_advices_important',
 'Q6_3_others_can_avoid_spreading',
 'Q6_5_ownership_of_advice',
 'Q6_6_clear_information_on_advice_reason',
 'Q6_7_advice_limits_daily_activities',
 'Q6_9_trust_political_strategy']
 
dict_combination = {
    'F0_cautious_behaviour': ['Q3_1_aware_hand_hygiene', 'Q3_2_avoid_contact', 'Q3_3_ensure_frequent_cleaning',
    'Q3_4_avoid_risk_groups', 'Q3_5_keep_distance', 'Q3_6_avoid_crowds', 'Q3_7_minimize_activities_w_contact', 'Q5_4_yourself_kept_distance'],
    'F1_perception_advice': ['Q6_2_advices_important', 'Q6_4_advices_create_fair_burden_dristribution', 'Q6_5_ownership_of_advice',
                            'Q6_6_clear_information_on_advice_reason', 'Q6_8_advices_enough_for_prevention', 'Q6_9_trust_political_strategy'],
    'F2_applicability_usefullness_advice': ['Q2_1_easy_to_follow_advice', 'Q2_2_can_follow_advice_if_wanted', 'Q2_3_if_follow_advice_safe', 'Q2_4_if_follow_advice_others_safe'],
    'F3_behaviour_others': ['Q5_1_others_took_distance', 'Q5_2_others_follow_advice'],
    'F4_consequence_advice': ['Q2_5_follow_advice_relationships_impared', 'Q2_6_follow_advice_life_degraded', 'Q6_7_advice_limits_daily_activities'],
    'F5_no_contacts': ['Q4_1_nr_contact_nonhouse_family', 'Q4_2_nr_contact_colleagues', 'Q4_3_nr_contact_friends', 'Q4_4_nr_contact_strangers'],
    'F6_perceived_threat': ['Q1_1_feel_exposed', 'Q1_2_covid_is_threat']}
 
factor_cols = ['F0_cautious_behaviour', 'F1_perception_advice', 'F2_applicability_usefullness_advice', 'F3_behaviour_others', 'F4_consequence_advice', 'F5_no_contacts', 'F6_perceived_threat']

behaviour_cols_combined = behaviour_cols + factor_cols
for values in dict_combination.values():
  for element in values:
    if element in behaviour_cols_combined:
      behaviour_cols_combined.remove(element)


def get_behaviour_cols():
    return behaviour_cols

def get_dict_combination():
    return dict_combination
    
def get_factor_cols():
    return factor_cols

def get_behaviour_cols_combined():
    return behaviour_cols_combined

def get_preprocessed_data(data_path='data/preprocessing/220427_preprocessed_data_without_imputation.csv', impute=True, impute_cols=behaviour_cols, start='2020-05-28', end='2021-12-02'):
    df = pd.read_csv(data_path)
    df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
    df = df[(df.date > start) & (df.date < end)]
    df.reset_index(inplace=True, drop=True)
    if impute:
        thresh_drop = int(len(impute_cols) * 0.9)
        df.dropna(thresh=thresh_drop, subset=impute_cols, inplace=True)
        df.dropna(subset=['date'], inplace=True)
        for column in impute_cols:
            df[column].fillna(value=df[column].mean(), inplace=True)
    return df


def get_cluster_input_data_new(scaler = MinMaxScaler(), pca_data=False, fa_data=False, start_train='2020-05-28', end_train='2021-12-02'):
  if pca_data:
      data_path = 'data/preprocessing/dim_reduction/220513_pca_data.csv'
      cluster_input_cols = []
      for i in np.arange(9):
        col_name = 'PC_' + str(i)
        cluster_input_cols.append(col_name)
      scaler = None

  elif fa_data:
      data_path = 'data/preprocessing/dim_reduction/220513_fa_data.csv'
      cluster_input_cols = factor_cols


  df = pd.read_csv(data_path)
  df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
  df_cluster_input = df[(df.date > start_train) & (df.date < end_train)].copy()
  df_cluster_input.reset_index(inplace=True, drop=True)
  cluster_input_train = df_cluster_input[cluster_input_cols].to_numpy()
  cluster_input_all = df[cluster_input_cols].to_numpy()
    
  if scaler != None:
      scaler.fit(cluster_input_train)
      cluster_input_train_scaled = scaler.transform(cluster_input_train)

  info_dict = {'data_path': data_path, 'cluster_input_cols': cluster_input_cols, 'scaler_type':  scaler.__str__(), 'pca_data': pca_data, 'fa_data': fa_data, 'start_train': start_train, 'end_train': end_train}

  return df, cluster_input_train, cluster_input_all, info_dict









def get_cluster_input_data(data_path='data/preprocessing/220427_preprocessed_data_without_imputation.csv', scaler = MinMaxScaler(), pca_data=False, fa_data=False, start='2020-05-28', end='2021-12-02'):
  if pca_data:
      data_path = 'data/preprocessing/dim_reduction/220513_pca_data.csv'
      df = pd.read_csv(data_path)
      cluster_input_cols = df.iloc[:, :9].columns.tolist()
      df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
      df_cluster_input = df[(df.date > start) & (df.date < end)]
      df_cluster_input.reset_index(inplace=True, drop=True)
      cluster_input = df_cluster_input[cluster_input_cols].to_numpy()
      scaler = None

  elif fa_data:
      data_path = 'data/preprocessing/dim_reduction/220513_fa_data.csv'
      df = pd.read_csv(data_path)
      cluster_input_cols = factor_cols
      df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
      df_cluster_input = df[(df.date > start) & (df.date < end)]
      df_cluster_input.reset_index(inplace=True, drop=True)
      cluster_input = df_cluster_input[cluster_input_cols].to_numpy()

  else:
      cluster_input_cols = behaviour_cols
      df = get_preprocessed_data(data_path, impute=True, impute_cols=cluster_input_cols)
      df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
      df_cluster_input = df[(df.date > start) & (df.date < end)]
      df_cluster_input.reset_index(inplace=True, drop=True)
      cluster_input = df_cluster_input[cluster_input_cols].to_numpy()
    
  if scaler != None:
      scaler.fit(cluster_input)
      cluster_input = scaler.transform(cluster_input)

  info_dict = {'data_path': data_path, 'cluster_input_cols': cluster_input_cols, 'scaler_type':  scaler.__str__(), 'pca_data': pca_data, 'fa_data': fa_data}

  return df, cluster_input, info_dict



