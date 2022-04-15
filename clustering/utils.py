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
 'Q3b_1_sneeze_sleeve',
 'Q5_4_yourself_kept_distance',
 'Q5_5_feel_urge_scold',
 'Q6_2_advices_important',
 'Q6_3_others_can_avoid_spreading',
 'Q6_5_ownership_of_advice',
 'Q6_6_clear_information_on_advice_reason',
 'Q6_7_advice_limits_daily_activities',
 'Q6_9_trust_political_strategy'] # ,'Ny1_nr_times_wearing_masks_last_week'
 
dict_combination = {
    'group_q2_following_advice': ['Q2_1_easy_to_follow_advice', 'Q2_2_can_follow_advice_if_wanted', 'Q2_3_if_follow_advice_safe', 'Q2_4_if_follow_advice_others_safe'],
    'group_q2_consequence_advice': ['Q2_5_follow_advice_relationships_impared', 'Q2_6_follow_advice_life_degraded'],
    'group_q3_cleaning': ['Q3_1_aware_hand_hygiene', 'Q3_3_ensure_frequent_cleaning'],
    'group_q3_distancing': ['Q3_2_avoid_contact', 'Q3_4_avoid_risk_groups', 'Q3_5_keep_distance', 'Q3_6_avoid_crowds', 'Q3_7_minimize_activities_w_contact', 'Q5_4_yourself_kept_distance'],
    'group_q5_beh_other': ['Q5_1_others_took_distance', 'Q5_2_others_follow_advice'],
    'group_q6_opinion': ['Q6_2_advices_important', 'Q6_4_advices_create_fair_burden_dristribution', 'Q6_5_ownership_of_advice',
                            'Q6_6_clear_information_on_advice_reason', 'Q6_9_trust_political_strategy'],
    'group_q7_symptoms': ['Q7_1_last_week_fever', 'Q7_2_last_week_cough', 'Q7_3_last_week_sore_throat', 'Q7_4_last_week_no_smell_taste', 'Q7_5_last_week_shortness_breath'],
    'group_q4_contacts': ['Q4_1_nr_contact_nonhouse_family', 'Q4_2_nr_contact_colleagues', 'Q4_3_nr_contact_friends', 'Q4_4_nr_contact_strangers'],
    'group_household': ['Q11_nr_members_household', 'Q12_nr_children_household']}
 
grouped_behaviour_cols = ['group_q2_following_advice', 'group_q2_consequence_advice', 'group_q3_cleaning',
'group_q3_distancing', 'group_q5_beh_other','group_q6_opinion', 'group_q4_contacts']

behaviour_cols_combined = behaviour_cols + grouped_behaviour_cols
for values in dict_combination.values():
  for element in values:
    if element in behaviour_cols_combined:
      behaviour_cols_combined.remove(element)


def get_behaviour_cols():
    return behaviour_cols

def get_dict_combination():
    return dict_combination
    
def get_grouped_behaviour_cols():
    return grouped_behaviour_cols

def get_behaviour_cols_combined():
    return behaviour_cols_combined

def get_preprocessed_data(data_path='data/preprocessing/220407_preprocessed_data_without_imputation.csv', impute=True, impute_cols=behaviour_cols):
    df = pd.read_csv(data_path)
    if impute:
        thresh_drop = int(len(impute_cols) * 0.9)
        df.dropna(thresh=thresh_drop, subset=impute_cols, inplace=True)
        for column in impute_cols:
            df[column].fillna(value=df[column].mean(), inplace=True)
    return df

def get_cluster_input_data(data_path='data/preprocessing/220407_preprocessed_data_without_imputation.csv', scaler = MinMaxScaler(), pca_data=False, grouped_data=False, combined_data=False, fa_data=False):
  if pca_data:
      data_path = 'data/preprocessing/dim_reduction/220407_pca_data.csv'
      df = pd.read_csv(data_path)
      cluster_input_cols = df.iloc[:, :10].columns.tolist()
      cluster_input = df[cluster_input_cols].to_numpy()
      scaler = None

  elif fa_data:
      data_path = 'data/preprocessing/dim_reduction/220415_fa_data.csv'
      df = pd.read_csv(data_path)
      cluster_input_cols = df.iloc[:, :6].columns.tolist()
      cluster_input = df[cluster_input_cols].to_numpy()
      scaler = None

  elif grouped_data:
      data_path = 'data/preprocessing/dim_reduction/220407_grouped_data.csv'
      df = pd.read_csv(data_path)
      cluster_input_cols = grouped_behaviour_cols
      cluster_input = df[cluster_input_cols].to_numpy()
      
  elif combined_data:
      data_path = 'data/preprocessing/dim_reduction/220407_grouped_data.csv'
      df = pd.read_csv(data_path)
      cluster_input_cols = behaviour_cols_combined
      cluster_input = df[cluster_input_cols].to_numpy()


  else:
      cluster_input_cols = behaviour_cols
      df = get_preprocessed_data(data_path, impute=True, impute_cols=cluster_input_cols)
      cluster_input = df[cluster_input_cols].to_numpy()
    
  if scaler != None:
      scaler.fit(cluster_input)
      cluster_input = scaler.transform(cluster_input)

  info_dict = {'data_path': data_path, 'cluster_input_cols': cluster_input_cols, 'scaler_type':  scaler.__str__(), 'pca_data': pca_data, 'grouped_data': grouped_data, 'combined_data': combined_data}

  return df, cluster_input, info_dict



