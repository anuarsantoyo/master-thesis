# SARS-CoV-2 Infections and Social Dynamics: Modeling the Pandemic in Denmark with Survey-based Behavioural Clusters
In our master thesis we use survey information taken from danish people regarding the Covid 19 pandemic to identify diferent behavioural groups throughout 
the pandemic. After identifying the different groups we try to model the pandemic using this information to find out how do the diferent groups affected 
the development of the pandemic.

## Preprocessing
* translation.ipynb: Loads data/rawcontacts_Msc_Thesis.csv translates from danish to english and creates file data/rawcontacts_Msc_Thesis_eng.csv
* preprocessing.ipynb: Loads data/rawcontacts_Msc_Thesis_eng.csv runs several preprocessing steps (mapping, rename columns, ...) and creates file data/preprocessing/preprocessed_data_without_imputation.csv
* get_observations.ipynb: Loads covid observation data in Demark (already saved in data) and combines it to create data/observations.csv
* 

