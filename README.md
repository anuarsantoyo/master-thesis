# SARS-CoV-2 Infections and Social Dynamics: Modeling the Pandemic in Denmark with Survey-based Behavioural Clusters
In our master thesis we use survey information taken from danish people regarding the Covid 19 pandemic to identify  
different behavioural groups throughout 
the pandemic. After identifying the different groups we try to model the pandemic using this information to find out 
how do the different groups affected 
the development of the pandemic.

## Preprocessing
* translation.ipynb: Loads data/rawcontacts_Msc_Thesis.csv translates from danish to english and creates file 
data/rawcontacts_Msc_Thesis_eng.csv
* preprocessing.ipynb: Loads data/rawcontacts_Msc_Thesis_eng.csv runs several preprocessing steps 
(mapping, rename columns, ...) and creates file data/preprocessing/preprocessed_data_without_imputation.csv
* get_observations.ipynb: Loads covid observation data in Demark (already saved in data) and combines it to create
data/observations.csv
* PCA.ipynb: Loads data/preprocessing/preprocessed_data_without_imputation.csv and creates PCA dimensionality reduction 
of the data, analyzes it and saves it in data/preprocessing/dim_reduction/pca_data.csv
* FactorAnalysis.ipynb: Loads data/preprocessing/preprocessed_data_without_imputation.csv and creates 
Factor Analysis dimensionality reduction of the data, analyzes it and saves it in 
data/preprocessing/dim_reduction/fa_data.csv

## Clustering
* Experiments_Clustering.ipynb: Runs clustering experiments, that are used to determine the best number of clusters and
method, and saves results in data/clustering/results/results_clustering.csv
* evaluation.ipynb: Visual evaluation of clustering results
* utils.py: collection of support function mainly used in clustering

## Model
* model_optimization_randomwalk.ipynb: Optimizes the model using random walk method and used to infer R-value and saves
it in data/inferred_rvalue.csv.
* model_optimization_linear.ipynb: Optimizes the model using linear regression method.
* model_optimization_nn.ipynb: Optimizes the model using Neural network.
* comparison_rvalue.ipynb: Compares the inferred R-value from data/inferred_rvalue.csv with the cluster percentage and
factor values overtime.
* modelcore.py: Function that runs the epidemiological model with R value as an input and expected newly infected and 
hospitalization as an output.
* modelhelper.py: Collection of support functions used in the model.
* methods.py: Collection of methods that convert several input data to R values to be used by the model.

Archiv folders are used to save files that we think we didn't need but were to afraid to erase.