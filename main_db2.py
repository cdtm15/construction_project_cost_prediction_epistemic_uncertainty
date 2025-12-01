#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:47:03 2024

@author: cristiantobar
"""
import pandas as pd


def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]

def ml_performance(classif_metrics, model):
    
    if model == 'ct':
        metric_model = classif_metrics[:,:,0]
        
    if model == 'forest':
        metric_model = classif_metrics[:,:,1]
        
    if model == 'logistic':
        metric_model = classif_metrics[:,:,2]
        
    if model == 'knn':
        metric_model = classif_metrics[:,:,3]
    
    if model == 'svm':
        metric_model = classif_metrics[:,:,4]
    
    if model == 'ann':
        metric_model = classif_metrics[:,:,5]
        
    metric_model = pd.DataFrame(metric_model)
    metric_model.columns = ['num_features1','num_features', 'train_size', 'test_size', 'R2', 'Accuracy', 'Sensitivity', 'Specificity', 'F1', 'MSE']

    new_metric = metric_model.drop('num_features1', axis=1).copy()
    
    return new_metric

clear_all()

import numpy as np
from file_data_reception_db2 import data_reception_db2
from file_data_understanding_db2 import data_understanding_db2
from file_data_preparation_db2 import data_preparation_db2
from file_modeling_db2 import modeling_db2
from file_modeling_regresion_db2 import modeling_regresion_db2

#data_location       = "/Users/cristiantobar/Library/CloudStorage/OneDrive-CDTCreaTIC/Fortalecimiento/Investigacion/doctorado_cristian/procesamiento_datos/base_datos_3/datos"
data_location       = "/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/base_datos_3/datos"
data_filename       = ["database_2_csv.csv"]
merged_path         = data_location + "/" + data_filename[0]
test_portion        = [0.2, 0.25, 0.3]
threshold           = ['Study', 'First', 'Median', 'Third', 'Adaptative', 'hierarchical']
models              = ['ct', 'forest', 'logistic', 'knn', 'svm', 'ann']
print_figures       = True
currency            = ['Million COPm', 'DOLLARm']
dollar2cop          = 4333.11
reg_models          = ['ann', 'svm', 'forest']


df                  = data_reception_db2(merged_path)
filt_df_encoded, filt_df, sorted_feat, df_clust, full_df_clust = data_preparation_db2(df, currency[1], dollar2cop)
#corr, desc         = data_understanding_db2(filt_df, currency[1])

df_proj_0           = full_df_clust[full_df_clust['output']==0]
df_proj_1           = full_df_clust[full_df_clust['output']==1]

#np.save('project_1.npy', df_proj_1)
#df_proj_1.to_csv('out.csv', index=False)  
#corr_0, desc_0      = data_understanding_db2(df_proj_1, currency[0])

reg_proj_0_ann, proj_0_sorted_feat = modeling_regresion_db2(df_proj_0, 'proj_0', 'ann')
reg_proj_0_svm, _ = modeling_regresion_db2(df_proj_0, 'proj_0', 'svm')
reg_proj_0_rf, _  = modeling_regresion_db2(df_proj_0, 'proj_0', 'rf')

reg_proj_1_ann, proj_1_sorted_feat = modeling_regresion_db2(df_proj_1, 'proj_1', 'ann')
reg_proj_1_svm, _ = modeling_regresion_db2(df_proj_1, 'proj_1', 'svm')
reg_proj_1_rf, _  = modeling_regresion_db2(df_proj_1, 'proj_1', 'rf')

#reg_proj_1 = modeling_regresion_db2(df_proj_1, 'proj_1')





# classif_metrics     = modeling_db2(df_clust, sorted_feat)

# met_1_ct     = ml_performance(classif_metrics,'ct')
# met_2_forest = ml_performance(classif_metrics,'forest')
# met_3_log    = ml_performance(classif_metrics,'logistic')
# met_4_knn    = ml_performance(classif_metrics,'knn')
# met_5_svm    = ml_performance(classif_metrics,'svm')
# met_6_ann    = ml_performance(classif_metrics,'ann')


    


#corr, desc         = data_understanding_db2(new_df)





