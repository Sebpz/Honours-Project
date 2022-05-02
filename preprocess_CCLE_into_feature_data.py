#!/usr/bin/env python3

import numpy as np 
import pandas as pd

#remove training labels the contain DU-15 + show number of remaing labels + save
#encoded_CCLE = np.load('../../Data/feature_data/CCL/encoded_CCLE_expression_14.npy')
CCLE = pd.read_csv("../../../Data/feature_data/CCL/CCLE_expression_14.csv")
print(CCLE)
Training_labels = pd.read_csv("../../../Data/Training_data/Training_data_tas_method_added_scores.tsv", sep="\t")
print(Training_labels.shape)
Training_labels = Training_labels[Training_labels["cell_line_name"] != "DU145"]
Training_labels.to_csv("../../../Data/Training_data/CCLE_no_DU145.csv", index_label="Old_index")
print(Training_labels.shape)
CCL_feature_data = pd.read_csv("../../../Data/feature_data/CCL/CCL_feature_data.csv")
print(CCL_feature_data.shape)
#'''
sample_info =  pd.read_csv("../../../Data/feature_data/CCL/sample_info.csv")
i = 0
#print()
for index, row in Training_labels.iterrows():
	tmp = sample_info[sample_info["stripped_cell_line_name"] == row.cell_line_name]
	if i == 0:
		CCL_feature_data = CCLE[CCLE["DepMap_ID"] == tmp.iloc[0][0]].copy()
	else:
		CCL_feature_data_tmp = CCLE[CCLE["DepMap_ID"] == tmp.iloc[0][0]]
		CCL_feature_data = CCL_feature_data.append(CCL_feature_data_tmp.copy(), ignore_index=True)
	i += 1
print(CCL_feature_data.shape)
CCL_feature_data.to_csv("../../../Data/feature_data/CCL/CCL_feature_data.csv", index=False)
#'''

