#!/usr/bin/env python3

import pandas as pd
import numpy as np

def make_cut_off_cols(lower, upper, training_labels):
	conditions = [
		(training_labels[score] <= lower),
		(training_labels[score] >= upper),
		(training_labels[score] > lower) & (training_labels[score] < upper)
	]
	#make corresponding labels
	values_categorical = ["antagonistic", "synergistic", "additive"]
	values_numerical = [0, 1, 2]
	training_labels["class_catergorical"] = np.select(conditions, values_categorical)
	training_labels["class_numerical"] = np.select(conditions, values_numerical)
	#print(sum(training_labels["class_catergorical"] == "antagonistic"))
	#print(sum(training_labels["class_catergorical"] == "synergistic"))
	#print(sum(training_labels["class_catergorical"] == "additive"))
	return training_labels.copy()


def cut_off(training_labels, score):
	#calculate IQR cut-offs and make training data
	Q1 = training_labels[score].quantile(0.25)
	Q3 = training_labels[score].quantile(0.75)
	#print(score+"	Q1:",Q1,"	Q3:",Q3)
	IQR_training_labels = make_cut_off_cols(Q1,Q3,training_labels)

	#calculate SD cut-offs and make training data
	positive_cut_off = training_labels[score].mean()+training_labels[score].std()
	negative_cut_off = training_labels[score].mean()-training_labels[score].std()
	#print(score+"	positive_cut_off:",positive_cut_off,"	negative_cut_off:",negative_cut_off)
	SD_training_labels = make_cut_off_cols(negative_cut_off,positive_cut_off,training_labels)

	#Save three class datasets
	IQR_three_class_labels = "../../../Data/Training_data/New_labels/tas_method/3_classes/"+score+"_IQR.tsv"
	SD_three_class_labels = "../../../Data/Training_data/New_labels/tas_method/3_classes/"+score+"_SD.tsv"
	IQR_training_labels.to_csv(IQR_three_class_labels, sep="\t", index=False)
	SD_training_labels.to_csv(SD_three_class_labels, sep="\t", index=False)

	#delete third class and save binary label data
	IQR_training_labels = IQR_training_labels[IQR_training_labels["class_catergorical"] != "additive"]
	IQR_binary_class_label = "../../../Data/Training_data/New_labels/tas_method/binary_classes/"+score+"_IQR.tsv"
	IQR_training_labels.to_csv(IQR_binary_class_label, sep="\t", index_label="New_index")
	#print(IQR_training_labels.shape)
	SD_training_labels = SD_training_labels[SD_training_labels["class_catergorical"] != "additive"]
	SD_binary_class_label = "../../../Data/Training_data/New_labels/tas_method/binary_classes/"+score+"_SD.tsv"
	SD_training_labels.to_csv(SD_binary_class_label, sep="\t", index_label="New_index")
	#print(SD_training_labels.shape)


scores_data = pd.read_csv("../CCLE/CCLE_no_DU145.csv")
scores = ['synergy_loewe', 'ZIP', 'Bliss', 'HSA']
score = 'synergy_loewe'
#
#scores_data = scores_data[scores_data["synergy_loewe"] >= 3 ]
#print(scores_data)

#threePointFive = make_cut_off_cols(scores_data['synergy_loewe'].quantile(0.25), 3.5, scores_data)
#threePointFive.to_csv("../../../Data/Training_data/New_labels/tas_method/3_classes/synergy_loewe_threePointFive.tsv", sep="\t", index=False)
for score in scores:
	cut_off(scores_data, score)
