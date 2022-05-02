#!/usr/bin/env python3

#import re
#def regex_changes(drug):

import pandas as pd

Training_data_post_aug = pd.read_csv("../LINCS_data_extraction/Training_data_post_averaging_aug_28_8.tsv", sep="\t")
DrugCombDB = pd.read_csv("../../Data/DrugCombDB/drugcombs_scored.csv")
no_of_samples = len(Training_data_post_aug.synergy_loewe)
Training_data_post_aug["ZIP"] = range(no_of_samples)
Training_data_post_aug["Bliss"] = range(no_of_samples)
Training_data_post_aug["HSA"] = range(no_of_samples)
print(Training_data_post_aug.shape)
i = 0
while i < no_of_samples:
	print(i)
	if i > 0 and Training_data_post_aug.iloc[i, 0:6].all() == Training_data_post_aug.iloc[i-1, 0:6].all():
		Training_data_post_aug.iloc[i, 7] = Training_data_post_aug.iloc[i-1, 7]
		Training_data_post_aug.iloc[i, 8] = Training_data_post_aug.iloc[i-1, 8]
		Training_data_post_aug.iloc[i, 9] = Training_data_post_aug.iloc[i-1, 9]
	#print(Training_data_post_aug.iloc[i, 0])
	drug1 = Training_data_post_aug.iloc[i, 0].replace("\'", "\\\'")
	#print(drug1)
	drug2 = Training_data_post_aug.iloc[i, 1].replace("\'", "\\\'")
	row = DrugCombDB[DrugCombDB["Drug1"] == drug1]
	#print(row)
	row = row[row["Drug2"] == drug2]
	#print(row)
	cell_line = Training_data_post_aug.iloc[i, 2].replace("MDAMB468", "MDA-MB-468")
	cell_line = cell_line.replace("MDAMB231", "MDA-MB-231")
	cell_line = cell_line.replace("PC3", "PC-3")
	cell_line = cell_line.replace("T47D", "T-47D")
	cell_line = cell_line.replace("K562", "K-562")
	cell_line = cell_line.replace("SKMEL5", "SK-MEL-5")
	cell_line = cell_line.replace("HS578T", "HS 578T")
	cell_line = cell_line.replace("DU145", "DU-145")
	row = row[row["Cell line"] == cell_line]

	#print(row.shape)
	if row.shape[0] > 1:
		Training_data_post_aug.iloc[i, 7] = float(row["ZIP"].mean())
		Training_data_post_aug.iloc[i, 8] = float(row["Bliss"].mean())
		Training_data_post_aug.iloc[i, 9] = float(row["HSA"].mean())
		print("duplicates case --- gotta match to loewe score in Data.tsv")
	else:
		Training_data_post_aug.iloc[i, 7] = float(row["ZIP"])
		Training_data_post_aug.iloc[i, 8] = float(row["Bliss"])
		Training_data_post_aug.iloc[i, 9] = float(row["HSA"])
	i += 1

Training_data_post_aug.to_csv("../../Data/Training_data/Training_data_tas_method_added_scores.tsv", sep="\t", index=False)
