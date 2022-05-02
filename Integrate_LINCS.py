#!/usr/bin/env python3

import pandas as pd
import numpy as np

import re
def cell_line_name_clean(cell_line):
	cell_line = re.sub("-", "", cell_line)
	cell_line = re.sub("\s", "", cell_line)
	return cell_line


import cmapPy
from cmapPy.pandasGEXpress.parse_gctx import parse
from cmapPy.pandasGEXpress.write_gctx import write
import random


#Create a function that sorts an array of sig_ids and returns an array of their averaged expression values
def aug_expression_signatures(Column_labels, sig_ids):
	if len(sig_ids) > 1:
		Sorting_sig_ids = {}
		index = 0
		for label in Column_labels:
			if label in sig_ids:
				Sorting_sig_ids[label] = index
			index += 1
		#taken from https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
		{k: v for k, v in sorted(Sorting_sig_ids.items(), key=lambda item: item[1])}
		Sorted_sig_ids = list(Sorting_sig_ids.keys())
		extracted_LINCS_data = parse("../../Data/LINCS L1000/level5_beta_trt_cp_n720216x12328.gctx", cid=Sorted_sig_ids)
		extracted_LINCS_data =  np.array(extracted_LINCS_data.data_df)
		extracted_LINCS_data = extracted_LINCS_data.T
		np.random.shuffle(extracted_LINCS_data)
		return extracted_LINCS_data.T
	else:
		extracted_LINCS_data = parse("../../Data/LINCS L1000/level5_beta_trt_cp_n720216x12328.gctx", cid=sig_ids)
		extracted_LINCS_data = np.array(extracted_LINCS_data.data_df)
		return extracted_LINCS_data.reshape(12328, 1)

def even_stacks(a, b):
	B = len(b.T)
	i = 0
	while len(a.T) > len(b.T):
		b = np.hstack((b, b[:, i].reshape(12328, 1)))
		i += 1
		if i == B:
			i = 0
	return b

def get_samples(meta):
	tmp_chk = meta[meta["is_hiq"] == 1]
	if len(tmp_chk) > 0:
		meta = meta[meta["is_hiq"] == 1]
	tas = np.array(meta.tas)
	threshold = tas.mean()+2*tas.std()
	x = 0
	while(x < threshold):
		tmp_chk = meta[meta["tas"] > x]
		if len(tmp_chk) > 0:
			meta = meta[meta["tas"] > x]
		else:
			break
		x += 0.01
	return meta
#'''


drugName_BRD_encoding = pd.read_csv('BRD_encodings_via_unique_drugs_to_inchi_keys.tsv', sep='\t')
Data = pd.read_csv("../../Data/Narjes_data/Data.txt", sep="\t")
Data["index"] = range(286421)
Data = Data[Data["drug_col"].isin(list(drugName_BRD_encoding.drugName))]
Data = Data[Data["drug_row"].isin(list(drugName_BRD_encoding.drugName))]

siginfo = pd.read_csv("../../Data/LINCS L1000/siginfo_beta.txt", sep="\t")
siginfo = siginfo.loc[siginfo["pert_type"] == "trt_cp"]

Data["cell_line_name"] = Data["cell_line_name"].apply(cell_line_name_clean)
Data = Data[Data["cell_line_name"].isin(list(siginfo.cell_iname.unique()))]
Data.to_csv("Training_data.tsv", sep="\t", index=False)

unique_drugNames = list(np.unique(np.concatenate((Data.drug_col, Data.drug_row))))
drugName_BRD_encoding = drugName_BRD_encoding[drugName_BRD_encoding["drugName"].isin(unique_drugNames)]
#drugName_BRD_encoding.to_csv("drugName_BRD_encoding_tmp.tsv", sep="\t", index=False)

siginfo = siginfo[siginfo["pert_id"].isin(list(drugName_BRD_encoding.BRD_Encoding))]
siginfo = siginfo[siginfo["cell_iname"].isin(list(Data.cell_line_name.unique()))]

unique_drugNames = list(np.unique(np.concatenate((Data.drug_col, Data.drug_row))))
drugName_BRD_encoding = drugName_BRD_encoding[drugName_BRD_encoding["drugName"].isin(unique_drugNames)]
drugName_BRD_encoding.to_csv("drugName_BRD_encoding_tmp.tsv", sep="\t", index=False)

f = open("Column_labels.txt", "r")
Column_labels = []
for line in f:
	line = re.sub("\n", "", line)
	Column_labels.append(line)
f.close()



############CREATE EXPRESSION DATA FEATURE TABLES##############
#0. create two 'encoding' columns in Data
no_of_samples = len(Data.synergy_loewe)
Data["drug_row_encoding"] = range(no_of_samples)
Data["drug_col_encoding"] = range(no_of_samples)
i = 0
print(Data)
while i < no_of_samples:
	encoding1 = 0
	encoding2 = 0
	for index, row in drugName_BRD_encoding.iterrows():
		if Data.iloc[i, 0] == row["drugName"]:
			Data.iloc[i, 5] = row["BRD_Encoding"]
			print(Data.iloc[i, 0])
			encoding1 = 1
		if Data.iloc[i, 1] == row["drugName"]:
			Data.iloc[i, 6] = row["BRD_Encoding"]
			print(Data.iloc[i, 1])
			encoding2 = 1
		if encoding1 == 1 and encoding2 == 1:
			break
	i += 1
#checkpoint 
Data.to_csv("Training_data_with_encodings.tsv", sep="\t", index=False)

#check encoding was successful
for index, row in Data.iterrows():
	if type(row["drug_row_encoding"]) == int:
		print(row)
	if type(row["drug_col_encoding"]) == int:
		print(row)
#'''
#'''
#1. create lists of sig_id 
Data = pd.read_csv("Training_data_post_averaging_24h.tsv", sep="\t")
drug_row_sig_ids = []
drug_col_sig_ids = []
next_drug_col_sample = []
next_drug_row_sample = []
first_sample_check = 0
append_size = 0
i = 0
New_Data = pd.DataFrame(columns=Data.columns, dtype=object)

for index, row1 in Data.iterrows():
	drug_row_sig_ids = []
	drug_col_sig_ids = []
	save = index
	target_meta = siginfo[siginfo["cell_iname"] == row1["cell_line_name"]]
	target_meta = target_meta[target_meta["pert_time"] == 24]
	#print(f"########################{save}#######################")
	drug_col = target_meta[target_meta["pert_id"] == row1["drug_col_encoding"]]
	drug_row = target_meta[target_meta["pert_id"] == row1["drug_row_encoding"]]
	if drug_col.empty or drug_row.empty:
		Data.drop(save, inplace=True)
		#print(f"Data dropped [drug_col: {row1.drug_col} drug_row: {row1.drug_row} cell_line_name: {row1.cell_line_name}")
	else:
		drug_col = get_samples(drug_col)
		drug_row = get_samples(drug_row)
		drug_col_sig_ids = list(drug_col.sig_id)
		drug_row_sig_ids = list(drug_row.sig_id)
		i += 1
		#print(f"sample number: {i} drug_col: {row1.drug_col} drug_row: {row1.drug_row} cell_line_name: {row1.cell_line_name}")
		if first_sample_check == 0:
			drug_col_samples = aug_expression_signatures(Column_labels, drug_col_sig_ids)
			drug_row_samples = aug_expression_signatures(Column_labels, drug_row_sig_ids)
			#print(drug_col_samples.shape, drug_row_samples, type(drug_col_samples), type(drug_row_samples))
			if drug_col_samples.shape[1] > drug_row_samples.shape[1]:
				append_size = drug_col_samples.shape[1]
				drug_row_samples = even_stacks(drug_col_samples, drug_row_samples)
			elif drug_col_samples.shape[1] < drug_row_samples.shape[1]:
				append_size = drug_row_samples.shape[1]
				drug_col_samples = even_stacks(drug_row_samples, drug_col_samples)
			else:
				append_size = drug_row_samples.shape[1]
			first_sample_check = 1
		else:
			next_drug_col_sample = aug_expression_signatures(Column_labels, drug_col_sig_ids)
			next_drug_row_sample = aug_expression_signatures(Column_labels, drug_row_sig_ids)
			if next_drug_col_sample.shape[1] > next_drug_row_sample.shape[1]:
				append_size = next_drug_col_sample.shape[1]
				next_drug_row_sample = even_stacks(next_drug_col_sample, next_drug_row_sample)
			elif next_drug_col_sample.shape[1] < next_drug_row_sample.shape[1]:
				append_size = next_drug_row_sample.shape[1]
				next_drug_col_sample = even_stacks(next_drug_row_sample, next_drug_col_sample)
			else:
				append_size = next_drug_row_sample.shape[1]
			drug_col_samples = np.hstack((drug_col_samples, next_drug_col_sample))
			drug_row_samples = np.hstack((drug_row_samples, next_drug_row_sample))      
		#print("line: ", Data.iloc[save])
		#print(drug_col_samples.shape, drug_row_samples.shape)
		j = 0
		#print("aug size = ", append_size)
		while j < append_size:
			New_Data = New_Data.append(Data.iloc[save].copy(), ignore_index=True)
			j += 1
New_Data.to_csv("Training_data_post_averaging_aug_28_8.tsv", sep="\t", index=False)
np.save("drug_col_samples_aug_28_8", drug_col_samples)
np.save("drug_row_samples_aug_28_8", drug_row_samples)

#'''
######CREATE DRUG FEATURE DATA#########
def drug_feature_extraction(old_file, new_file, matched_indexes):
	f = open(old_file, "r")
	g = open(new_file, "w")
	i = 0
	for line in f:
		if i in matched_indexes:
			x = 0
			repeats = matched_indexes.count(i) 
			while x < repeats:
				g.write(line)
				x += 1
		i += 1
	f.close()
	g.close()

Data = pd.read_csv("Training_data_post_averaging_aug_28_8.tsv", sep="\t")
matched_indexes = list(Data.index)
old_file = "../../Data/Narjes_data/drug1_chem/drug1_chem.csv"
new_file = "../../Data/Narjes_data/drug1_chem/28_8_drug1_chem.csv"
drug_feature_extraction(old_file, new_file, matched_indexes)
old_file = "../../Data/Narjes_data/drug2_chem/drug2_chem.csv"
new_file = "../../Data/Narjes_data/drug2_chem/28_8_drug2_chem.csv"
drug_feature_extraction(old_file, new_file, matched_indexes)
#'''
