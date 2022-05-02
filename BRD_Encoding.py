#!/usr/bin/env python3

import pandas as pd
import numpy as np

######################NEW PLAN - see plan.jpeg in directory (lol)##########################
unique_drugs_to_inchi_keys = pd.read_csv('unique_drugs_to_inchi_keys.txt', sep='\t')
print(unique_drugs_to_inchi_keys.head()) 
compoundinfo_beta = pd.read_csv('../../Data/LINCS L1000/compoundinfo_beta.txt', sep='\t')
compoundinfo_beta = compoundinfo_beta[compoundinfo_beta.compound_aliases != ""] #Do i need this?
compoundinfo_beta = compoundinfo_beta.drop_duplicates('inchi_key')
print(len(list(compoundinfo_beta.inchi_key)))
f = open("BRD_encodings_via_unique_drugs_to_inchi_keys.tsv", "w")
previous_unique_drug_name_match = ""
for index, row in unique_drugs_to_inchi_keys.iterrows():
    if previous_unique_drug_name_match == row["drugName"]:
        continue
    inchi_key = row["inchi_key"]
    unique_drug_name = row["drugName"]
    for index, row in compoundinfo_beta.iterrows():
        if inchi_key == row["inchi_key"]:
            f.write(str(unique_drug_name))
            f.write("\t")
            f.write(str(inchi_key))
            f.write("\t")
            f.write(str(row["pert_id"]))
            f.write("\n")
            previous_unique_drug_name_match = unique_drug_name
            print(unique_drug_name)
            break
f.close()
