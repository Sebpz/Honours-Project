import numpy as np 
import pandas as pd

CCLE = pd.read_csv("../../../Data/feature_data/CCL/CCLE_expression.csv")
sample_info =  pd.read_csv("../../../Data/feature_data/CCL/sample_info.csv")
#Data = pd.read_csv("../../Data/Narjes_data/Data.tsv", sep="\t")
#print(Data[Data["cell_line_name"] == "DU-145"])
print(CCLE.shape, sample_info.shape)
#print(sample_info.cell_line_name)
CCLs = ['A549', 'HT29', 'MCF7', 'MDAMB468', 'PC3', 'T47D',
		'A375', 'VCAP', 'MDAMB231', 'ES2', 'K562',
		'SKMEL5', 'HS578T', 'DU145', 'HCT116']
sample_info = sample_info[sample_info["stripped_cell_line_name"].isin(CCLs)]
#print(sample_info.stripped_cell_line_name.unique())
#rint(CCLE.DepMap_ID.unique())
CCLE = CCLE[CCLE["DepMap_ID"].isin(list(sample_info.DepMap_ID))]
print(CCLE)
CCLE.to_csv("../../../Data/feature_data/CCL/CCLE_expression_14.csv", index=False)
