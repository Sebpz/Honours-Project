main_df = pd.read_csv("../../Data/Training_data/New_labels/tas_method/3_classes/Bliss_IQR.tsv", sep="\t")

def get_test_samples(main_df):
	#extract all unique D+CCL pairs from drug_col+cell_line_name and drug_row+cell_line_name = DCCL
	#Use pd.series.functions
	D1_CCL = main_df["drug_row"]+"_"+main_df["cell_line_name"]
	D2_CCL = main_df["drug_col"]+"_"+main_df["cell_line_name"]
	D_CCL = pd.concat([D1_CCL, D2_CCL]).unique()

	#Make list of current D1D2CCL = currD1D2CCL
	currD1D2CCL = main_df["drug_row"]+"_"+main_df["drug_col"]+"_"+main_df["cell_line_name"]
	currD1D2CCL = currD1D2CCL.unique()


	#increase size of currD1D2CCL to include D2D1CCL as well as D1D2CCL
	def double_sample(D1D2CCL):
		D1D2CCL = D1D2CCL.split("_")
		D2D1CCL = D1D2CCL[1]+"_"+D1D2CCL[0]+"_"+D1D2CCL[2]
		return D2D1CCL

	currD1D2CCL = pd.DataFrame(currD1D2CCL, columns=['D1D2CCL'])
	currD1D2CCL["D2D1CCL"] = currD1D2CCL["D1D2CCL"].map(lambda x: double_sample(x))
	currDDCCL = pd.concat([currD1D2CCL["D2D1CCL"], currD1D2CCL["D1D2CCL"]])


	#For all Unique CCL strings in main_df
	CCL = main_df["cell_line_name"].unique()
	New_samples = []

	def join_samples(sample):
		a, b = sample[0], sample[1]
		b = b.split("_")
		return b[0]+"_"+a

	for ccl in CCL:
		
		#Reduce DCCL to DCCLi includes the ith CCL
		tmp_D_CCL = list(D_CCL.copy())
		tmp_D_CCL = [x for x in tmp_D_CCL if ccl in x]

		#Append to List of D1D2CCLi not in currD1D2CCL
		com = itertools.combinations(tmp_D_CCL, 2)
		#print(list(com))
		for sample in list(com):
			sample = join_samples(sample)
			if sample not in list(currDDCCL):
				New_samples.append(sample)
	
	Test = pd.DataFrame(New_samples, columns=["New_samples"])
	Test["Drug_1"] = Test["New_samples"].map(lambda x: x.split("_").pop(0))
	Test["Drug_2"] = Test["New_samples"].map(lambda x: x.split("_").pop(1))
	Test["CCL"] = Test["New_samples"].map(lambda x: x.split("_").pop(2))
	
	def match_LINCS(row, main_df, drug_n):
		tmp = main_df[main_df["cell_line_name"] == row["CCL"]].copy()
		if row[drug_n] in list(tmp.drug_col):
			tmp = tmp[tmp["drug_col"] == row[drug_n]]
			a = "drug_col"
		elif row[drug_n] in list(tmp.drug_row):
			tmp = tmp[tmp["drug_row"] == row[drug_n]]
			a = "drug_row"
		return [tmp.index[0], a]

	Test["D1_index"] = Test.apply(lambda row: match_LINCS(row, main_df, "Drug_1"), axis=1)
	Test["D2_index"] = Test.apply(lambda row: match_LINCS(row, main_df, "Drug_2"), axis=1)
	Test.to_csv("Test_labels.tsv", sep="\t")

	get_Chemopy_features(Test, "D1_index")
	get_Chemopy_features(Test, "D2_index")
	get_CCLE_features(Test)
	get_LINCS_features(Test, "D1_index")
	get_LINCS_features(Test, "D2_index")

def remove_DU145(a):
	no_DU145 = pd.read_csv("../../Data/Training_data/CCLE_no_DU145.csv")
	a = a[no_DU145.Old_index,]
	return a


#Get LINCS features
def get_LINCS_features(Test, n):
	drug_row = remove_DU145(np.load("../LINCS_data_extraction/drug_row_samples_aug_28_8.npy").T)
	drug_col = remove_DU145(np.load("../LINCS_data_extraction/drug_col_samples_aug_28_8.npy").T)
	x = 0
	for index, row in Test.iterrows():
		if x == 0:
			if row[n][1] == "drug_row":
				drug_n = np.copy(drug_row[row[n][0]][:])
			elif row[n][1] == "drug_col":
				drug_n = np.copy(drug_col[row[n][0]][:])
		else:
			if row[n][1] == "drug_row":
				drug_n = np.vstack((drug_n, np.copy(drug_row[row[n][0]][:])))
			elif row[n][1] == "drug_col":
				drug_n = np.vstack((drug_n, np.copy(drug_col[row[n][0]][:])))
		x += 1
	n = n.replace("_index", "")
	np.save(f"{n}_LINCS_features", drug_n)

#Get ChemoPy features
def get_Chemopy_features(Test, n):
	drug1_chem = pd.DataFrame(remove_DU145(np.array(pd.read_csv("../../Data/Narjes_data/drug1_chem/28_8_drug1_chem.csv", header=None))))
	drug2_chem = pd.DataFrame(remove_DU145(np.array(pd.read_csv("../../Data/Narjes_data/drug2_chem/28_8_drug2_chem.csv", header=None))))
	x = 0
	for index, row in Test.iterrows():
		if x == 0:
			if row[n][1] == "drug_row":
				drug_n = pd.DataFrame(drug1_chem.iloc[[row[n][0]]].copy(), columns=range(541))
			elif row[n][1] == "drug_col":
				drug_n = pd.DataFrame(drug2_chem.iloc[[row[n][0]]].copy(),  columns=range(541))
		else:
			if row[n][1] == "drug_row":
				drug_n = pd.concat([drug_n, drug1_chem.iloc[[row[n][0]]].copy()])
			elif row[n][1] == "drug_col":
				drug_n = pd.concat([drug_n, drug2_chem.iloc[[row[n][0]]].copy()])
		x += 1
	n = n.replace("_index", "")
	drug_n.to_csv(f"{n}_Chemopy_features.csv", index=False)
	
#Get CCLE features
def get_CCLE_features(Test):
	CCLE= pd.read_csv("../../Data/feature_data/CCL/CCL_feature_data.csv")
	x = 0
	for index, row in Test.iterrows():
		if x == 0:
			CCL = pd.DataFrame(CCLE.iloc[[row["D1_index"][0]]].copy())
		else:
			CCL = pd.concat([CCL, CCLE.iloc[[row["D1_index"][0]]].copy()])
		x += 1
	CCL.to_csv("CCLE_features.csv", index=False)

get_test_samples(main_df)
