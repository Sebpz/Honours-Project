import pandas as pd
import numpy as np
import sys
from os import listdir
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from tensorflow.keras import regularizers
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



def CONIC_NN_Regression(X_train, y_train, X_test, y_test, X_val, y_val, synthetic_samples, method):

	input_shape = X_train.shape[1]

	inputs = keras.Input(shape=(input_shape,))
	x = layers.Dense(8000, activation="relu", kernel_initializer="he_normal")(inputs)
	x = layers.AlphaDropout(rate=0.2)(x)
	x = layers.Dense(4000, activation="relu", kernel_initializer="he_normal")(x)
	x = layers.AlphaDropout(rate=0.2)(x)
	outputs = layers.Dense(1)(x)
	model = keras.Model(inputs=inputs, outputs=outputs)
	model.summary()

	optimizer = keras.optimizers.Adam(lr=0.00001)
	model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=["mean_squared_error", tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.CosineSimilarity()])
	#'''

	#Train the model for 30 epoch from Numpy Data
	batch_size = 32
	history = model.fit(X_train, y_train, batch_size=batch_size, epochs=1000, validation_data=(X_val, y_val))

	print("################################################")
	
	#Run model on test data
	test = model.evaluate(X_test, y_test)
	y_pred = model.predict(X_test)
	synth_pred = model.predict(synthetic_samples)
	
	columns = ["Classifier", "MSE", "RMSE", "CosineSimilarity"]
	Data = pd.DataFrame(columns=columns, dtype=object)
	Data = Data.append({'Classifier': method, "MSE": test[1], "RMSE": test[2], "CosineSimilarity": test[3]}, ignore_index=True)
	#print(Data)

	return history, y_pred, Data, synth_pred

def results(history, y_pred, y_test, label, synth_pred):
	print("Results")
	y_pred = y_pred.reshape(y_pred.shape[0],)
	y_test = np.array(y_test).reshape(y_test.shape[0],)
	
	print(pd.DataFrame(history.history))

	#Model Training History
	plt.plot(history.history["cosine_similarity"], color="maroon")
	plt.plot(history.history["val_cosine_similarity"], color="lightcoral")
	plt.title("Training Cosine Similarity Performance " + label)
	plt.xlabel("Epoch")
	plt.grid(True)
	plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
	plt.savefig(f"Results/images/Exploratory/Training_History_consinesim_{label}_{n}")
	plt.clf()

	plt.plot(history.history["mean_squared_error"], color="darkorange")
	plt.plot(history.history["val_mean_squared_error"], color="peachpuff")
	plt.title("Training MSE Performance " + label)
	plt.xlabel("Epoch")
	plt.grid(True)
	plt.savefig(f"Results/images/Exploratory/Training_History_mean_squared_error_{label}_{n}")
	plt.clf()

	plt.plot(history.history["root_mean_squared_error"], color="peru")
	plt.plot(history.history["val_root_mean_squared_error"], color="burlywood")
	plt.title("Training RMSE Performance " + label)
	plt.xlabel("Epoch")
	plt.grid(True)
	plt.savefig(f"Results/images/Exploratory/Training_History_root_mean_squared_error_{label}_{n}")
	plt.clf()

	plt.plot(history.history["loss"], color="midnightblue")
	plt.plot(history.history["val_loss"], color="cornflowerblue")
	plt.title("Training Loss Performance " + label)
	plt.xlabel("Epoch")
	plt.savefig(f"Results/images/Exploratory/Training_History_Loss_{label}_{n}")
	plt.clf()

	#Plot y_pred vs y_test
	# Calculating the parameters using the least square method
	#theta = np.polyfit(y_pred, y_test, 1)
	#y_line = theta[1] + theta[0] * y_pred
	plt.scatter(y_pred, y_test)
	#plt.plot(y_pred, y_line, 'r', label=f"The parameters of the line: {theta}")
	plt.title('True Loewe Synergy Scores vs Predicted by Neural Network')
	plt.xlabel('Predicted Values')
	plt.ylabel('True Values')
	R = np.corrcoef(y_pred, y_test)
	r = R[0][1]
	plt.text(-100,35,f'Pearson\'s Correlation Coefficient: {r}')
	plt.tight_layout()
	plt.savefig(f"Results/images/Exploratory/y_pred_vs_y_test_{n}")
	plt.clf()

	#Save synth_pred in df w labels
	df = pd.read_csv("../Create_Test_Data/Test_labels.tsv", sep="\t")
	df[f"pred_{n}"] = synth_pred
	df.to_csv(f"model_results/Explore/df_{n}", index=False)


#COLLECT FEATURE DATA
def grab_feature_data():
	drug_row = np.load("../LINCS_data_integration/drug_row_samples_aug_28_8.npy")
	drug1_chem = np.array(pd.read_csv("../../Data/Narjes_data/drug1_chem/28_8_drug1_chem.csv", header=None))
	drug_row = np.concatenate((drug_row.T, drug1_chem), axis=1)
	drug_col = np.load("../LINCS_data_integration/drug_col_samples_aug_28_8.npy")
	drug2_chem = np.array(pd.read_csv("../../Data/Narjes_data/drug2_chem/28_8_drug2_chem.csv", header=None))
	drug_col = np.concatenate((drug_col.T, drug2_chem), axis=1)
	feature_data = np.concatenate((drug_row, drug_col), axis=1)
	no_DU145 = pd.read_csv("../../Data/Training_data/CCLE_no_DU145.csv")
	feature_data = feature_data[no_DU145.Old_index,]
	CCLE = pd.read_csv("../../Data/feature_data/CCL/CCL_feature_data.csv")
	CCLE = np.array(CCLE.drop(columns=["DepMap_ID"]))
	feature_data = np.concatenate((feature_data, CCLE), axis=1)
	return feature_data

#COLLECT SYNTHETIC SAMPLES
def grab_synthetic_samples():
	D1_LINCS_features = np.load("../Create_Test_Data/D1_LINCS_features.npy")
	D2_LINCS_features = np.load("../Create_Test_Data/D2_LINCS_features.npy")
	D1_Chemopy_features = np.array(pd.read_csv("../Create_Test_Data/D1_Chemopy_features.csv"))
	D2_Chemopy_features = np.array(pd.read_csv("../Create_Test_Data/D2_Chemopy_features.csv"))
	D1 = np.concatenate((D1_LINCS_features, D1_Chemopy_features), axis=1)
	D2 = np.concatenate((D2_LINCS_features, D2_Chemopy_features), axis=1)
	synthetic_samples =np.concatenate((D1, D2), axis=1)
	CCLE_features = pd.read_csv("../Create_Test_Data/CCLE_features.csv")
	CCLE_features = np.array(CCLE_features.drop(columns=["DepMap_ID"]))
	synthetic_samples = np.concatenate((synthetic_samples, CCLE_features), axis=1)
	return synthetic_samples

#Adapted from: https://github.com/KristinaPreuer/DeepSynergy/blob/master/normalize.ipynb
def tanh_preprocessing(X, Mean=None, Std=None, filtered_feat=None):
	if Std is None:
		Std = np.std(X, axis=0)
	#'''
	if filtered_feat is None:
		filtered_feat = Std!=0
	X = X[:,filtered_feat]
	#'''
	X = np.ascontiguousarray(X)
	if Mean is None:
		Mean = np.mean(X, axis=0)
	X = (X-Mean)/Std[filtered_feat]
	return(np.tanh(X), Mean, Std, filtered_feat)

def check_synth_feature_similarity(synthetic_samples):
	print(synthetic_samples.shape)
	for i, x in enumerate(synthetic_samples):
		if i < synthetic_samples.shape[0]-1:
			print(np.linalg.norm(x-synthetic_samples[i+1]))


def check_n(n, arr):
	if n in arr:
		return check_n(random.randint(1, 50), arr)
	else:
		return n

#Array job Number:
n = sys.argv[1]
pred_files = listdir("model_results/Explore")
pred_files = [word.replace("df_", "") for word in pred_files if 'df' in word]
n = check_n(n, pred_files)




#LOAD SYNERGY SCORES
training_data = pd.read_csv("../../Data/Training_data/New_labels/tas_method/3_classes/synergy_loewe_IQR.tsv", sep="\t")
labels = training_data["synergy_loewe"]
indices = training_data.New_index

#LOAD FEATURE DATA
feature_data = grab_feature_data()

#TRAIN-TEST-SPLIT - create train + test sets and do preprocessing
X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.2, random_state=None)



#LOAD AND PREPROCESS SYNTHETIC SAMPLES
synthetic_samples = grab_synthetic_samples()

####CHECKING SYNTH SAMPLES AREN'T COPIES
#check_synth_feature_similarity(synthetic_samples)


#TRAIN-TEST-SPLIT - create train + val sets from train and do preprocessing
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=None)
X_train, Mean, Std, filtered_feat = tanh_preprocessing(X_train)
X_test, Mean, Std, filtered_feat = tanh_preprocessing(X_test, Mean, Std, filtered_feat)
X_val, Mean, Std, filtered_feat = tanh_preprocessing(X_val, Mean, Std, filtered_feat)
synthetic_samples, Mean, Std, filtered_feat = tanh_preprocessing(synthetic_samples, Mean, Std, filtered_feat)
check_synth_feature_similarity(X_test)
check_synth_feature_similarity(X_val)
check_synth_feature_similarity(synthetic_samples)


history, y_pred, Data, synth_pred = CONIC_NN_Regression(X_train, y_train, X_test, y_test, X_val, y_val, synthetic_samples, f"Loewe_regression_{n}")
Data.to_csv(f"Results/tables/Exploratory/Regression_model_{n}.tsv", sep="\t", index=False)
results(history, y_pred, y_test, "Loewe_regression", synth_pred)
