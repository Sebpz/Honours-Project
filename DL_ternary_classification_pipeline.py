#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from tensorflow.keras import regularizers
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix

#Sampling
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

def one_hot_encode(label):
	lb = LabelBinarizer()
	dummy_y = lb.fit_transform(label)
	return dummy_y

def reverse_one_hot_encoding(label):
	lb = LabelBinarizer()
	multiclass = lb.inverse_transform(label)
	return multiclass

def sampling(feature_data, dataset_labels, name, method):
	#No Sampling
	if method == "No_Sampling":
		resampled_features = feature_data
		resampled_labels = dataset_labels
		resampling_method = "No_Sampling_"+name
		return 	resampled_features, resampled_labels, resampling_method

	#Oversamplings
	elif method == "Random_Oversampling":
		ros = RandomOverSampler(random_state=1)
		resampled_features, resampled_labels = ros.fit_resample(feature_data, dataset_labels)
		resampling_method = "Oversampling_"+name
		return 	resampled_features, resampled_labels, resampling_method

	#Undersampling
	elif method == "Random_Undersampling":
		rus = RandomUnderSampler(random_state=1)
		resampled_features, resampled_labels = rus.fit_resample(feature_data, dataset_labels)
		resampling_method = "Undersampling_"+name
		return 	resampled_features, resampled_labels, resampling_method
	
	#SMOTE
	elif method == "SMOTE":
		smote = SMOTE(random_state=1)
		resampled_features, resampled_labels = smote.fit_resample(feature_data, dataset_labels)
		resampling_method = "SMOTE_"+name
		return 	resampled_features, resampled_labels, resampling_method

	#SMOTEENN
	elif method == "SMOTEENN":
		smoteenn = SMOTEENN(random_state=1)
		resampled_features, resampled_labels = smoteenn.fit_resample(feature_data, dataset_labels)
		resampling_method = "SMOTEENN_"+name
		return 	resampled_features, resampled_labels, resampling_method


def CONIC_NN(X_train, y_train, X_test, y_test, X_val, y_val, Data, method):

	input_shape = X_train.shape[1]

	#3 class classifier
	inputs = keras.Input(shape=(input_shape,))
	x = layers.Dense(20000, activation="relu", kernel_initializer="he_normal")(inputs)
	x = layers.AlphaDropout(rate=0.2)(x)
	x = layers.Dense(8000, activation="relu", kernel_initializer="he_normal")(x)
	x = layers.AlphaDropout(rate=0.2)(x)
	x = layers.Dense(4000, activation="relu", kernel_initializer="he_normal")(x)
	x = layers.AlphaDropout(rate=0.2)(x)
	outputs = layers.Dense(3, activation="softmax")(x)
	model = keras.Model(inputs=inputs, outputs=outputs)
	model.summary()

	optimizer = keras.optimizers.Adam(lr=0.00001)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])


	#Train the model for 30 epoch from Numpy Data
	batch_size = 32
	history = model.fit(X_train, y_train, batch_size=batch_size, epochs=200, validation_data=(X_val, y_val))

	print("################################################")
	
	#Run model on test data
	test = model.evaluate(X_test, y_test)
	y_pred = model.predict(X_test)
	y_prob = model.predict(X_test)
	New_data = Data.append({'Classifier': method, "Accuracy": test[1], "ROC AUC": test[2],
								 "Recall": test[3], "Precision": test[4]}, ignore_index=True)

	#Binarise softmax output
	for i, pred in enumerate(y_pred):
		j = 0
		while j < 3:
			if max(y_pred[i,:]) == y_pred[i,j]:
				y_pred[i,j] = 1
			else:
				y_pred[i,j] = 0
			j += 1

	#reverse one_hot_encoding for confusion matrix
	y_pred = lb.inverse_transform(y_pred)
	y_test = lb.inverse_transform(y_test)

	return history, y_pred, y_prob, y_test, New_data

def results(history, y_pred, y_prob, y_test, label, Index_test, main_df):

	labels = ["Antagonistic", "Synergistic", "Additive"]

	#Produce table of prediction probabilities
	stacked = np.column_stack([y_prob, np.array(Index_test)])
	df = pd.DataFrame(stacked, columns =['Antagonistic Class Pred', 'Synergistic Class Pred', 'Additive Class Pred', 'New_index'])
	joined = main_df.merge(df, on='New_index', how='inner')
	joined = joined.sort_values('Synergistic Class Pred')
	joined.to_csv(f"model_results/{Model[1]}_{label}_predictions1.tsv", sep="\t", index=False)
	

	#Model Training History
	pd.DataFrame(history.history).plot(figsize=(8, 5))
	plt.title("Training Performance " + label)
	plt.xlabel("Epoch")
	plt.grid(True)
	plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
	plt.savefig("Results/images/3_class_classifiers/3_class_clf_new/Training_History_"+label)
	plt.clf()

	#Confusion Matrix
	cm = confusion_matrix(y_test, y_pred)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
	disp.plot()
	disp.ax_.set_title("Confusion matrix "+label)
	plt.tight_layout()
	plt.savefig("Results/images/3_class_classifiers/3_class_clf_new/Confusion matrix"+label, pad_inches=5)
	plt.clf()


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

#Adapted from: https://github.com/KristinaPreuer/DeepSynergy/blob/master/normalize.ipynb
def tanh_preprocessing(X, Mean=None, Std=None, filtered_feat=None):
	if Std is None:
		Std = np.std(X, axis=0)
	if filtered_feat is None:
		filtered_feat = Std!=0
	X = X[:,filtered_feat]
	X = np.ascontiguousarray(X)
	if Mean is None:
		Mean = np.mean(X, axis=0)
	X = (X-Mean)/Std[filtered_feat]
	return(np.tanh(X), Mean, Std, filtered_feat)


#Read Instructions from .dat file
Instructions = open(str(sys.argv[1]), "r")
Model = [line[:-1] for line in Instructions]
#print(Model)
Instructions.close()

#COLLECT LABELS
training_data = pd.read_csv(f"../../Data/Training_data/New_labels/tas_method/3_classes/{Model[0]}_IQR.tsv", sep="\t")
labels = training_data.class_numerical
indices = training_data.New_index
names = Model[1]


#LOAD FEATURE DATA
feature_data = grab_feature_data()


#Create results dataframe
columns = ["Classifier", "Accuracy", "ROC AUC", "Recall", "Precision"]
Results_Data = pd.DataFrame(columns=columns, dtype=object)

#OHE Labels
lb = LabelBinarizer()
dummy_y = lb.fit_transform(labels)

#TRAIN-TEST-SPLIT
X_train, X_test, Index_train, Index_test, y_train, y_test = train_test_split(feature_data, indices, dummy_y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=None)

#Preprocessing and Normalisation
X_train, Mean, Std, filtered_feat = tanh_preprocessing(X_train)
X_test, Mean, Std, filtered_feat = tanh_preprocessing(X_test, Mean, Std, filtered_feat)
X_val, Mean, Std, filtered_feat = tanh_preprocessing(X_val, Mean, Std, filtered_feat)

#Sampling
resampled_features, resampled_labels, resampled_methods = sampling(X_train, y_train, names, Model[2])
history, y_pred, y_prob, y_test_results, Results_Data = CONIC_NN(resampled_features, resampled_labels, X_test, y_test, X_val, y_val, Results_Data, resampled_methods)
Results_Data.to_csv(f"Results/tables/3_class_clf_new/3_class_conic_{Model[1]}_{Model[2]}_results.tsv", sep="\t", index=False)
main_df = training_data.copy()
results(history, y_pred, y_prob, y_test_results, resampled_methods, Index_test, main_df)
