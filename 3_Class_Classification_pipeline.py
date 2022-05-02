#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import XGBClassifier

#model-selection
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import time

#metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

#Sampling
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

#preprocessing
from sklearn.preprocessing import LabelBinarizer

#Dimensionality Reduction/Feature Selection
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import phate
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#################################Classfication Learniners#################################
def one_hot_encode(label):
	lb = LabelBinarizer()
	dummy_y = lb.fit_transform(label)
	return dummy_y

def RBF_correction(arr):
	arr = np.concatenate((arr, np.array([0,1,2])))
	return arr

def random_cv(param_grid, estimator, X, y, method):
	print(f"{method} randomised cross-validation:")
	random = RandomizedSearchCV(estimator=estimator, n_iter=4,
								param_distributions=param_grid, 
								cv = 5, n_jobs=-1, scoring='f1_macro')
	start_time = time.time()
	random_result = random.fit(X, y)
	# Summarize results
	print("Best %s: %f using %s" % ('f1_macro', random_result.best_score_, random_result.best_params_))
	print("Execution time: " + str((time.time() - start_time)) + ' ms')
	return_model = random_result.best_estimator_
	return return_model

def results(classifier, y_pred, y_test):
	print(f"{classifier} Results:\n", "roc_auc_score: ", roc_auc_score(y_test, y_pred),"\n",
			"accuracy_score: ", accuracy_score(y_test, y_pred),"\n",
			"f1_score: ", f1_score(y_test, y_pred),"\n",
			"precision_score: ", precision_score(y_test, y_pred),"\n",
			"recall_score: ", recall_score(y_test, y_pred),"\n",
			"matthews_corrcoef: ", matthews_corrcoef(y_test, y_pred),"\n",)

def results_data_update(y_pred_OHE, y_test_OHE):
	roc_auc = roc_auc_score(y_pred_OHE, y_test_OHE, multi_class='ovr')
	accuracy = accuracy_score(y_test_OHE, y_pred_OHE)
	f1 = f1_score(y_pred_OHE, y_test_OHE, average='macro')
	precision = precision_score(y_pred_OHE, y_test_OHE, average='macro')
	recall = recall_score(y_pred_OHE, y_test_OHE, average='macro')
	#mcc = matthews_corrcoef(y_test, y_pred)
	return roc_auc, accuracy, f1, precision, recall

def graphical_results(y_test, y_pred, label, method):
	labels = ["Antagonistic", "Synergistic", "Additive"]
	cm = confusion_matrix(y_test, y_pred)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
	disp.plot()
	disp.ax_.set_title("Confusion matrix "+label+" "+method)
	plt.tight_layout()
	plt.savefig("new_results/3_class_clf/confusion_matricies/Confusion matrix"+label+" "+method, pad_inches=5)
	plt.clf()

	
#SVC Polynomial
def SVC_Poly(X_train, X_test, y_train, y_test, Data, label):
	method = "SVM: Polynomial Kernel"
	coef0 = [0.001, 0.01, 0.1, 1, 10, 100]
	C = [ 0.1, 1, 10, 100, 1000]
	param_grid = dict(coef0=coef0, C=C)
	poly_svm_clf = SVC(kernel="poly", degree=3, decision_function_shape='ovo')
	poly_svm_clf = random_cv(param_grid, poly_svm_clf, X_train, y_train, method)
	y_pred = poly_svm_clf.predict(X_test)
	graphical_results(y_test, y_pred, label, method)
	y_pred_OHE = one_hot_encode(RBF_correction(y_pred))
	y_test_OHE = one_hot_encode(RBF_correction(y_test))
	roc_auc, acc, f1, prec, rec = results_data_update(y_pred_OHE, y_test_OHE)
	New_Data = Data.append({'Classifier': method, "ROC AUC": roc_auc, "Accuracy": acc,
								 "F1": f1, "Precision": prec, "Recall": rec}, ignore_index=True)
	return poly_svm_clf, New_Data

#SVC RBF
def RBF(X_train, X_test, y_train, y_test, Data, label):
	method = "SVM: Radial Basis Function"
	gamma = [0.001, 0.01]
	C = [0.001, 0.01, 0.1, 1]
	param_grid = dict(gamma=gamma, C=C)
	SVC_RBF = SVC(decision_function_shape='ovo')
	SVC_RBF = random_cv(param_grid, SVC_RBF, X_train, y_train, method)
	y_pred = SVC_RBF.predict(X_test)
	graphical_results(y_test, y_pred, label, method)
	y_pred_OHE = one_hot_encode(RBF_correction(y_pred))
	y_test_OHE = one_hot_encode(RBF_correction(y_test))
	roc_auc, acc, f1, prec, rec = results_data_update(y_pred_OHE, y_test_OHE)
	New_Data = Data.append({'Classifier': method, "ROC AUC": roc_auc, "Accuracy": acc,
								 "F1": f1, "Precision": prec, "Recall": rec}, ignore_index=True)
	return SVC_RBF, New_Data

#Random forest
def RFC(X_train, X_test, y_train, y_test, Data, label):
	method = "Random forest"
	n_estimators = [3, 10, 40, 100, 300, 500]
	param_grid = dict(n_estimators=n_estimators)
	rnd_clf = RandomForestClassifier()
	rnd_clf = random_cv(param_grid, rnd_clf, X_train, y_train, method)
	y_pred = rnd_clf.predict(X_test)
	graphical_results(y_test, y_pred, label, method)
	y_pred_OHE = one_hot_encode(RBF_correction(y_pred))
	y_test_OHE = one_hot_encode(RBF_correction(y_test))
	roc_auc, acc, f1, prec, rec = results_data_update(y_pred_OHE, y_test_OHE)
	New_Data = Data.append({'Classifier': method, "ROC AUC": roc_auc, "Accuracy": acc,
								 "F1": f1, "Precision": prec, "Recall": rec}, ignore_index=True)
	return rnd_clf, New_Data

#AdaBoost classifier
def AdaBoost(X_train, X_test, y_train, y_test, Data, label):
	method = "AdaBoost classifier"
	n_estimators = [50, 300, 700, 1000, 1500, 3000]
	param_grid = dict(n_estimators=n_estimators)
	AdaBoost_clf = AdaBoostClassifier()
	AdaBoost_clf = random_cv(param_grid, AdaBoost_clf, X_train, y_train, method)
	y_pred = AdaBoost_clf.predict(X_test)
	graphical_results(y_test, y_pred, label, method)
	y_pred_OHE = one_hot_encode(y_pred)
	y_test_OHE = one_hot_encode(y_test)
	roc_auc, acc, f1, prec, rec = results_data_update(y_pred_OHE, y_test_OHE)
	New_Data = Data.append({'Classifier': method, "ROC AUC": roc_auc, "Accuracy": acc,
								 "F1": f1, "Precision": prec, "Recall": rec}, ignore_index=True)
	return AdaBoost_clf, New_Data

#XGBoost 
def XGBoost(X_train, X_test, y_train, y_test, Data, label):
	method = "XGBoost"
	xgb = XGBClassifier()
	xgb.fit(X_train, y_train)
	y_pred = xgb.predict(X_test) 
	y_pred  = [round(value) for value in y_pred]
	graphical_results(y_test, y_pred, label, method)	
	y_test_OHE  = one_hot_encode(y_test)
	y_pred_OHE = one_hot_encode(y_pred)	
	roc_auc, acc, f1, prec, rec = results_data_update(y_pred_OHE, y_test_OHE)
	New_Data = Data.append({'Classifier': method, "ROC AUC": roc_auc, "Accuracy": acc,
								 "F1": f1, "Precision": prec, "Recall": rec}, ignore_index=True)
	return xgb, New_Data

#K-nearest neighbours
def KNN(X_train, X_test, y_train, y_test, Data, label):
	method = "K-nearest neighbours"
	n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8]
	param_grid = dict(n_neighbors=n_neighbors)
	KNN_clf = KNeighborsClassifier()
	KNN_clf = random_cv(param_grid, KNN_clf, X_train, y_train, method)
	y_pred = KNN_clf.predict(X_test)
	graphical_results(y_test, y_pred, label, method)
	y_pred_OHE = one_hot_encode(y_pred)
	y_test_OHE = one_hot_encode(y_test)
	roc_auc, acc, f1, prec, rec = results_data_update(y_pred_OHE, y_test_OHE)
	New_Data = Data.append({'Classifier': method, "ROC AUC": roc_auc, "Accuracy": acc,
								 "F1": f1, "Precision": prec, "Recall": rec}, ignore_index=True)
	return KNN_clf, New_Data

#Gaussian Naive Bayes
def GNB(X_train, X_test, y_train, y_test, Data, label):
	method = "Gaussian Naive Bayes Classifier"
	GNB_clf = GaussianNB()
	GNB_clf = GNB_clf.fit(X_train, y_train)
	y_pred = GNB_clf.predict(X_test)
	graphical_results(y_test, y_pred, label, method)
	y_pred_OHE = one_hot_encode(y_pred)
	y_test_OHE = one_hot_encode(y_test)
	roc_auc, acc, f1, prec, rec = results_data_update(y_pred_OHE, y_test_OHE)
	New_Data = Data.append({'Classifier': method, "ROC AUC": roc_auc, "Accuracy": acc,
								 "F1": f1, "Precision": prec, "Recall": rec}, ignore_index=True)
	return GNB_clf, New_Data

#Decision Tree Classifier
def Tree(X_train, X_test, y_train, y_test, Data, label):
	method = "Decision Tree Classifier"
	tree_clf = DecisionTreeClassifier()
	tree_clf = tree_clf.fit(X_train, y_train)
	y_pred = tree_clf.predict(X_test)
	graphical_results(y_test, y_pred, label, method)
	y_pred_OHE = one_hot_encode(y_pred)
	y_test_OHE = one_hot_encode(y_test)
	roc_auc, acc, f1, prec, rec = results_data_update(y_pred_OHE, y_test_OHE)
	New_Data = Data.append({'Classifier': method, "ROC AUC": roc_auc, "Accuracy": acc,
								 "F1": f1, "Precision": prec, "Recall": rec}, ignore_index=True)
	return tree_clf, New_Data

#Bagging Classifier
def Bagging(X_train, X_test, y_train, y_test, Classifiers, Data, label):
	method = "Bagging Classifier"
	n_estimators = [2, 4, 6,]
	param_grid = dict(n_estimators=n_estimators, base_estimator=Classifiers)
	Bagging_clf = BaggingClassifier()
	Bagging_clf = random_cv(param_grid, Bagging_clf, X_train, y_train, method)
	y_pred = Bagging_clf.predict(X_test)
	graphical_results(y_test, y_pred, label, method)
	y_pred_OHE = one_hot_encode(RBF_correction(y_pred))
	y_test_OHE = one_hot_encode(RBF_correction(y_test))
	roc_auc, acc, f1, prec, rec = results_data_update(y_pred_OHE, y_test_OHE)
	New_Data = Data.append({'Classifier': method, "ROC AUC": roc_auc, "Accuracy": acc,
								 "F1": f1, "Precision": prec, "Recall": rec}, ignore_index=True)
	return Bagging_clf, New_Data



#Random Over-Sampling
def Ros(X, y):
	ros = RandomOverSampler(random_state=1)
	X_resampled, y_resampled = ros.fit_resample(X, y)
	return X_resampled, y_resampled

#Random Under-Sampling
def Rus(X, y):
	rus = RandomUnderSampler(random_state=1)
	X_resampled, y_resampled = rus.fit_resample(X, y)
	return X_resampled, y_resampled
	
#Synthetic Minority Oversampling TEchnique (SMOTE)
def Smote(X, y):
	smote = SMOTE(random_state=1)
	X_resampled, y_resampled = smote.fit_resample(X, y)
	return X_resampled, y_resampled
	
#SMOTE + Edited Nearest Neighbours (SMOTEENN)
def Smoteen(X, y):
	smoteenn = SMOTEENN(random_state=1)
	X_resampled, y_resampled = smoteenn.fit_resample(X, y)
	return X_resampled, y_resampled




#Run estimators on different embedded datasets
def compare_learners(X_train, X_test, y_train, y_test, Name):
	columns = ["Classifier", "ROC AUC", "Accuracy", "F1", "Precision", "Recall"]
	resampling_method = ""
	if Model[2] == "Random_Oversampling":
		resampling_method = Model[2]
		X_train, y_train = Ros(X_train, y_train)
	elif Model[2] == "Random_Undersampling":
		resampling_method = Model[2]
		X_train, y_train = Rus(X_train, y_train)
	elif Model[2] == "SMOTE":
		resampling_method = Model[2]
		X_train, y_train = Smote(X_train, y_train)
	elif Model[2] == "SMOTEENN":
		resampling_method = Model[2]
		X_train, y_train = Smoteen(X_train, y_train)
	elif Model[2] == "No_Sampling":
		resampling_method = Model[2]

	label = Name+"_"+resampling_method

	print(f"####################################{label}########################################")
	Results_Data = pd.DataFrame(columns=columns, dtype=object)
	Classifiers = []
	poly_svm_clf, Results_Data = SVC_Poly(X_train, X_test, y_train, y_test, Results_Data, label)
	Classifiers.append(poly_svm_clf)
	SVC_RBF, Results_Data = RBF(X_train, X_test, y_train, y_test, Results_Data, label)
	Classifiers.append(SVC_RBF)
	rnd_clf, Results_Data = RFC(X_train, X_test, y_train, y_test, Results_Data, label)
	Classifiers.append(rnd_clf)
	AdaBoost_clf, Results_Data = AdaBoost(X_train, X_test, y_train, y_test, Results_Data, label)
	Classifiers.append(AdaBoost_clf)
	XGB, Results_Data = XGBoost(X_train, X_test, y_train, y_test, Results_Data, label)
	Classifiers.append(XGB)
	KNN_clf, Results_Data = KNN(X_train, X_test, y_train, y_test, Results_Data, label)
	Classifiers.append(KNN_clf)
	GNB_clf, Results_Data  = GNB(X_train, X_test, y_train, y_test, Results_Data, label)
	Classifiers.append(GNB_clf)
	tree_clf, Results_Data  = Tree(X_train, X_test, y_train, y_test, Results_Data, label)
	Classifiers.append(tree_clf)
	Bagging_clf, Results_Data  = Bagging(X_train, X_test, y_train, y_test, Classifiers, Results_Data, label)
	Results_Data.to_csv(f"new_results/3_class_clf/tables/{label}_results.tsv", sep="\t", index=False)
	print("#####################################################################################")

def remove_DU145(a):
	no_DU145 = pd.read_csv("../../Data/Training_data/CCLE_no_DU145.csv")
	a = a[no_DU145.Old_index,]
	return a

def perform_PCA(train, test, n):
	pca = PCA(n_components=n)
	pca.fit(train)
	train = pca.transform(train)
	test = pca.transform(test)
	return train, test

def perform_phate(train, test, n):
	phate_op = phate.PHATE(n_components=n)
	phate_op.fit(train)
	train = phate_op.transform(train)
	test = phate_op.transform(test)
	return train, test

def perform_tSNE(train, test, n):
	tsne = TSNE(n_components=n)
	tsne.fit(train)
	train = tsne.transform(train)
	test = tsne.transform(train)
	return train, test

def perform_AE(train, test, input_shape, n):
	og_train = train
	#create validation set
	train, val = train_test_split(train, test_size=0.1, random_state=42)
	
	#Build a simple AE
	inputs = keras.Input(shape=(input_shape,))
	x = layers.Dense(2000, activation="relu")(inputs)
	encoded = layers.Dense(n, activation="relu")(x)
	x = layers.Dense(2000, activation="relu")(encoded)
	outputs = layers.Dense(input_shape,)(x) 
	autoencoder = keras.Model(inputs=inputs, outputs=outputs)
	encoder = keras.Model(inputs=inputs, outputs=encoded)
	autoencoder.summary()

	#Compile AE
	autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])#tf.keras.losses.MeanSquaredError())

	#Train the model
	batch_size = 32
	history = autoencoder.fit(train, train, batch_size=batch_size, epochs=30, validation_data=(val, val), shuffle=True)

	#Make Embeddings
	train = encoder.predict(og_train)
	test = encoder.predict(test)
	return train, test

#Performs a appropriated embedding, using a given function 'fn', on each feature dataset independently
def perform_embeddings(fn, fn_type, drug_row_train, drug_row_test, drug_col_train, drug_col_test, drug1_chem_train, drug1_chem_test, drug2_chem_train, drug2_chem_test, CCLE_train, CCLE_test):
	#AutoEncoder requires explicit declaration of feature space input size
	if fn_type == 1:
		drug_row_train, drug_row_test = fn(drug_row_train, drug_row_test, 12328, 175)
		drug_col_train, drug_col_test = fn(drug_col_train, drug_col_test, 12328, 175)
		drug1_chem_train, drug1_chem_test = fn(drug1_chem_train, drug1_chem_test, 541, 25)
		drug2_chem_train, drug2_chem_test = fn(drug2_chem_train, drug2_chem_test, 541, 25)
		CCLE_train, CCLE_test = fn(CCLE_train, CCLE_test, 19177, 100)
	elif fn_type == 2:
		drug_row_train, drug_row_test = fn(drug_row_train, drug_row_test, y_train, 100)
		drug_col_train, drug_col_test = fn(drug_col_train, drug_col_test, y_train, 100)
		drug1_chem_train, drug1_chem_test = fn(drug1_chem_train, drug1_chem_test, y_train, 20)
		drug2_chem_train, drug2_chem_test = fn(drug2_chem_train, drug2_chem_test, y_train, 20)
		CCLE_train, CCLE_test = fn(CCLE_train, CCLE_test, y_train, 50)
	elif fn_type == 3:
		drug_row_train, drug_row_test = fn(drug_row_train, drug_row_test, 26)
		drug_col_train, drug_col_test = fn(drug_col_train, drug_col_test, 26)
		drug1_chem_train, drug1_chem_test = fn(drug1_chem_train, drug1_chem_test, 7)
		drug2_chem_train, drug2_chem_test = fn(drug2_chem_train, drug2_chem_test, 7)
		CCLE_train, CCLE_test = fn(CCLE_train, CCLE_test, 13)
	else:
		drug_row_train, drug_row_test = fn(drug_row_train, drug_row_test, 50)
		drug_col_train, drug_col_test = fn(drug_col_train, drug_col_test, 50)
		drug1_chem_train, drug1_chem_test = fn(drug1_chem_train, drug1_chem_test, 15)
		drug2_chem_train, drug2_chem_test = fn(drug2_chem_train, drug2_chem_test, 15)
		CCLE_train, CCLE_test = fn(CCLE_train, CCLE_test, 50)
	#Embedded features joined 
	X_train = join_embeddings(drug_row_train, drug_col_train, drug1_chem_train, drug2_chem_train, CCLE_train)
	x_test = join_embeddings(drug_row_test, drug_col_test, drug1_chem_test, drug2_chem_test, CCLE_test)

	return X_train, x_test

#Joins embedded featuredatasets together
def join_embeddings(drug_row, drug_col, drug1_chem, drug2_chem, CCLE):
	drug_row = np.concatenate((drug_row, drug1_chem), axis=1)
	drug_col = np.concatenate((drug_col, drug2_chem), axis=1)
	feature_data = np.concatenate((drug_row, drug_col), axis=1)
	feature_data = np.concatenate((feature_data, CCLE), axis=1) 
	return feature_data


#Collect Array Job
Instructions = open(str(sys.argv[1]), "r")
Model = [line[:-1] for line in Instructions]
Instructions.close()

#Load Labels Data
data = pd.read_csv("../../Data/Training_data/New_labels/tas_method/3_classes/"+Model[0], sep="\t")
name = Model[0].replace("_IQR.tsv", "")
Labels = data.class_numerical

#COLLECT FEATURE DATA
drug_row = remove_DU145(np.load("../LINCS_data_integration/drug_row_samples_aug_28_8.npy").T)
drug1_chem = remove_DU145(np.array(pd.read_csv("../../Data/Narjes_data/drug1_chem/28_8_drug1_chem.csv", header=None)))
drug_col = remove_DU145(np.load("../LINCS_data_integration/drug_col_samples_aug_28_8.npy").T)
drug2_chem = remove_DU145(np.array(pd.read_csv("../../Data/Narjes_data/drug2_chem/28_8_drug2_chem.csv", header=None)))
CCLE = pd.read_csv("../../Data/feature_data/CCL/CCL_feature_data.csv")
CCLE = np.array(CCLE.drop(columns=["DepMap_ID"]))


#Perform Train-Test-Split on Labels and All Feature Datasets
drug_row_train, drug_row_test, drug_col_train, drug_col_test, drug1_chem_train, drug1_chem_test, drug2_chem_train, drug2_chem_test, CCLE_train, CCLE_test, y_train, y_test = train_test_split(drug_row, drug_col, drug1_chem, drug2_chem, CCLE, Labels, test_size=0.2, random_state=42)

#According to Array job specifications perform Dimensionality reduction technique to create embedded feature data
if Model[1] == "AE":
	X_train, x_test = perform_embeddings(perform_AE, 1, drug_row_train, drug_row_test, drug_col_train, drug_col_test, drug1_chem_train, drug1_chem_test, drug2_chem_train, drug2_chem_test, CCLE_train, CCLE_test)
	Name = "AE_"+name
elif Model[1] == "PCA":
	X_train, x_test = perform_embeddings(perform_PCA, 0, drug_row_train, drug_row_test, drug_col_train, drug_col_test, drug1_chem_train, drug1_chem_test, drug2_chem_train, drug2_chem_test, CCLE_train, CCLE_test)
	Name = "PCA_"+name
elif Model[1] == "CHI2":
	X_train, x_test = perform_CHI2(perform_phate, 2, drug_row_train, drug_row_test, drug_col_train, drug_col_test, drug1_chem_train, drug1_chem_test, drug2_chem_train, drug2_chem_test, CCLE_train, CCLE_test)
	Name = "CHI2_"+name
elif Model[1] == "phate":
	X_train, x_test = perform_embeddings(perform_phate, 3, drug_row_train, drug_row_test, drug_col_train, drug_col_test, drug1_chem_train, drug1_chem_test, drug2_chem_train, drug2_chem_test, CCLE_train, CCLE_test)
	Name = "phate_"+name
	
compare_learners(X_train, x_test, y_train, y_test, Name)
