'''
Investigate the relative importance of each feature set. Will be inferred by respective model performance.
To save time only one model (and one set of training labels) will be used to test feature importance in the classification task.
Furthermore, only combinations of features that would intuitively be able to predict synergism explored:
	1. Just the LINCS Monotherapy Data for each Drug
	2. ChemoPy data for each drug + CCLE data
	3. LINCS data for each drug + CCLE data
'''

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


def Binarise(y_pred):
	#Binarise softmax output
	for i, pred in enumerate(y_pred):
		j = 0
		while j < 3:
			if max(y_pred[i,:]) == y_pred[i,j]:
				y_pred[i,j] = 1
			else:
				y_pred[i,j] = 0
			j += 1
	return y_pred

def LINCS(X_train, y_train, X_test, y_test, X_val, y_val, Data, method):
	No_Of_L1000_rows = 12328
	input_shape = 2*No_Of_L1000_rows

	#3 class classifier
	inputs = keras.Input(shape=(input_shape,))
	x = layers.Dense(8000, activation="relu", kernel_initializer="he_normal")(inputs)
	x = layers.AlphaDropout(rate=0.2)(x)
	x = layers.Dense(3000, activation="relu", kernel_initializer="he_normal")(x)
	x = layers.AlphaDropout(rate=0.2)(x)
	outputs = layers.Dense(3, activation="softmax")(x)
	model = keras.Model(inputs=inputs, outputs=outputs)
	model.summary()

	optimizer = keras.optimizers.Adam(lr=0.00001)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
	#'''

	#Train the model for 30 epoch from Numpy Data
	batch_size = 32
	history = model.fit(X_train, y_train, batch_size=batch_size, epochs=75, validation_data=(X_val, y_val))

	print("################################################")
	
	#Run model on test data
	test = model.evaluate(X_test, y_test)
	y_pred = model.predict(X_test)
	y_prob = model.predict(X_test)
	New_data = Data.append({'Classifier': method, "Accuracy": test[1], "ROC AUC": test[2],
								 "Recall": test[3], "Precision": test[4]}, ignore_index=True)

	y_pred = Binarise(y_pred)

	#reverse one_hot_encoding for confusion matrix
	y_pred = lb.inverse_transform(y_pred)
	y_test = lb.inverse_transform(y_test)

	return history, y_pred, y_prob, y_test, New_data

def ChemoPy_CCLE(X_train, y_train, X_test, y_test, X_val, y_val, Data, method):
	No_Of_drug_features = 541
	No_Of_CCLE_features = 19177
	input_shape = 2*No_Of_drug_features + No_Of_CCLE_features

	#3 class classifier
	inputs = keras.Input(shape=(input_shape,))
	x = layers.Dense(8000, activation="relu", kernel_initializer="he_normal")(inputs)
	x = layers.AlphaDropout(rate=0.2)(x)
	x = layers.Dense(3000, activation="relu", kernel_initializer="he_normal")(x)
	x = layers.AlphaDropout(rate=0.2)(x)
	outputs = layers.Dense(3, activation="softmax")(x)
	model = keras.Model(inputs=inputs, outputs=outputs)
	model.summary()

	optimizer = keras.optimizers.Adam(lr=0.00001)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
	#'''

	#Train the model for 30 epoch from Numpy Data
	batch_size = 32
	history = model.fit(X_train, y_train, batch_size=batch_size, epochs=75, validation_data=(X_val, y_val))

	print("################################################")
	
	#Run model on test data
	test = model.evaluate(X_test, y_test)
	y_pred = model.predict(X_test)
	y_prob = model.predict(X_test)
	New_data = Data.append({'Classifier': method, "Accuracy": test[1], "ROC AUC": test[2],
								 "Recall": test[3], "Precision": test[4]}, ignore_index=True)
	
	y_pred = Binarise(y_pred)

	#reverse one_hot_encoding for confusion matrix
	y_pred = lb.inverse_transform(y_pred)
	y_test = lb.inverse_transform(y_test)

	return history, y_pred, y_prob, y_test, New_data

def LINCS_CCLE(X_train, y_train, X_test, y_test, X_val, y_val, Data, method):
	No_Of_L1000_rows = 12328
	No_Of_CCLE_features = 19177
	input_shape = 2*No_Of_L1000_rows + No_Of_CCLE_features

	#3 class classifier
	inputs = keras.Input(shape=(input_shape,))
	x = layers.Dense(8000, activation="relu", kernel_initializer="he_normal")(inputs)
	x = layers.AlphaDropout(rate=0.2)(x)
	x = layers.Dense(3000, activation="relu", kernel_initializer="he_normal")(x)
	x = layers.AlphaDropout(rate=0.2)(x)
	outputs = layers.Dense(3, activation="softmax")(x)
	model = keras.Model(inputs=inputs, outputs=outputs)
	model.summary()

	optimizer = keras.optimizers.Adam(lr=0.00001)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
	#'''

	#Train the model for 30 epoch from Numpy Data
	batch_size = 32
	history = model.fit(X_train, y_train, batch_size=batch_size, epochs=75, validation_data=(X_val, y_val))

	print("################################################")
	
	#Run model on test data
	test = model.evaluate(X_test, y_test)
	y_pred = model.predict(X_test)
	y_prob = model.predict(X_test)
	New_data = Data.append({'Classifier': method, "Accuracy": test[1], "ROC AUC": test[2],
								 "Recall": test[3], "Precision": test[4]}, ignore_index=True)
	
	y_pred = Binarise(y_pred)

	#reverse one_hot_encoding for confusion matrix
	y_pred = lb.inverse_transform(y_pred)
	y_test = lb.inverse_transform(y_test)

	return history, y_pred, y_prob, y_test, New_data

def LINCS_ChemoPy(X_train, y_train, X_test, y_test, X_val, y_val, Data, method):
	No_Of_L1000_rows = 12328
	No_Of_drug_features = 541
	input_shape = 2*(No_Of_L1000_rows+No_Of_drug_features)

	#3 class classifier
	inputs = keras.Input(shape=(input_shape,))
	x = layers.Dense(8000, activation="relu", kernel_initializer="he_normal")(inputs)
	x = layers.AlphaDropout(rate=0.2)(x)
	x = layers.Dense(3000, activation="relu", kernel_initializer="he_normal")(x)
	x = layers.AlphaDropout(rate=0.2)(x)
	outputs = layers.Dense(3, activation="softmax")(x)
	model = keras.Model(inputs=inputs, outputs=outputs)
	model.summary()

	optimizer = keras.optimizers.Adam(lr=0.00001)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
	#'''

	#Train the model for 30 epoch from Numpy Data
	batch_size = 32
	history = model.fit(X_train, y_train, batch_size=batch_size, epochs=75, validation_data=(X_val, y_val))

	print("################################################")
	
	#Run model on test data
	test = model.evaluate(X_test, y_test)
	y_pred = model.predict(X_test)
	y_prob = model.predict(X_test)
	New_data = Data.append({'Classifier': method, "Accuracy": test[1], "ROC AUC": test[2],
								 "Recall": test[3], "Precision": test[4]}, ignore_index=True)
	
	y_pred = Binarise(y_pred)

	#reverse one_hot_encoding for confusion matrix
	y_pred = lb.inverse_transform(y_pred)
	y_test = lb.inverse_transform(y_test)

	return history, y_pred, y_prob, y_test, New_data


def LINCS_ChemoPy_CCLE(X_train, y_train, X_test, y_test, X_val, y_val, Data, method):
	No_Of_L1000_rows = 12328
	No_Of_drug_features = 541
	No_Of_CCLE_features = 19177
	input_shape = 2*(No_Of_L1000_rows+No_Of_drug_features) + No_Of_CCLE_features

	#3 class classifier
	inputs = keras.Input(shape=(input_shape,))
	x = layers.Dense(8000, activation="relu", kernel_initializer="he_normal")(inputs)
	x = layers.AlphaDropout(rate=0.2)(x)
	x = layers.Dense(3000, activation="relu", kernel_initializer="he_normal")(x)
	x = layers.AlphaDropout(rate=0.2)(x)
	outputs = layers.Dense(3, activation="softmax")(x)
	model = keras.Model(inputs=inputs, outputs=outputs)
	model.summary()

	optimizer = keras.optimizers.Adam(lr=0.00001)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
	#'''

	#Train the model for 30 epoch from Numpy Data
	batch_size = 32
	history = model.fit(X_train, y_train, batch_size=batch_size, epochs=75, validation_data=(X_val, y_val))

	print("################################################")
	
	#Run model on test data
	test = model.evaluate(X_test, y_test)
	y_pred = model.predict(X_test)
	y_prob = model.predict(X_test)
	New_data = Data.append({'Classifier': method, "Accuracy": test[1], "ROC AUC": test[2],
								 "Recall": test[3], "Precision": test[4]}, ignore_index=True)
	
	y_pred = Binarise(y_pred)

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
	joined.to_csv(f"model_results/feature_importance/{array_job}_predictions.tsv", sep="\t", index=False)
	

	#Model Training History
	pd.DataFrame(history.history).plot(figsize=(8, 5))
	plt.title("Training Performance " + label)
	plt.xlabel("Epoch")
	plt.grid(True)
	plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
	plt.savefig("Results/images/feature_importance/Training_History_"+label+array_job)
	plt.clf()

	#Confusion Matrix
	cm = confusion_matrix(y_test, y_pred)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels+array_job)
	disp.plot()
	disp.ax_.set_title("Confusion matrix "+label)
	plt.tight_layout()
	plt.savefig("Results/images/feature_importance/Confusion matrix"+label+array_job, pad_inches=5)
	plt.clf()


#Read Instructions from commandline
Instructions = int(sys.argv[1])%4
array_job = int(sys.argv[1])
if int(sys.argv[1]) > 60:
	Instructions = 4
print(Instructions)

#Load relevant feature datasets per commandline instruction

if Instructions == 0:
	print("0")
	drug_row = np.load("../LINCS_data_integration/drug_row_samples_aug_28_8.npy").T
	drug_col = np.load("../LINCS_data_integration/drug_col_samples_aug_28_8.npy").T
	feature_data = np.concatenate((drug_row, drug_col), axis=1)
	no_DU145 = pd.read_csv("../../Data/Training_data/CCLE_no_DU145.csv")
	feature_data = feature_data[no_DU145.Old_index,]
elif Instructions == 1:
	print("1")
	drug1_chem = np.array(pd.read_csv("../../Data/Narjes_data/drug1_chem/28_8_drug1_chem.csv", header=None))
	drug2_chem = np.array(pd.read_csv("../../Data/Narjes_data/drug2_chem/28_8_drug2_chem.csv", header=None))
	feature_data = np.concatenate((drug1_chem, drug2_chem), axis=1)
	no_DU145 = pd.read_csv("../../Data/Training_data/CCLE_no_DU145.csv")
	feature_data = feature_data[no_DU145.Old_index,]
	CCLE = pd.read_csv("../../Data/feature_data/CCL/CCL_feature_data.csv")
	CCLE = np.array(CCLE.drop(columns=["DepMap_ID"]))
	feature_data = np.concatenate((feature_data, CCLE), axis=1)
elif Instructions == 2:
	print("2")
	drug_row = np.load("../LINCS_data_integration/drug_row_samples_aug_28_8.npy").T
	drug_col = np.load("../LINCS_data_integration/drug_col_samples_aug_28_8.npy").T
	feature_data = np.concatenate((drug_row, drug_col), axis=1)
	no_DU145 = pd.read_csv("../../Data/Training_data/CCLE_no_DU145.csv")
	feature_data = feature_data[no_DU145.Old_index,]
	CCLE = pd.read_csv("../../Data/feature_data/CCL/CCL_feature_data.csv")
	CCLE = np.array(CCLE.drop(columns=["DepMap_ID"]))
	feature_data = np.concatenate((feature_data, CCLE), axis=1)
elif Instructions == 3:
	print("3")
	drug_row = np.load("../LINCS_data_integration/drug_row_samples_aug_28_8.npy")
	drug1_chem = np.array(pd.read_csv("../../Data/Narjes_data/drug1_chem/28_8_drug1_chem.csv", header=None))
	drug_row = np.concatenate((drug_row.T, drug1_chem), axis=1)
	drug_col = np.load("../LINCS_data_integration/drug_col_samples_aug_28_8.npy")
	drug2_chem = np.array(pd.read_csv("../../Data/Narjes_data/drug2_chem/28_8_drug2_chem.csv", header=None))
	drug_col = np.concatenate((drug_col.T, drug2_chem), axis=1)
	feature_data = np.concatenate((drug_row, drug_col), axis=1)
	no_DU145 = pd.read_csv("../../Data/Training_data/CCLE_no_DU145.csv")
	feature_data = feature_data[no_DU145.Old_index,]
elif Instructions == 4:
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
else:
	print(Instructions)


#COLLECT LABELS
training_data = pd.read_csv(f"../../Data/Training_data/New_labels/tas_method/3_classes/synergy_loewe_IQR.tsv", sep="\t")
labels = training_data.class_numerical
indices = training_data.New_index
#######################################################################################


#Create results dataframe
columns = ["Classifier", "Accuracy", "ROC AUC", "Recall", "Precision"]
Results_Data = pd.DataFrame(columns=columns, dtype=object)
lb = LabelBinarizer()
dummy_y = lb.fit_transform(labels)
X_train, X_test, Index_train, Index_test, y_train, y_test = train_test_split(feature_data, indices, dummy_y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)
if Instructions == 0:
	history, y_pred, y_prob, y_test_results, Results_Data = LINCS(X_train, y_train, X_test, y_test, X_val, y_val, Results_Data, "LINCS")
	Results_Data.to_csv(f"Results/tables/feature_importance/LINCS_results_{array_job}.tsv", sep="\t", index=False)
	main_df = training_data.copy()
	results(history, y_pred, y_prob, y_test_results, "LINCS", Index_test, main_df)
elif Instructions == 1:
	history, y_pred, y_prob, y_test_results, Results_Data = ChemoPy_CCLE(X_train, y_train, X_test, y_test, X_val, y_val, Results_Data, "ChemoPy_CCLE")
	Results_Data.to_csv(f"Results/tables/feature_importance/ChemoPy_CCLE_results_{array_job}.tsv", sep="\t", index=False)
	main_df = training_data.copy()
	results(history, y_pred, y_prob, y_test_results, "ChemoPy_CCLE", Index_test, main_df)
elif Instructions == 2:
	history, y_pred, y_prob, y_test_results, Results_Data = LINCS_CCLE(X_train, y_train, X_test, y_test, X_val, y_val, Results_Data, "LINCS_CCLE")
	Results_Data.to_csv(f"Results/tables/feature_importance/LINCS_CCLE_results_{array_job}.tsv", sep="\t", index=False)
	main_df = training_data.copy()
	results(history, y_pred, y_prob, y_test_results, "LINCS_CCLE", Index_test, main_df)
elif Instructions == 3:
	history, y_pred, y_prob, y_test_results, Results_Data = LINCS_ChemoPy(X_train, y_train, X_test, y_test, X_val, y_val, Results_Data, "LINCS_ChemoPy")
	Results_Data.to_csv(f"Results/tables/feature_importance/LINCS_ChemoPy_results_{array_job}.tsv", sep="\t", index=False)
	main_df = training_data.copy()
	results(history, y_pred, y_prob, y_test_results, "LINCS_ChemoPy", Index_test, main_df)
elif Instructions == 4:
	history, y_pred, y_prob, y_test_results, Results_Data = LINCS_ChemoPy_CCLE(X_train, y_train, X_test, y_test, X_val, y_val, Results_Data, "LINCS_ChemoPy_CCLE")
	Results_Data.to_csv(f"Results/tables/feature_importance/LINCS_ChemoPy_CCLE_{array_job}.tsv", sep="\t", index=False)
	main_df = training_data.copy()
	results(history, y_pred, y_prob, y_test_results, "LINCS_ChemoPy_CCLE", Index_test, main_df)
