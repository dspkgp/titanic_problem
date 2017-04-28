import pandas as pd
import os

def read_training_data():
	filepath=os.getcwd() + "/train.csv"
	return pd.read_csv(filepath)

def read_test_data():
	filepath=os.getcwd() + "/test.csv"
	return pd.read_csv(filepath)

def read_gender_data():
	filepath=os.getcwd() + "/gender_submission.csv"
	return pd.read_csv(filepath)
