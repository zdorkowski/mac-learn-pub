# Titanic Classification Analysis
# Step 1 — Import libraries and load dataset

import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.metrics as metrics

# Load the dataset
FILE_PATH = "Titanic-Dataset.csv"
titanic = pd.read_csv(FILE_PATH)
# Preview structure
print("Shape:", titanic.shape)
titanic.head()