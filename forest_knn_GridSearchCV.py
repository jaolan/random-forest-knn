import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import GridSearchCV
#import matplotlib.pyplot as plt
#import numpy as np

HEADER_FILE = "/Users/superman/Downloads/your-header-file.txt"
TRAINING_FILE = '/Users/superman/Downloads/your-train-file.txt'
TESTING_FILE = '/Users/superman/Downloads/your-test-file.txt'


def get_headers():
    with open(HEADER_FILE) as hf:
        headers = [line.rstrip('\n') for line in hf]
        hf.close
        return headers

headers = get_headers()
df = pd.read_table(TRAINING_FILE, delimiter=' ', names=headers)

#TODO  drop last col till can figure out why headers dont match up
df.drop(df.columns[[-1,]], axis=1, inplace=True)
target = df.iloc[:,-1]
#2nd drop to pass features matrix all but binary classifier col
df.drop(df.columns[[-1,]], axis=1, inplace=True)
features = df.astype(float)

'''
df2 = pd.read_table(TESTING_FILE, delimiter=' ', names=headers)
#TODO  drop last col till can figure out why headers dont match up
df2.drop(df.columns[[-1,]], axis=1, inplace=True)
target_test = df2.iloc[:,-1]
#2nd drop to pass features matrix all but binary classifier col
df2.drop(df2.columns[[-1,]], axis=1, inplace=True)
features_test = df2.astype(float)
'''
# Create a pipeline
pipe = Pipeline([("classifier", RandomForestClassifier())])
# Create dictionary with candidate learning algorithms and their hyperparameters
search_space = [{"classifier": [LogisticRegression()],
                     "classifier__penalty": ['l1', 'l2'],
                     "classifier__C": np.logspace(0, 4, 10, 20, 30)},
                    {"classifier": [RandomForestClassifier()],
                     "classifier__n_estimators": [10, 100],
                     "classifier__max_features": [1, 2, 3, 6, 15]}]
    # Create grid search
gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0) # Fit grid search
best_model = gridsearch.fit(features, target)#print(y_pred)
print(best_model.best_estimator_.get_params()["classifier"])
