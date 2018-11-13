import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

#df = pd.read_table('/Users/superman/Downloads/train.nmv.txt', delimiter=" ", sep =" ")
HEADER_FILE = "/Users/superman/Downloads/attr.txt"
TRAINING_FILE = '/Users/superman/Downloads/train.nmv.txt'
#CSV_EXPORT_PATH = "/Users/superman/Downloads/train.nmv.csv"

def get_headers():
    with open(HEADER_FILE) as hf:
        headers = [line.rstrip('\n') for line in hf]
        hf.close
        return headers

headers = get_headers()
df = pd.read_table(TRAINING_FILE, delimiter=' ', names=headers)
#weight=balanced since year col and others skew attribute balance
random_forest = RandomForestClassifier(random_state=0, n_jobs=-1, class_weight="balanced")
#print(headers)
#df.iloc
#TODO  drop last col till can figure out why headers dont match up
df.drop(df.columns[[-1,]], axis=1, inplace=True)
print(df)
print(df.describe())
print(df.plot())
