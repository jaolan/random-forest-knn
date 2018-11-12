import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#df = pd.read_table('/Users/superman/Downloads/train.nmv.txt', delimiter=" ", sep =" ")
HEADER_FILE = "/Users/superman/Downloads/attr.txt"
TRAINING_FILE = '/Users/superman/Downloads/train.nmv.txt'
#CSV_EXPORT_PATH = "/Users/superman/Downloads/train.nmv.csv"

def get_headers():
    with open(HEADER_FILE) as hf:
        headers = [line.rstrip('\n') for line in hf]
        hf.close
        return headers\

headers = get_headers()
df = pd.read_table(TRAINING_FILE, delimiter=' ', names=headers)

#print(headers)
#df.iloc
print(df)