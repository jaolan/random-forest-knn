import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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



print(target)
print(features)

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.1, random_state=42)

standardizer = StandardScaler()
features_standardized = standardizer.fit_transform(features)
classifer = RandomForestClassifier(class_weight="balanced")
classifer.fit(features_train, target_train)
y_pred = classifer.predict(features_test)
print ("training accuracy: ", classifer.score(features_test, target_test))

'''
# Build a plot
plt.scatter(y_pred, target_test)
plt.xlabel('Prediction')
plt.ylabel('Real value')

# Now add the perfect prediction line
diagonal = np.linspace(0, np.max(target_test), 100)
plt.plot(diagonal, diagonal, '-r')
plt.show()
'''
#prints all 0 1 predictions
#print(y_pred)
