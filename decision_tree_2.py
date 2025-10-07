#-------------------------------------------------------------------------
# AUTHOR: Kenia Velasco
# FILENAME: decision_tree_2.py
# SPECIFICATION: Train Decision Tree models (entropy, max_depth=5) on three contact lens
#                training sets, evaluate on a common test set, repeat 10 times, and
#                print the average accuracy per training set.
# FOR: CS 4210 - Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

from sklearn import tree
import pandas as pd
from pathlib import Path
import sys

# Datasets
dataSets = [
    'contact_lens_training_1.csv',
    'contact_lens_training_2.csv',
    'contact_lens_training_3.csv'
]

def norm(s: str) -> str:
    return str(s).strip().capitalize()


AGE_MAP   = {"Young": 1, "Prepresbyopic": 2, "Presbyopic": 3}
SPEC_MAP  = {"Myope": 1, "Hypermetrope": 2}
ASTIG_MAP = {"No": 1, "Yes": 2}
TEAR_MAP  = {"Reduced": 1, "Normal": 2}
LABEL_MAP = {"Yes": 1, "No": 2}

def encode_features(age, spec, astig, tear):
    return [
        AGE_MAP[norm(age)],
        SPEC_MAP[norm(spec)],
        ASTIG_MAP[norm(astig)],
        TEAR_MAP[norm(tear)],
    ]

def encode_label(lbl):
    return LABEL_MAP[norm(lbl)]

# Read the test data once
dbTest = []
df_test = pd.read_csv('contact_lens_test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())

for ds in dataSets:
    df_train = pd.read_csv(ds)

    X, Y = [], []
    for _, row in df_train.iterrows():
        age, spec, astig, tear, label = row.tolist()
        X.append(encode_features(age, spec, astig, tear))
        Y.append(encode_label(label))

    accuracies = []
    for i in range(10):
       
        clf = tree.DecisionTreeClassifier(
            criterion='entropy',
            max_depth=5,
            splitter='best',   
            random_state=i
        )
        clf.fit(X, Y)

    
        correct, total = 0, 0
        for rec in dbTest:
            age, spec, astig, tear, true_lbl = rec
            feats = encode_features(age, spec, astig, tear)
            pred = clf.predict([feats])[0]           # integer 1 or 2
            if pred == encode_label(true_lbl):
                correct += 1
            total += 1

        accuracies.append(correct / total if total else 0.0)

    avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
    print(f"Average accuracy for {ds}: {avg_acc:.2f}")
