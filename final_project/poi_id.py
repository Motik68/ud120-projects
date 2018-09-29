#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'total_stock_value', 'total_payments']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('BELFER ROBERT')
data_dict.pop('BHATNAGAR SANJAY')

for name in data_dict:
    for feature in data_dict[name]:
        if data_dict[name][feature] == 'NaN':
            data_dict[name][feature] = 0

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

for name in my_dataset:
    if my_dataset[name]['total_payments'] > 0 and my_dataset[name]['total_stock_value'] > 0:
        my_dataset[name]['stock_plus_payments'] = my_dataset[name]['total_payments'] + my_dataset[name]['total_stock_value']
    else:
        my_dataset[name]['stock_plus_payments'] = 0

features_list.append('stock_plus_payments')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn import tree, model_selection
# clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

np.random.seed(42)
parameters = {'criterion':('gini', 'entropy'), 'max_depth':(10, 50, 100, None), 'class_weight':('balanced', None)}
svr = tree.DecisionTreeClassifier()
clf = model_selection.GridSearchCV(svr, parameters, cv=10, scoring=['precision', 'recall', 'f1'], 
                                   refit='precision', n_jobs=1, verbose=1)
clf.fit(features, labels)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf.best_estimator_, my_dataset, features_list)