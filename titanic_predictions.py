#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:28:29 2017

@author: hayabusa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#grab data
data_train_raw = pd.read_csv('train.csv')
data_test_raw = pd.read_csv('test.csv')



# some functions to clean/massage the data

def format_names(dt):
    # take only the last name and prefix
    dt['Last_name'] = dt.Name.apply(lambda x: x.split(', ')[0])
    dt['Name_prefix'] = dt.Name.apply(lambda x: x.split(', ')[1].split(' ')[0])
    return dt


def format_ages(dt):
    # put the ages into bins
    dt.Age = dt.Age.fillna(-0.5)
    
    bins = (-1, 0, 4, 12, 18, 28, 38, 60, 140)
    age_categories = ['Unknown', 'Baby', 'Child', 'Teenager', 'Young_adult', 'Adult', 'Middle_aged', 'Senior' ]

    cats = pd.cut(dt.Age, bins, labels=age_categories)
    dt.Age = cats
    return dt


def format_fares(dt):
    # put the fares into quartiles
    dt.Fare = dt.Fare.fillna(-0.5)
    
    bins = (-1, 0, 8, 15, 31, 1000)
    fare_categories = ['Unknown', 'FirstQ', 'SecondQ', 'ThirdQ', 'FourthQ']
    
    cats = pd.cut(dt.Fare, bins, labels=fare_categories)
    dt.Fare = cats
    return dt
    

def format_cabins(dt):
    # take only the cabin deck number -- discard room number 
    dt.Cabin = dt.Cabin.fillna('N')
    dt.Cabin = dt.Cabin.apply(lambda x: x[0])
    return dt


def drop_junk_features(dt):
    return dt.drop(['Name', 'Ticket', 'Embarked'], axis=1)


def format_data(dt):
    dt = format_names(dt)
    dt = format_ages(dt)
    dt = format_fares(dt)
    dt = format_cabins(dt)
    dt = drop_junk_features(dt)
    return dt
    
    

# obtain clean & formatted data:
dt_train = format_data(data_train_raw)
dt_test = format_data(data_test_raw)    

 
    
# encode strings w/ LabelEncoder()
from sklearn import preprocessing

def encode_strings(train, test):
    features = ['Sex', 'Age', 'Fare', 'Cabin', 'Last_name', 'Name_prefix']
    data = pd.concat([train[features], test[features]])
    
    for feature in features:    
        le = preprocessing.LabelEncoder()
        le = le.fit(data[feature])
        train[feature] = le.transform(train[feature])
        test[feature] = le.transform(test[feature])
        
    return train, test

dt_train, dt_test = encode_strings(dt_train, dt_test)


    
# perform an 80/20 train/test split 
from sklearn.model_selection import train_test_split

X = dt_train.drop(['PassengerId', 'Survived'], axis=1)
Y = dt_train.Survived

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)    



# Import RandomForestClassifier and some metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score

clf = RandomForestClassifier()


# some parameters to try with GridSearchCV
parameters = {'n_estimators': [2, 3, 6, 9, 12, 15, 19], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 4, 7, 10, 14, 19, 24], 
              'min_samples_split': [2, 3, 5, 8, 12],
              'min_samples_leaf': [1,2,3,5,8,13]
             }


# use GridSearchCV to fit the RandomForestClassifier to the best parameter  
# values defined above
accuracy = make_scorer(accuracy_score)

grid = GridSearchCV(clf, parameters, scoring=accuracy)
grid = grid.fit(x_train, y_train)


# set the classifer to the best values
clf = grid.best_estimator_


# Fit the classifier to the train data, then test:
clf.fit(x_train, y_train)

predictions = clf.predict(x_test)
acc_score = accuracy_score(y_test,predictions)
print("Single shot accuracy on train/test split: %s" % acc_score) ### note: scores around 0.83 on a trial run   

     
# Do cross-validation with Kfold on our classifier, 
# and print the aggregate results:
from sklearn.cross_validation import KFold

def run_kfold(clf):
    
    kf = KFold(len(X), n_folds=10)
    outcomes = []
    fold = 1
    
    for train_index, test_index in kf:
        x_train, x_test = X.values[train_index], X.values[test_index]
        y_train, y_test = Y.values[train_index], Y.values[test_index]
        
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        acc = accuracy_score(y_test, predictions)
        
        outcomes.append(acc)
        print("Fold %s, Accuracy = %s" % (fold, acc))
        fold += 1
        
    mean_accuracy = np.mean(outcomes)        
    print("*** Mean Accuracy = %s" % mean_accuracy)

print("Running KFold CV with 10 splits:")
run_kfold(clf)
 

# Now run the classifier on the actual test data:   
def output_results(dt):
    print("")
    print(" *** Outputting test results as .csv ***")    
    ids = dt.PassengerId
    preds = clf.predict(dt.drop('PassengerId', axis=1))
    
    output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': preds })
    output.to_csv('titanic_predictions.csv', index = False)
    return output
    
output_results(dt_test)