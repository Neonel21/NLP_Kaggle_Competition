# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 22:23:03 2021

@author: Murat Baran Polat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.impute import SimpleImputer

sample_submission = pd.read_csv("sample_submission.csv")
test= pd.read_csv("test.csv")
train= pd.read_csv("train.csv")

count_vectorizer = feature_extraction.text.CountVectorizer()
example_train_vectors = count_vectorizer.fit_transform(train["text"][0:5])
print(example_train_vectors[0].todense().shape)
print(example_train_vectors[0].todense())

train_vectors = count_vectorizer.fit_transform(train["text"])

#ÖNEMLİ:
#-----------------------------------------------------------------------------------------
## note that we're NOT using .fit_transform() here. Using just .transform() makes sure
# that the tokens in the train vectors are the only ones mapped to the test vectors - 
# i.e. that the train and test vectors use the same set of tokens.
#-----------------------------------------------------------------------------------------

test_vectors = count_vectorizer.transform(test["text"])

clf = linear_model.RidgeClassifier()
percentages = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=3, scoring="f1")
print(percentages)

clf.fit(train_vectors, train["target"])
sample_submission["target"] = clf.predict(test_vectors)
sample_submission.head()


sample_submission.to_csv("submission.csv", index=False)