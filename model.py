#!/usr/bin/env python
# coding: utf-8

# In[92]:


import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# In[93]:


df = pd.read_csv('heart.csv')
print(df)


# In[94]:


X = df.drop('output', axis = 1)
Y = df['output']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


# In[95]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[96]:


knn = KNeighborsClassifier()

param_grid = {'n_neighbors': [3,5,7, 9]}

search_grid = GridSearchCV(knn, param_grid, cv = 5)

search_grid.fit(X_train, Y_train)

best_model = search_grid.best_estimator_

best_model.fit(X_train, Y_train)


# In[97]:


# make pickle
import pickle
pickle.dump(best_model, open("model.pkl", "wb"))

