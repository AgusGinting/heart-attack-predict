import pickle

import numpy as np
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

#load data
df = pd.read_csv('./heart.csv')
print(df)


df.info()

#cek null data
df.isnull().sum()

#cek 10 baris data
df.head(10)

#deskripsi
df.describe()

#korelasi antar data
plt.figure(figsize= (20, 15))
sns.heatmap(df.corr(), cmap = "YlGnBu", annot = True)
plt.show

#cek perbandingan jumlah output
tar = df['output'].value_counts()
plt.bar(tar.index, tar.values)
plt.xticks([0,1])

#hapus baris yang double
df[df.duplicated()]
df.drop_duplicates(keep='first',inplace=True)

#split train dan test data
X = df.drop('output', axis = 1)
Y = df['output']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

#scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Model
knn = KNeighborsClassifier()

param_grid = {'n_neighbors': [3,5,7, 9]}

search_grid = GridSearchCV(knn, param_grid, cv = 5)

search_grid.fit(X_train, Y_train)

best_model = search_grid.best_estimator_

best_model.fit(X_train, Y_train)

Y_pred_train = best_model.predict(X_train)
accuracy_train = accuracy_score(Y_train, Y_pred_train)

Y_pred_test = best_model.predict(X_test)
accuracy_test = accuracy_score(Y_test, Y_pred_test)

print("accuracy train : ", accuracy_train)
print("accuracy test : ", accuracy_test)

print(classification_report(Y_test, Y_pred_test))


# make pickle
pickle.dump(best_model, open("model.pkl", "wb"))



