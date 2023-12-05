import numpy as np 
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/beer_profile_and_ratings.csv', sep=',', header=0)

df.drop(columns=['Brewery','Beer Name (Full)','Description'], inplace=True)

attribute_names = df.columns[3:]
attribute_names_no_review = df.columns[3:16] 

common_prefix = np.unique([style.split(' - ')[0] for style in df['Style']])

style_mapping = {style: style.split(' - ')[0] for style in df['Style']}

df['StyleSimple'] = df['Style'].map(style_mapping)

label_encoder = LabelEncoder()

X = df.iloc[:, 3:22]
# X_no_review = df.iloc[:, 3:16]

y_encoded = label_encoder.fit_transform(df['StyleSimple'])
# y_encoded_r = label_encoder.fit_transform(df['StyleSimple'].iloc[:16])

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=40)
# X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_no_review, y_encoded_r, test_size=0.2, random_state=42)

accuracy_list = [] 
iS = []
for i in range(2,20):
    dtc = tree.DecisionTreeClassifier(criterion="gini", min_samples_split=i)
    dtc = dtc.fit(X_train, y_train)

    y_pred_test = dtc.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    accuracy_list.append(accuracy_test)
    iS.append(i)

print("Max accuracy: ", max(accuracy_list))
plt.figure(figsize=(10,10))
plt.scatter(iS,accuracy_list)
plt.show()


# dtcR = tree.DecisionTreeClassifier(criterion="gini", min_samples_split=100)
# dtcR = dtcR.fit(X_train_r, y_train_r)

# y_pred_test_r = dtcR.predict(X_test_r)

# accuracy_testR = accuracy_score(y_test_r, y_pred_test_r)

# plt.figure(figsize=(10, 10))
# plot_tree(dtc, filled=True, feature_names=attribute_names, class_names=label_encoder.classes_)
# plt.show()


# print("Test Set No Review Accuracy: ", accuracy_testR)
