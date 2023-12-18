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

from sklearn.ensemble import RandomForestClassifier


#############################IMPORTING DATA ##############################################

df = pd.read_csv('./data/beer_profile_and_ratings.csv', sep=',', header=0)
df.drop(columns=['Brewery','Beer Name (Full)','Description'], inplace=True)

attribute_names = df.columns[3:]

X = df.iloc[:, 3:22]

############################ SIMPLEFYING THE CLASSES #####################################
common_prefix = np.unique([style.split(' - ')[0] for style in df['Style']])
style_mapping = {style: style.split(' - ')[0] for style in df['Style']}
df['StyleSimple'] = df['Style'].map(style_mapping)

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(df['StyleSimple'])

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=1)

accuracy_list = []
accuracy_listB = []
depthList = [] 
for depth in range(20,30):
    rfcB = RandomForestClassifier(max_depth=depth, random_state=1)
    rfcB.fit(X_train, y_train)

    predictionsB = rfcB.predict(X_test)

    accuracyB = accuracy_score(y_test, predictionsB)

    rfc = RandomForestClassifier(max_depth=depth, random_state=1,bootstrap=False)
    rfc.fit(X_train, y_train)

    predictions = rfc.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    accuracy_list.append(accuracy)
    accuracy_listB.append(accuracyB)
    depthList.append(depth)

print("Max accuracy no bootstrap: ", max(accuracy_list))
print("Max accuracy with bootstrap: ", max(accuracy_listB))
plt.figure(figsize=(10,10))
plt.scatter(depthList,accuracy_list)
plt.scatter(depthList, accuracy_listB)
plt.show()

# CONCLUSION:
# NO BOOTSTRAP IS BETTER



