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
optimal_depthList = []

for k in range(1,20):
    print(k)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=k)

    accuracy_list = []
    depthList = [] 
    for depth in range(2,40):
        rfc = RandomForestClassifier(max_depth=depth, random_state=0)
        rfc.fit(X_train, y_train)

        predictions = rfc.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        accuracy_list.append(accuracy)
        depthList.append(depth)
    max_accuracy_index = np.argmax(accuracy_list)
    optimal_depth = depthList[max_accuracy_index]
    optimal_depthList.append(optimal_depth)
    optimal_accuracy = accuracy_list[max_accuracy_index]
    print(f"Optimal Max Depth: {optimal_depth}")
    print("Max accuracy: ", max(accuracy_list))

def Average(lst): 
    return sum(lst) / len(lst) 

print(Average(optimal_depthList))

plt.figure(figsize=(10,10))
plt.scatter(depthList,accuracy_list)
plt.show()





 