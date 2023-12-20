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


############################# CREATING TEST/TRAIN SETS ###################################
max_depth_list = []
for split in range(2, 25):
    current_depths = [] 
    for i in range(1,15):
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=i)
        dtc = tree.DecisionTreeClassifier(criterion="gini", min_samples_split=split, random_state=i)
        dtc = dtc.fit(X_train, y_train)

        y_pred_test = dtc.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        current_depths.append(accuracy_test)
    max_depth_list.append(np.mean(current_depths))
    

print("Optimal depth with accuracy: ", max(max_depth_list))

plt.figure(figsize=(10,10))
plt.scatter(list(range(2, 25)), max_depth_list)
plt.plot(list(range(2, 25)),max_depth_list)

plt.title("Finding Optimal min_samples_split")
plt.xlabel("min_samples_split")
plt.ylabel("Average accuracy")

plt.savefig("min_split.jpg", bbox_inches="tight", dpi=300)

plt.show()


