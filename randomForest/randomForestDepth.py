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


accuracy_list = []
for depth in range(1,30):
    accuracys = [] 
    for state in range(1,10):
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=state)
        rfc = RandomForestClassifier(max_depth=depth, random_state=1)
        rfc.fit(X_train, y_train)

        predictionsB = rfc.predict(X_test)

        accuracy = accuracy_score(y_test, predictionsB)

        predictions = rfc.predict(X_test)

        accuracys.append(accuracy)
    accuracy_list.append(np.mean(accuracys))
    
print(max(accuracy_list),accuracy_list.index(max(accuracy_list)))

plt.figure(figsize=(5,5))
plt.scatter(list(range(1,30)),accuracy_list)
plt.plot(list(range(1,30)),accuracy_list)

plt.title("Finding max_depth")
plt.xlabel("max_depth")
plt.ylabel("Average accuracy")

plt.savefig("deep forest.jpg", bbox_inches="tight", dpi=300)


plt.show()




