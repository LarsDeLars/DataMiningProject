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


# IMPORTING DATA
df = pd.read_csv('./data/beer_profile_and_ratings.csv', sep=',', header=0)
df.drop(columns=['Brewery','Beer Name (Full)','Description'], inplace=True)

attribute_names = df.columns[3:]

X = df.iloc[:, 3:22]

# SIMPLEFYING THE CLASSES
common_prefix = np.unique([style.split(' - ')[0] for style in df['Style']])
style_mapping = {style: style.split(' - ')[0] for style in df['Style']}
df['StyleSimple'] = df['Style'].map(style_mapping)

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(df['StyleSimple'])
accuracyB_list = []
accuracy_list = []


for k in range(0,20):
    print(k)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=k)

    rfc = RandomForestClassifier(random_state=0, bootstrap=False)
    rfc.fit(X_train, y_train)

    predictions = rfc.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    accuracy_list.append(accuracy)
    # print("Accuracy = ",accuracy)


for k in range(0,20):
    print(k)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=k)

    rfc = RandomForestClassifier(random_state=0, bootstrap=True)
    rfc.fit(X_train, y_train)

    predictions = rfc.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    accuracyB_list.append(accuracy)
    # print("Accuracy = ",accuracy)

# fig, ax = plt.subplots()

# fruits = ['Bootstrap', 'No Bootstrap']
counts = [np.mean(accuracyB_list), np.mean(accuracy_list)]
# bar_labels = ['Bootstrap', 'No Bootstrap']
# bar_colors = ['tab:blue', 'tab:orange']

# ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

# ax.set_ylabel("Average accuracy")
# ax.set_title("Bootstrap vs No Bootstrap")

# plt.savefig("boobtrap.jpg", bbox_inches="tight", dpi=300)

# plt.show()


print(counts)

# max depth chosen by getting the average max_depth of 19 random_states
# bootstrap = False works better than bootstrap = True





 