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

bootstrap = False
depth = 13
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=1)

fdtc = RandomForestClassifier(max_depth=depth, bootstrap=bootstrap, random_state=0)
fdtc.fit(X_train, y_train)
score = fdtc.score(X_test, y_test)
print("Pre prune score: ", score)

path = fdtc.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

for alpha in ccp_alphas:
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=1)

    rfc = RandomForestClassifier(random_state=0, bootstrap=bootstrap, max_depth=depth, ccp_alpha=alpha)
    rfc.fit(X_train, y_train)

    predictions = rfc.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    accuracy_list.append(accuracy)
    # print("Accuracy = ",accuracy)


print(max(accuracy_list),accuracy_list.index(max(accuracy_list)))

plt.figure(figsize=(5,5))
plt.scatter(list(range(1,30)),accuracy_list)
plt.plot(list(range(1,30)),accuracy_list)

plt.title("Finding max_depth")
plt.xlabel("max_depth")
plt.ylabel("Average accuracy")

# plt.savefig("deep forest.jpg", bbox_inches="tight", dpi=300)


plt.show()






 