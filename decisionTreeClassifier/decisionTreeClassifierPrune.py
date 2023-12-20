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
split = 5
depth = 13
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=1)

fdtc = tree.DecisionTreeClassifier(criterion="gini", min_samples_split=split, max_depth=depth, random_state=0)
fdtc.fit(X_train, y_train)
score = fdtc.score(X_test, y_test)
print("Pre prune score: ", score)

path = fdtc.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities


clfs = []
for ccp_alpha in ccp_alphas:   
    current_depths = [] 
    # for i in range(1,15):
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)
    dtc = tree.DecisionTreeClassifier(criterion="gini",min_samples_split=split, max_depth=depth,ccp_alpha=ccp_alpha, random_state=1)
    dtc = dtc.fit(X_train, y_train)

    y_pred_test = dtc.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    current_depths.append(accuracy_test)
    clfs.append(accuracy_test)
    # clfs.append(np.mean(current_depths))
    

print("Optimal depth with accuracy: ", max(clfs))
optimal_alpha = ccp_alphas[clfs.index(max(clfs))]

print("Optimal alpha: ", optimal_alpha)

plt.figure(figsize=(10,10))
plt.plot(ccp_alphas,clfs)

plt.title("Finding ccp_alpha")
plt.xlabel("ccp_alpha")
plt.ylabel("Average accuracy")

plt.savefig("optimal ccp.jpg", bbox_inches="tight", dpi=300)

plt.show()


