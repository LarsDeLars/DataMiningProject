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
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=1)


############################# DECISION TREE CLASSIFIER MAX SPLIT CALCULATION ##################
total_splits = []
total_depths = [] 
total_prune = []
for random_state in range(1,15):
    min_split_accuracy_list = [] 
    split_list = []
    for split in range(2,20):
        dtc = tree.DecisionTreeClassifier(criterion="gini", min_samples_split=split, random_state=random_state)
        dtc = dtc.fit(X_train, y_train)

        y_pred_test = dtc.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        min_split_accuracy_list.append(accuracy_test)
        split_list.append(split)

    min_split = split_list[min_split_accuracy_list.index(max(min_split_accuracy_list))]
    # print("Max accuracy min split: ", min_split, max(min_split_accuracy_list))
    # plt.figure(figsize=(10,10))
    # plt.scatter(iS,accuracy_list)
    # plt.show()
    total_splits.append(min_split)
    
    ############################# OPTIMAL DEPTH DECISION TREE CLASSIFIER ################################
    max_depth_list = []
    depthList = []
    for depth in range(2,30):
        dtc = tree.DecisionTreeClassifier(criterion="gini", min_samples_split=min_split, max_depth=depth, random_state=random_state)
        dtc = dtc.fit(X_train, y_train)

        y_pred_test = dtc.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        max_depth_list.append(accuracy_test)
        depthList.append(depth)

    max_depth = depthList[max_depth_list.index(max(max_depth_list))]
    # print("Optimal depth with accuracy: ", max_depth, max(max_depth_list))
    # plt.figure(figsize=(10,10))
    # plt.scatter(depthList,accuracy_list2)
    # plt.show()
    total_depths.append(max_depth)


    ############################ POST PRUNING ###################################################

    path = dtc.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities


    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(criterion="gini", min_samples_split=min_split,max_depth=max_depth, ccp_alpha=ccp_alpha, random_state=random_state)
        clf.fit(X_train, y_train)
        clfs.append(clf)

    accuracies = [clf.score(X_test, y_test) for clf in clfs]
    optimal_alpha = ccp_alphas[accuracies.index(max(accuracies))]

    pruned_clf = tree.DecisionTreeClassifier(criterion="gini", min_samples_split=min_split, max_depth=max_depth, ccp_alpha=optimal_alpha, random_state=random_state)

    pruned_clf.fit(X_train, y_train)

    # Evaluate the accuracy on the test set after pruning
    accuracy_after_pruning = pruned_clf.score(X_test, y_test)
    # print(f'Accuracy after pruning: {accuracy_after_pruning:.4f}')

    total_prune.append(optimal_alpha)

mean_split = np.mean(total_splits)
mean_depth = np.mean(total_depths)
mean_prune = np.mean(total_prune)

print("Mean Split:", mean_split)
print("Mean Depth:", mean_depth)
print("Mean Prune:", mean_prune)

fdtc = tree.DecisionTreeClassifier(criterion="gini", min_samples_split=mean_split, max_depth=mean_depth, ccp_alpha=mean_prune)
score = fdtc.score(X_test, y_test)
print("Final accuracy: ", score )

