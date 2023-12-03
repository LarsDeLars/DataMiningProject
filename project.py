import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import plot_tree
# import scipy.io as sio

# df = pd.read_csv('./data/beer_profile_and_ratings.csv', sep=',', header=None)
df = pd.read_csv('./data/beer_profile_and_ratings.csv', sep=',', header=0)

df.drop(columns=['Brewery','Beer Name (Full)','Description'], inplace=True)

attribute_names = df.columns

X = df.iloc[:1039, 3:22]
y = df['Style'].iloc[:1039]
 #altbiercount = (df['Style'] == "Altbier").sum()
 #print(altbiercount)
dtc = tree.DecisionTreeClassifier(criterion = "gini", min_samples_split = 100)
dtc = dtc.fit(X, y)
plt.figure(figsize=(20, 20)) 
plot_tree(dtc, filled=True, node_ids = True, feature_names=attribute_names, class_names=y)
plt.show()



# below_3_count = (df['review_overall'] < 3).sum()
# above_4_count = (df['review_overall'] > 4.5).sum()
# print(below_3_count)
# print(above_4_count)

# plt.hist(df['review_overall'], bins=100, color='blue', edgecolor='black')

# plt.xlabel('Values')
# plt.ylabel('Frequency')
# plt.title('Distribution Plot of Column review_overall')
# plt.show()