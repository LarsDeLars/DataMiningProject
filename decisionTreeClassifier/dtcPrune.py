import numpy as np 
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Importing the data using the pandas library for importing .csv files. 
df = pd.read_csv('./data/beer_profile_and_ratings.csv', sep=',', header=0)
df.drop(columns=['Brewery','Beer Name (Full)','Description'], inplace=True)

attribute_names = df.columns[3:]

# Create our matrix with data. 
X = df.iloc[:, 3:22]

# We simplefy our classes by merging classes with similar prefixes. 
common_prefix = np.unique([style.split(' - ')[0] for style in df['Style']])
style_mapping = {style: style.split(' - ')[0] for style in df['Style']}
df['StyleSimple'] = df['Style'].map(style_mapping)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['StyleSimple'])

# These are the optimal values we found in dtcSplit.py and dtcDepth.py. 
split = 3
depth = 17
# Create a train and test set. 
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=1)

# Run the classifier without any pruning
dtc = tree.DecisionTreeClassifier(criterion="gini", min_samples_split=split, max_depth=depth, random_state=0)
dtc.fit(X_train, y_train)
score = dtc.score(X_test, y_test)

# Find the path and find the ccp_alphas and impurities
path = dtc.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities


mean_alphas = []
# We will check every ccp_alpha
for ccp_alpha in ccp_alphas:   
    current_alpha = [] 
    # Run the algorithm 15 times to ensure that the tests are fair. 
    for i in range(1,15):
        # Create a (new) random train/test set. 
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=i)
        # Create the decision tree
        dtc = tree.DecisionTreeClassifier(criterion="gini",min_samples_split=split, max_depth=depth,ccp_alpha=ccp_alpha, random_state=0)
        dtc = dtc.fit(X_train, y_train)

        # Predict the test set and add it to the list. 
        y_pred_test = dtc.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        current_alpha.append(accuracy_test)
    # Add the mean of the 15 runs to the list. 
    mean_alphas.append(np.mean(current_alpha))
    
# Show the highest accuracy we generated
print("Optimal prune with accuracy: ", max(mean_alphas))
# Find the optimal alpha. 
optimal_alpha = ccp_alphas[mean_alphas.index(max(mean_alphas))]
print("Optimal alpha: ", optimal_alpha)

# Create a plot to visualize the alphas and average accuracy score. 
plt.figure(figsize=(5,5))
plt.plot(ccp_alphas,mean_alphas)

plt.title("Finding ccp_alpha")
plt.xlabel("ccp_alpha")
plt.ylabel("Average accuracy")

plt.savefig("optimal ccp.jpg", bbox_inches="tight", dpi=300)

plt.show()


