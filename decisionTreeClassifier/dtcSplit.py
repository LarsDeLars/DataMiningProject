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


split_list = []
# We check for minimal splits 2 through 25. 
for split in range(2, 25):
    current_split = [] 
    # Run the algorithm 15 times to ensure that the tests are fair. 
    for i in range(1,15):
        # Create a (new) random train/test set. 
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=i)
        # Create the decision tree
        dtc = tree.DecisionTreeClassifier(criterion="gini", min_samples_split=split, random_state=i)
        dtc = dtc.fit(X_train, y_train)

        # Predict the test set and add it to the list. 
        y_pred_test = dtc.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        current_split.append(accuracy_test)
    # Add the mean of the 15 runs to the list. 
    split_list.append(np.mean(current_split))
    
# Show the highest accuracy we generated and the optimal minimal split
print(f"Optimal minimal split {split_list.index(max(split_list))} with accuracy: {max(split_list)} ")

# Create a plot to visualize the minimal splits and average accuracy score. 
plt.figure(figsize=(5,5))
plt.scatter(list(range(2, 25)), split_list)
plt.plot(list(range(2, 25)),split_list)

plt.title("Finding Optimal min_samples_split")
plt.xlabel("min_samples_split")
plt.ylabel("Average accuracy")

plt.savefig("min_split.jpg", bbox_inches="tight", dpi=300)

plt.show()


