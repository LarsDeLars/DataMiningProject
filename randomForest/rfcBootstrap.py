import numpy as np 
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


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

accuracyB_list = []
accuracy_list = []

# First we test the algorithm without bootstrapping
for k in range(0,20):
    # Create a (new) random train/test set. 
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=k)

    # Create the random forest	
    rfc = RandomForestClassifier(random_state=0, bootstrap=False)
    rfc.fit(X_train, y_train)

    # Predict the test set and add it to the list. 
    predictions = rfc.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracy_list.append(accuracy)

# Calculate the mean accuracy without bootstrapping 
meanNoBoot = np.mean(accuracy_list)

# Now we test it with bootstrapping
for k in range(0,20):
    # Create a (new) random train/test set. 
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=k)

    # Create the random forest	
    rfc = RandomForestClassifier(random_state=0, bootstrap=True)
    rfc.fit(X_train, y_train)

    # Predict the test set and add it to the list. 
    predictions = rfc.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracyB_list.append(accuracy)

# Calculate the mean accuracy with bootstrapping 
meanBoot = np.mean(accuracyB_list)

# Show the accuracy
print(f"Mean accuracy with bootstrapping: {meanBoot}, Mean accuracy without bootstrapping: {meanNoBoot}")




 