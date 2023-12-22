import numpy as np 
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
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


accuracy_list = []
# We check for depths 1 through 30. 
for depth in range(1,30):
    accuracys = [] 
    # Run the algorithm 15 times to ensure that the tests are fair. 
    for state in range(1,15):
        # Create a train and test set. 
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=state)
        # Create the random forest	
        rfc = RandomForestClassifier(max_depth=depth, random_state=1)
        rfc.fit(X_train, y_train)

        # Predict the test set and add it to the list. 
        predictionsB = rfc.predict(X_test)
        accuracy = accuracy_score(y_test, predictionsB)
        predictions = rfc.predict(X_test)
        accuracys.append(accuracy)
    # Add the mean of the 15 runs to the list. 
    accuracy_list.append(np.mean(accuracys))

# Show the highest accuracy we generated and at what depth it occurerred. 
print(f'Highest mean accuracy: {max(accuracy_list)}, with depth {accuracy_list.index(max(accuracy_list))}')


# Create a plot to visualize the depths and average accuracy score. 
plt.figure(figsize=(5,5))
plt.scatter(list(range(1,30)),accuracy_list)
plt.plot(list(range(1,30)),accuracy_list)

plt.title("Finding max_depth")
plt.xlabel("max_depth")
plt.ylabel("Average accuracy")

plt.savefig("deep forest.jpg", bbox_inches="tight", dpi=300)

plt.show()




