import numpy as np 
import pandas as pd
import matplotlib
from scipy import stats 
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy.stats import chi2_contingency
import seaborn as sns



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

splitDTC = 5
depthDTC = 13
ccp_alpha = 0.0007361568013986981

depthRFC = 17
bootstrap = False

runs = 2

############################# CREATING TEST/TRAIN SETS ###################################
accuracy_dtc = []
accuracy_rfc = []
for state in range(1, runs):
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.5, random_state=state)
    dtc = tree.DecisionTreeClassifier(criterion="gini", max_depth=depthDTC, random_state=0, min_samples_split=splitDTC, ccp_alpha=ccp_alpha)
    dtc = dtc.fit(X_train, y_train)

    rfc = RandomForestClassifier(criterion="gini", bootstrap=bootstrap, max_depth=depthRFC, random_state=0)
    rfc = rfc.fit(X_train, y_train)

    y_pred_test_dtc = dtc.predict(X_test)
    accuracy_test_dtc = accuracy_score(y_test, y_pred_test_dtc)
    accuracy_dtc.append(accuracy_test_dtc)
    
    y_pred_test_rfc = rfc.predict(X_test)
    accuracy_test_rfc = accuracy_score(y_test, y_pred_test_rfc)
    accuracy_rfc.append(accuracy_test_rfc)
    
meanT = np.mean(accuracy_dtc)
meanF = np.mean(accuracy_rfc)
print(f"Mean tree:{meanT}, Mean forest:{meanF}")

confidence_dtc = stats.t.interval(0.95, len(accuracy_dtc) - 1, loc=np.mean(accuracy_dtc), scale=stats.sem(accuracy_dtc))
confidence_rfc = stats.t.interval(0.95, len(accuracy_rfc) - 1, loc=np.mean(accuracy_rfc), scale=stats.sem(accuracy_rfc))


std_dev_dtc = np.std(accuracy_dtc)

# Calculate standard deviation for Random Forest
std_dev_rfc = np.std(accuracy_rfc)

print(f"Standard Deviation Decision Tree: {std_dev_dtc}")
print(f"Standard Deviation Random Forest: {std_dev_rfc}")


# plt.figure(figsize=(10,5))
# plt.scatter(list(range(1, runs)), accuracy_dtc,  label='Decision Tree')
# plt.plot(list(range(1, runs)),accuracy_dtc)

# plt.scatter(list(range(1, runs)), accuracy_rfc,  label='Random Forest')
# plt.plot(list(range(1, runs)),accuracy_rfc)
# plt.fill_between(list(range(1, runs)), confidence_dtc[0], confidence_dtc[1], alpha=0.2, color='blue')
# plt.fill_between(list(range(1, runs)), confidence_rfc[0], confidence_rfc[1], alpha=0.2, color='orange')

# plt.title("Decision Tree vs Random Forest with Confidence Intervals")
# plt.xlabel("Iteration")
# plt.ylabel("Average accuracy")
# plt.legend()


# plt.savefig("results.jpg", bbox_inches="tight", dpi=300)

# plt.show()


# Calculate confusion matrix for Decision Tree
cm_dtc = confusion_matrix(y_test, y_pred_test_dtc)
# plt.figure(figsize=(8, 8))
# sns.heatmap(cm_dtc, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
# plt.title('Confusion Matrix - Decision Tree')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.savefig("confusion matrix dtc.jpg", bbox_inches="tight", dpi=300)
# plt.show()

# Calculate confusion matrix for Random Forest
cm_rfc = confusion_matrix(y_test, y_pred_test_rfc)
# plt.figure(figsize=(8, 8))
# sns.heatmap(cm_rfc, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
# plt.title('Confusion Matrix - Random Forest')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.savefig("confusion matrix rfc.jpg", bbox_inches="tight", dpi=300)
# plt.show()


# Perform McNemar Test
def mcnemar_test(matrix):
    chi2, p, _, _ = chi2_contingency(matrix, correction=False)
    return chi2, p

# Apply McNemar Test for Decision Tree and Random Forest
chi2_dtc, p_dtc = mcnemar_test(cm_dtc)
chi2_rfc, p_rfc = mcnemar_test(cm_rfc)

print(f"McNemar Test Decision Tree: Chi2 = {chi2_dtc}, p-value = {p_dtc}")
print(f"McNemar Test Random Forest: Chi2 = {chi2_rfc}, p-value = {p_rfc}")




