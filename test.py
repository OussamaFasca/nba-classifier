import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def score_classifier(dataset, classifier, labels):
    """
    Performs 3-fold cross-validation to build a confusion matrix and calculate F1 score
    """
    kf = KFold(n_splits=3, random_state=50, shuffle=True)
    confusion_mat = np.zeros((2,2))
    f1 = 0
    for training_ids, test_ids in kf.split(dataset):
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        test_set = dataset[test_ids]
        test_labels = labels[test_ids]
        classifier.fit(training_set, training_labels)
        predicted_labels = classifier.predict(test_set)
        confusion_mat += confusion_matrix(test_labels, predicted_labels)
        f1 += f1_score(test_labels, predicted_labels)
    f1 /= 3
    print("Confusion Matrix:")
    print(confusion_mat)
    print(f"F1 Score: {f1}")
    return f1

# Load dataset
df = pd.read_csv("nba_logreg.csv")

# Extract names, labels, features names and values
names = df['Name'].values.tolist()  # players names
labels = df['TARGET_5Yrs'].values  # labels
paramset = df.drop(['TARGET_5Yrs', 'Name'], axis=1).columns.values
df_vals = df.drop(['TARGET_5Yrs', 'Name'], axis=1).values

# Replace NaN values (only present when no 3 points attempts have been performed by a player)
for x in np.argwhere(np.isnan(df_vals)):
    df_vals[x] = 0.0

# Normalize dataset
X = MinMaxScaler().fit_transform(df_vals)

# Set up the classifiers with hyperparameter grids
svc_params = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
svc = GridSearchCV(SVC(), svc_params, cv=3, scoring='f1')

lr_params = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
lr = GridSearchCV(LogisticRegression(solver='liblinear'), lr_params, cv=3, scoring='f1')

rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}
rf = GridSearchCV(RandomForestClassifier(), rf_params, cv=3, scoring='f1')

# List of classifiers to test
classifiers = [
    ("Support Vector Classifier", svc),
    ("Logistic Regression", lr),
    ("Random Forest", rf)
]

# Test each classifier and store results
results = []
for name, clf in classifiers:
    print(f"\nTesting {name}:")
    f1 = score_classifier(X, clf, labels)
    results.append((name, clf, f1))

# Find the best model
best_model = max(results, key=lambda x: x[2])
print(f"\nBest model: {best_model[0]} with F1 score: {best_model[2]}")
print("Best parameters:", best_model[1].best_params_)

# Train the best model on the full dataset
best_clf = best_model[1].best_estimator_
best_clf.fit(X, labels)

# Save the best model
joblib.dump(best_clf, f'best_classifier.pkl')