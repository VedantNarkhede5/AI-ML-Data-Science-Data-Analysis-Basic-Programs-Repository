import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample dataset
data = {
    'Age': [25, 30, 45, 35, 22, 27, 40, 50],
    'Salary': [50000, 60000, 80000, 65000, 45000, 52000, 70000, 90000],
    'Purchased': [0, 1, 1, 0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Features and target variable
X = df[['Age', 'Salary']]
y = df['Purchased']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------
# Decision Tree Classifier
# -----------------------------
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

# Evaluation for Decision Tree
print("Decision Tree Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# -----------------------------
# Naive Bayes Classifier
# -----------------------------
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)

# Evaluation for Naive Bayes
print("\nNaive Bayes Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))
