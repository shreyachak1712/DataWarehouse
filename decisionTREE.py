import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree

# Load the dataset
df = pd.read_csv('/Users/Sanjay/Downloads/car_data.csv')

# Inspect the first few rows of the dataset
print(df.head())

# Preprocess the data

# Drop 'User ID' column as it's not relevant for prediction
df = df.drop(columns=['User ID'])

# Convert 'Gender' column to numerical (Female = 0, Male = 1)
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

# Check for missing values (if any)
print("Missing values:\n", df.isnull().sum())

# Define features (X) and target (y)
X = df[['Gender', 'Age', 'AnnualSalary']]  # Features: Gender, Age, Annual Salary
y = df['Purchased']  # Corrected Target: 'Purchased'

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree classifier (you can tweak parameters like max_depth)
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))  # Set the figure size
plot_tree(clf, feature_names=['Gender', 'Age', 'AnnualSalary'], class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree for Car Purchase Decision")
plt.show()
