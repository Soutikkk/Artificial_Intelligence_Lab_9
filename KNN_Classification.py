# K-Nearest Neighbors (KNN) Classification

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create KNN Model
model = KNeighborsClassifier(n_neighbors=3)

# Train Model
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("KNN Classification")
print("------------------")
print("Accuracy:", accuracy * 100, "%")

# Predict a New Flower
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)

print("Predicted Flower:", iris.target_names[prediction[0]])
