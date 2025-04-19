from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
 
# Load dataset
data = load_iris()
X = data.data
y = data.target
 
# Split the data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Initialize the KNN model
model_v1 = KNeighborsClassifier()
# Train the model on the training data
model_v1.fit(X_train, y_train)
# Make predictions on the test data
y_pred = model_v1.predict(X_test)
# Check the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Hyperparameter values to try
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
# Create a GridSearchCV object with 5-fold cross-validation
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
# Train the model with different hyperparameters
grid_search.fit(X_train, y_train)
# Get the best model with the optimal n_neighbors
best_model = grid_search.best_estimator_
# Predict using the tuned model
y_pred_best = best_model.predict(X_test)
# Calculate and print the accuracy of the tuned model
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Tuned Model Accuracy: {accuracy_best}")

# Compare the accuracies
print(f"Original Model Accuracy: {accuracy}")
print(f"Tuned Model Accuracy: {accuracy_best}")

import joblib
# Save the original model (v1)
joblib.dump(model_v1, 'model_v1.pkl')
# Save the tuned model (v2)
joblib.dump(best_model, 'model_v2.pkl')
print("Models Saved as model_v1.pkl and model_v2.pkl")