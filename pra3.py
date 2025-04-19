from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# Load data
X, y = load_iris(return_X_y=True)
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Train model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

import joblib
# Save the model to a file
joblib.dump(model, 'model.pkl')

# Load the saved model
model = joblib.load('model.pkl')
# Use it to predict
y_pred = model.predict(X_test)
# Check accuracy
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))