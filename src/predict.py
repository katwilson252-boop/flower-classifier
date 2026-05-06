import joblib
from sklearn.datasets import load_iris

# Load trained model
model = joblib.load("models/model.pkl")

# Load dataset (for label names)
data = load_iris()

# Example input
print("Enter flower details:")

sepal_length = float(input("Sepal Length: "))
sepal_width = float(input("Sepal Width: "))
petal_length = float(input("Petal Length: "))
petal_width = float(input("Petal Width: "))

sample = [[sepal_length, sepal_width, petal_length, petal_width]]

# Predict
prediction = model.predict(sample)

print("🌸 Predicted Flower:", data.target_names[prediction][0])